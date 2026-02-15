"""
Stage 1 Fine-tuning: Qwen2.5-7B → KOSPI 퀀트 트레이딩 LLM

5개 금융 데이터셋을 통합하여 Qwen2.5-7B-Instruct를 파인튜닝합니다.

데이터 비중:
  - Finance-Instruct-500k     (40%) : 금융 전반 지식
  - Sujet-Finance-Instruct    (25%) : 금융 NLP 18개 태스크
  - KRX Won-Instruct          (20%) : 한국시장 전문 (핵심)
  - Trading Dataset v2         (10%) : 기술적 지표 → 매매 시그널
  - FinSentiment               (5%) : 뉴스 감성 분석

Usage:
    conda activate unsloth_env
    python finetune/training/stage1.py
    python finetune/training/stage1.py --epochs 3 --total-samples 200000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetune.config import QUANT_SYSTEM_PROMPT, DATASET_CONFIG
from finetune.data.loaders import load_and_normalize_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train(
    total_samples: int = 200_000,
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    max_seq_length: int = 2048,
    output_dir: Path = PROJECT_ROOT / "finetune" / "output_unsloth",
):
    """
    Stage 1 파인튜닝 파이프라인.

    6단계:
      1) Qwen2.5-7B-Instruct 4-bit 양자화 로드
      2) LoRA 어댑터 부착
      3) 5개 데이터셋 로드 및 ChatML 통합
      4) SFTTrainer 학습
      5) LoRA 어댑터 저장
      6) GGUF 변환 (Ollama용)
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    # Step 1: Load Model (4-bit quantized)
    logger.info("=" * 60)
    logger.info("Step 1: Loading Qwen2.5-7B-Instruct (4-bit)")
    logger.info("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Step 2: Attach LoRA Adapters
    logger.info("=" * 60)
    logger.info(f"Step 2: Attaching LoRA (rank={lora_rank}, alpha={lora_alpha})")
    logger.info("=" * 60)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable: {trainable:,} / {total_params:,} "
        f"({100 * trainable / total_params:.3f}%)"
    )

    # Step 3: Load & Prepare Data
    logger.info("=" * 60)
    logger.info(f"Step 3: Loading datasets (total target: {total_samples:,})")
    logger.info("=" * 60)

    examples = load_and_normalize_all(total_samples)
    dataset = Dataset.from_list(examples)

    # Step 4: Train
    logger.info("=" * 60)
    logger.info("Step 4: Training")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {grad_accum}")
    logger.info(f"  Effective batch: {batch_size * grad_accum}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Total steps: ~{len(dataset) * epochs // (batch_size * grad_accum)}")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=True,
    )

    logger.info("Training started...")
    stats = trainer.train()
    logger.info(f"Training complete! {stats}")

    # Step 5: Save
    logger.info("=" * 60)
    logger.info("Step 5: Saving LoRA adapter")
    logger.info("=" * 60)

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    config = {
        "base_model": "unsloth/Qwen2.5-7B-Instruct",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_seq_length": max_seq_length,
        "total_samples": len(dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "dataset_config": {
            k: {"ratio": v["ratio"], "description": v["description"]}
            for k, v in DATASET_CONFIG.items()
        },
    }
    with open(final_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"LoRA adapter saved to: {final_dir}")

    # Step 6: Save GGUF
    logger.info("=" * 60)
    logger.info("Step 6: Exporting to GGUF (Q4_K_M) for Ollama")
    logger.info("=" * 60)

    try:
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(
            str(gguf_dir), tokenizer, quantization_method="q4_k_m",
        )
        logger.info(f"GGUF saved to: {gguf_dir}")
        _create_modelfile(gguf_dir)
    except Exception as e:
        logger.warning(f"GGUF export failed (can do manually later): {e}")

    # Done
    print("\n" + "=" * 60)
    print("STAGE 1 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  LoRA adapter : {final_dir}")
    print(f"  GGUF model   : {output_dir / 'gguf'}")
    print(f"\nNext steps:")
    print(f"  1. ollama create qwen2.5-kospi-ft -f {output_dir}/gguf/Modelfile")
    print(f"  2. ollama run qwen2.5-kospi-ft")
    print(f"  3. python finetune/training/stage2.py")


def _create_modelfile(gguf_dir: Path) -> None:
    """Ollama Modelfile 생성."""
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        logger.warning("No GGUF file found, skipping Modelfile creation")
        return

    gguf_path = gguf_files[0]
    # 시스템 프롬프트를 config에서 가져옴
    system_text = QUANT_SYSTEM_PROMPT.replace('"""', '\\"\\"\\"')

    modelfile_content = f"""\
FROM {gguf_path.resolve()}

PARAMETER temperature 0.3
PARAMETER num_predict 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

SYSTEM \"\"\"{system_text}\"\"\"
"""
    modelfile_path = gguf_dir / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    logger.info(f"Modelfile created: {modelfile_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Unsloth QLoRA fine-tuning (Qwen2.5-7B → KOSPI 퀀트 LLM)"
    )
    parser.add_argument("--total-samples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "finetune" / "output_unsloth",
    )
    args = parser.parse_args()

    train(
        total_samples=args.total_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
