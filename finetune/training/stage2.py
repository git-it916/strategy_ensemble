"""
Stage 2 Fine-tuning: Synthetic Data로 전략 특화 학습

Stage 1에서 학습된 LoRA 어댑터 위에
우리 전략에 맞는 synthetic 데이터로 추가 학습합니다.

핵심 차이점 (vs Stage 1):
  - 데이터: 범용 금융 → 전략 전용 synthetic data (1,500건)
  - 베이스: 원본 Qwen2.5 → Stage 1 LoRA가 병합된 모델
  - 학습률: 2e-4 → 5e-5 (이미 학습된 모델이므로 낮게)
  - Epochs: 2 → 3~5 (데이터가 적으므로 더 많이 반복)

Usage:
    # 1단계: synthetic 데이터 생성
    python finetune/data/synthetic.py

    # 2단계: Stage 2 학습
    conda activate unsloth_env
    python finetune/training/stage2.py
    python finetune/training/stage2.py --epochs 5 --lr 3e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetune.config import QUANT_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 기본 경로
SYNTHETIC_DATA_PATH = PROJECT_ROOT / "finetune" / "data" / "synthetic_data.json"
STAGE1_ADAPTER_DIR = PROJECT_ROOT / "finetune" / "output_unsloth" / "final"
OUTPUT_DIR = PROJECT_ROOT / "finetune" / "output_stage2"


def train_stage2(
    synthetic_data_path: Path = SYNTHETIC_DATA_PATH,
    stage1_adapter_dir: Path = STAGE1_ADAPTER_DIR,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 5e-5,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    max_seq_length: int = 2048,
    output_dir: Path = OUTPUT_DIR,
):
    """
    Stage 2 학습 파이프라인.

    Stage 1 LoRA를 베이스 모델에 병합한 후, 새로운 LoRA를 부착하여
    synthetic 데이터로 추가 학습합니다.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    # Step 1: Load Synthetic Data
    logger.info("=" * 60)
    logger.info("Stage 2 - Step 1: Loading synthetic data")
    logger.info("=" * 60)

    if not synthetic_data_path.exists():
        logger.error(
            f"Synthetic data not found: {synthetic_data_path}\n"
            f"Run first: python finetune/data/synthetic.py"
        )
        sys.exit(1)

    with open(synthetic_data_path, "r", encoding="utf-8") as f:
        synthetic_examples = json.load(f)

    logger.info(f"Loaded {len(synthetic_examples)} synthetic examples")

    from collections import Counter
    sources = Counter(ex["source"] for ex in synthetic_examples)
    for src, cnt in sources.most_common():
        logger.info(f"  {src}: {cnt}")

    # Step 2: Load Stage 1 Model (base + LoRA merged)
    logger.info("=" * 60)
    logger.info("Stage 2 - Step 2: Loading base model + Stage 1 LoRA")
    logger.info("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if stage1_adapter_dir.exists():
        from peft import PeftModel
        logger.info(f"Loading Stage 1 LoRA from: {stage1_adapter_dir}")
        model = PeftModel.from_pretrained(model, str(stage1_adapter_dir))
        model = model.merge_and_unload()
        logger.info("Stage 1 LoRA merged into base model")
    else:
        logger.warning(
            f"Stage 1 adapter not found at {stage1_adapter_dir}. "
            f"Training from base model."
        )

    # Step 3: Attach New LoRA for Stage 2
    logger.info("=" * 60)
    logger.info(f"Stage 2 - Step 3: Attaching new LoRA (rank={lora_rank})")
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

    # Step 4: Train
    logger.info("=" * 60)
    logger.info("Stage 2 - Step 4: Training on synthetic data")
    logger.info(f"  Data: {len(synthetic_examples)} examples")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {grad_accum}")
    logger.info(f"  Effective batch: {batch_size * grad_accum}")
    logger.info(f"  Learning rate: {learning_rate}")
    est_steps = len(synthetic_examples) * epochs // (batch_size * grad_accum)
    logger.info(f"  Estimated steps: ~{est_steps}")
    logger.info("=" * 60)

    dataset = Dataset.from_list(synthetic_examples)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
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

    # Step 5: Save Stage 2 LoRA
    logger.info("=" * 60)
    logger.info("Stage 2 - Step 5: Saving LoRA adapter")
    logger.info("=" * 60)

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    config = {
        "base_model": "unsloth/Qwen2.5-7B-Instruct",
        "stage1_adapter": str(stage1_adapter_dir),
        "stage": 2,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_seq_length": max_seq_length,
        "synthetic_examples": len(synthetic_examples),
        "epochs": epochs,
        "learning_rate": learning_rate,
    }
    with open(final_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Stage 2 LoRA saved to: {final_dir}")

    # Step 6: Export GGUF
    logger.info("=" * 60)
    logger.info("Stage 2 - Step 6: Exporting to GGUF")
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
        logger.warning(
            f"GGUF auto-export failed: {e}\n"
            f"Manual conversion:\n"
            f"  1. python llama.cpp/convert_hf_to_gguf.py {gguf_dir} "
            f"--outfile {gguf_dir}/model-f16.gguf --outtype f16\n"
            f"  2. llama.cpp/build/bin/llama-quantize "
            f"{gguf_dir}/model-f16.gguf {gguf_dir}/model-Q4_K_M.gguf Q4_K_M"
        )

    # Done
    print("\n" + "=" * 60)
    print("STAGE 2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Stage 2 LoRA : {final_dir}")
    print(f"  GGUF model   : {output_dir / 'gguf'}")
    print(f"\nNext steps:")
    print(f"  1. ollama create qwen2.5-kospi-ft-s2 -f {output_dir}/gguf/Modelfile")
    print(f"  2. ollama run qwen2.5-kospi-ft-s2")


def _create_modelfile(gguf_dir: Path) -> None:
    """Ollama Modelfile 생성."""
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        logger.warning("No GGUF file found, skipping Modelfile creation")
        return

    gguf_path = gguf_files[0]
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
        description="Stage 2: Fine-tuning with synthetic strategy data"
    )
    parser.add_argument(
        "--synthetic-data", type=Path, default=SYNTHETIC_DATA_PATH,
    )
    parser.add_argument(
        "--stage1-adapter", type=Path, default=STAGE1_ADAPTER_DIR,
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    train_stage2(
        synthetic_data_path=args.synthetic_data,
        stage1_adapter_dir=args.stage1_adapter,
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
