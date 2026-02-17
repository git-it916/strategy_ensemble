"""
Stage 3 DPO (Direct Preference Optimization) Fine-tuning.

Stage 2에서 SFT로 학습된 모델을 DPO로 추가 정렬합니다.
DPO는 reward model 없이 preference pair(chosen/rejected)만으로
모델이 "좋은 응답"을 선호하도록 학습합니다.

핵심:
  - chosen: Stage 2 synthetic data의 정답 응답
  - rejected: 의도적으로 결함이 있는 응답 (스키마 위반, 전략 오류 등)
  - beta: KL divergence 패널티 (높을수록 reference model에 가깝게 유지)

Usage:
    # 1단계: DPO 데이터 생성
    python finetune/data/gen_dpo_pairs.py

    # 2단계: DPO 학습
    conda activate unsloth_env
    python finetune/training/stage3_dpo.py
    python finetune/training/stage3_dpo.py --beta 0.05 --epochs 2 --lr 5e-7
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
DPO_DATA_PATH = PROJECT_ROOT / "finetune" / "data" / "dpo_pairs.json"
STAGE2_ADAPTER_DIR = PROJECT_ROOT / "finetune" / "output_stage2" / "final"
OUTPUT_DIR = PROJECT_ROOT / "finetune" / "output_stage3"


def train_stage3(
    dpo_data_path: Path = DPO_DATA_PATH,
    stage2_adapter_dir: Path = STAGE2_ADAPTER_DIR,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 5e-7,
    beta: float = 0.1,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    output_dir: Path = OUTPUT_DIR,
    skip_stage2: bool = False,
):
    """
    Stage 3 DPO 학습 파이프라인.

    Stage 2 LoRA를 로드하고, DPO preference pair로 추가 정렬합니다.
    """
    from unsloth import FastLanguageModel
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    # Step 1: Load DPO Data
    logger.info("=" * 60)
    logger.info("Stage 3 DPO - Step 1: Loading preference pairs")
    logger.info("=" * 60)

    if not dpo_data_path.exists():
        logger.error(
            f"DPO data not found: {dpo_data_path}\n"
            f"Run first: python finetune/data/gen_dpo_pairs.py"
        )
        sys.exit(1)

    with open(dpo_data_path, "r", encoding="utf-8") as f:
        dpo_pairs = json.load(f)

    logger.info(f"Loaded {len(dpo_pairs)} DPO preference pairs")

    from collections import Counter
    reject_types = Counter(p["reject_type"] for p in dpo_pairs)
    for rt, cnt in reject_types.most_common():
        logger.info(f"  {rt}: {cnt}")

    # Step 2: Load Model
    logger.info("=" * 60)
    logger.info("Stage 3 DPO - Step 2: Loading model")
    logger.info("=" * 60)

    adapter_dir = Path("/nonexistent") if skip_stage2 else stage2_adapter_dir

    if adapter_dir.exists():
        logger.info(f"Loading Stage 2 LoRA from: {adapter_dir}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=max_length,
            dtype=None,
            load_in_4bit=True,
        )
        model.gradient_checkpointing_enable()
        logger.info("Stage 2 LoRA loaded — continuing with DPO training")
    else:
        logger.warning(
            f"Stage 2 adapter not found at {adapter_dir}. "
            f"Training from base model with fresh LoRA."
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-7B-Instruct",
            max_seq_length=max_length,
            dtype=None,
            load_in_4bit=True,
        )
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

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable: {trainable:,} / {total_params:,} "
        f"({100 * trainable / total_params:.3f}%)"
    )

    # Step 3: Prepare Dataset
    logger.info("=" * 60)
    logger.info("Stage 3 DPO - Step 3: Preparing dataset")
    logger.info("=" * 60)

    # DPOTrainer expects: prompt, chosen, rejected columns
    dataset_rows = [
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in dpo_pairs
    ]
    dataset = Dataset.from_list(dataset_rows)
    logger.info(f"Dataset ready: {len(dataset)} pairs")

    # Step 4: Train with DPO
    logger.info("=" * 60)
    logger.info("Stage 3 DPO - Step 4: DPO Training")
    logger.info(f"  Pairs: {len(dataset)}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {grad_accum}")
    logger.info(f"  Effective batch: {batch_size * grad_accum}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Beta (KL penalty): {beta}")
    logger.info(f"  Max length: {max_length}")
    logger.info(f"  Max prompt length: {max_prompt_length}")
    est_steps = len(dataset) * epochs // (batch_size * grad_accum)
    logger.info(f"  Estimated steps: ~{est_steps}")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        # DPO-specific
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        loss_type="sigmoid",  # standard DPO loss
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # use implicit reference (no separate ref model in 4-bit)
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("DPO Training started...")
    stats = trainer.train()
    logger.info(f"DPO Training complete! {stats}")

    # Step 5: Save Stage 3 LoRA
    logger.info("=" * 60)
    logger.info("Stage 3 DPO - Step 5: Saving LoRA adapter")
    logger.info("=" * 60)

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    config = {
        "base_model": "unsloth/Qwen2.5-7B-Instruct",
        "stage2_adapter": str(stage2_adapter_dir),
        "stage": 3,
        "method": "DPO",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_length": max_length,
        "dpo_pairs": len(dpo_pairs),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "beta": beta,
    }
    with open(final_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Stage 3 LoRA saved to: {final_dir}")

    # Step 6: Export GGUF
    logger.info("=" * 60)
    logger.info("Stage 3 DPO - Step 6: Exporting to GGUF")
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
            f"  1. Merge LoRA: python merge_lora.py (or use unsloth merge)\n"
            f"  2. python llama.cpp/convert_hf_to_gguf.py {final_dir} "
            f"--outfile {gguf_dir}/model-f16.gguf --outtype f16\n"
            f"  3. llama.cpp/build/bin/llama-quantize "
            f"{gguf_dir}/model-f16.gguf {gguf_dir}/model-Q4_K_M.gguf Q4_K_M"
        )

    # Done
    print("\n" + "=" * 60)
    print("STAGE 3 DPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Stage 3 LoRA : {final_dir}")
    print(f"  GGUF model   : {output_dir / 'gguf'}")
    print(f"\nNext steps:")
    print(f"  1. ollama create qwen2.5-kospi-ft-s3 -f {output_dir}/gguf/Modelfile")
    print(f"  2. ollama run qwen2.5-kospi-ft-s3")
    print(f"  3. python finetune/evaluate.py --base-model qwen2.5-kospi-ft-s2 "
          f"--ft-model qwen2.5-kospi-ft-s3")


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
        description="Stage 3: DPO fine-tuning for preference alignment"
    )
    parser.add_argument(
        "--dpo-data", type=Path, default=DPO_DATA_PATH,
        help="Path to DPO pairs JSON",
    )
    parser.add_argument(
        "--stage2-adapter", type=Path, default=STAGE2_ADAPTER_DIR,
        help="Stage 2 LoRA adapter directory",
    )
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Skip Stage 2 adapter, train from base model directly")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of DPO epochs (1-2 recommended)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size (DPO needs 2x memory)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate (should be very low for DPO)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta: KL penalty strength (0.05-0.5)")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max total sequence length")
    parser.add_argument("--max-prompt-length", type=int, default=1024,
                        help="Max prompt length")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    train_stage3(
        dpo_data_path=args.dpo_data,
        stage2_adapter_dir=args.stage2_adapter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        output_dir=args.output_dir,
        skip_stage2=args.skip_stage2,
    )


if __name__ == "__main__":
    main()
