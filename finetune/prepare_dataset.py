"""
Prepare SFT Dataset from Reasoning Logs

Converts the JSON reasoning logs (logs/reasoning/YYYY-MM-DD/*.json)
into instruction-tuning format for LoRA fine-tuning.

Each reasoning log becomes a training example:
  - system: SIGNAL_SYSTEM_PROMPT or ORCHESTRATOR_SYSTEM_PROMPT
  - input: the market context that was provided
  - output: the model's JSON response (signals + reasoning)

Usage:
    python finetune/prepare_dataset.py
    python finetune/prepare_dataset.py --min-confidence 0.5 --min-signals 3
    python finetune/prepare_dataset.py --task signal_generation --output finetune/data/signal_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Task -> system prompt mapping
SYSTEM_PROMPTS = {
    "signal_generation": """\
You are a Korean equity market quantitative analyst.
You analyze market data and generate trading signals for KOSPI200 stocks.

You MUST respond in valid JSON format with this exact structure:
{
  "market_context": "brief market analysis in Korean",
  "analysis": "detailed reasoning for signals",
  "factors_considered": ["factor1", "factor2"],
  "signals": [
    {"ticker": "005930", "score": 0.7, "rationale": "reason for this score"}
  ],
  "confidence": 0.85,
  "regime_assessment": "bull|bear|sideways"
}

Rules:
- score range: -1.0 (strong sell) to +1.0 (strong buy), 0.0 = neutral
- Only include stocks with |score| >= 0.2 (skip neutral stocks)
- Consider: price momentum, mean reversion, volatility, volume, fundamentals
- Provide rationale in Korean for each signal
- confidence: 0.0 to 1.0, your overall confidence in the analysis""",

    "ensemble_orchestration": """\
You are a senior portfolio manager overseeing multiple quantitative strategies for Korean equities.

You receive signals from 8 alpha strategies (4 rule-based, 3 ML, 1 LLM-based) and must decide which signals to trust and how to weight them.

You MUST respond in valid JSON format with this exact structure:
{
  "reasoning": {
    "market_assessment": "current market conditions analysis in Korean",
    "strategy_evaluation": {
      "strategy_name": {"trust_level": 0.8, "rationale": "why trust/distrust"}
    },
    "portfolio_logic": "overall portfolio construction reasoning",
    "risk_considerations": "key risks being managed"
  },
  "final_signals": [
    {"ticker": "005930", "score": 0.65, "primary_driver": "strategy_name", "rationale": "reason"}
  ],
  "strategy_weights": {"strategy_name": 0.15},
  "confidence": 0.80,
  "position_sizing_notes": "any special sizing considerations"
}

Rules:
- Think step by step (chain of thought)
- Evaluate each strategy's recent performance
- Consider current market regime
- Identify conflicts between strategies and resolve them
- score range: -1.0 to +1.0
- Only include stocks with |score| >= 0.15
- Your reasoning should be thorough and in Korean""",
}


def load_reasoning_logs(
    log_dir: Path,
    task_filter: str | None = None,
) -> list[dict]:
    """Load all reasoning logs from disk."""
    logs = []
    if not log_dir.exists():
        logger.warning(f"Log directory not found: {log_dir}")
        return logs

    for day_dir in sorted(log_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        for json_file in sorted(day_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    entry = json.load(f)

                if task_filter and entry.get("task") != task_filter:
                    continue

                entry["_source_file"] = str(json_file)
                logs.append(entry)
            except Exception as e:
                logger.warning(f"Failed to read {json_file}: {e}")

    logger.info(f"Loaded {len(logs)} reasoning logs from {log_dir}")
    return logs


def filter_high_quality(
    logs: list[dict],
    min_confidence: float = 0.0,
    min_signals: int = 1,
    max_latency_ms: int | None = None,
) -> list[dict]:
    """Filter logs for training quality."""
    filtered = []
    for entry in logs:
        confidence = entry.get("confidence")
        if confidence is not None and confidence < min_confidence:
            continue

        signals = entry.get("signals", [])
        if len(signals) < min_signals:
            continue

        latency = entry.get("latency_ms")
        if max_latency_ms and latency and latency > max_latency_ms:
            continue

        # Validate signals have required fields
        valid_signals = [
            s for s in signals
            if "ticker" in s and "score" in s
        ]
        if len(valid_signals) < min_signals:
            continue

        filtered.append(entry)

    logger.info(
        f"Filtered: {len(filtered)}/{len(logs)} logs passed quality check "
        f"(min_confidence={min_confidence}, min_signals={min_signals})"
    )
    return filtered


def log_to_sft_example(entry: dict) -> dict | None:
    """
    Convert a single reasoning log into an SFT training example.

    Format: ChatML-compatible instruction tuning
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    task = entry.get("task", "signal_generation")
    system_prompt = SYSTEM_PROMPTS.get(task)
    if not system_prompt:
        return None

    # Reconstruct the user prompt from metadata
    metadata = entry.get("metadata", {})
    reasoning = entry.get("reasoning", {})

    # Build a compact user context from what's available
    user_parts = []

    timestamp = entry.get("timestamp", "")
    if timestamp:
        date_str = timestamp[:10]
        user_parts.append(f"날짜: {date_str}")

    market_ctx = reasoning.get("market_context", "")
    if market_ctx:
        user_parts.append(f"시장 컨텍스트: {market_ctx}")

    # If metadata has prompt or context info, include it
    if metadata.get("prompt_context"):
        user_parts.append(metadata["prompt_context"])

    if metadata.get("regime"):
        user_parts.append(f"현재 레짐: {metadata['regime']}")

    if not user_parts:
        user_parts.append("KOSPI200 종목을 분석하고 매매 시그널을 생성하세요.")

    user_content = "\n".join(user_parts)

    # Build the ideal assistant response (the ground truth we want the model to learn)
    if task == "signal_generation":
        assistant_response = {
            "market_context": reasoning.get("market_context", ""),
            "analysis": reasoning.get("analysis", ""),
            "factors_considered": reasoning.get("factors_considered", []),
            "signals": entry.get("signals", []),
            "confidence": entry.get("confidence", 0.5),
            "regime_assessment": metadata.get("regime_assessment", "sideways"),
        }
    elif task == "ensemble_orchestration":
        assistant_response = {
            "reasoning": reasoning,
            "final_signals": entry.get("signals", []),
            "strategy_weights": metadata.get("strategy_weights", {}),
            "confidence": entry.get("confidence", 0.5),
            "position_sizing_notes": metadata.get("position_sizing_notes", ""),
        }
    else:
        return None

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(
                assistant_response, ensure_ascii=False, indent=2
            )},
        ],
        "task": task,
        "source_timestamp": entry.get("timestamp"),
        "original_confidence": entry.get("confidence"),
    }


def prepare_dataset(
    log_dir: Path,
    output_path: Path,
    task_filter: str | None = None,
    min_confidence: float = 0.0,
    min_signals: int = 1,
    max_latency_ms: int | None = None,
    train_ratio: float = 0.9,
) -> dict:
    """
    Full pipeline: load logs -> filter -> convert -> save JSONL.

    Returns statistics dict.
    """
    # Load
    logs = load_reasoning_logs(log_dir, task_filter=task_filter)
    if not logs:
        logger.warning("No reasoning logs found. Run the trading system first!")
        return {"total": 0, "train": 0, "eval": 0}

    # Filter
    logs = filter_high_quality(
        logs,
        min_confidence=min_confidence,
        min_signals=min_signals,
        max_latency_ms=max_latency_ms,
    )

    # Convert to SFT examples
    examples = []
    skipped = 0
    for entry in logs:
        example = log_to_sft_example(entry)
        if example:
            examples.append(example)
        else:
            skipped += 1

    logger.info(f"Converted {len(examples)} examples ({skipped} skipped)")

    if not examples:
        logger.warning("No valid training examples generated")
        return {"total": 0, "train": 0, "eval": 0}

    # Split train/eval
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = output_path.with_suffix(".train.jsonl")
    eval_path = output_path.with_suffix(".eval.jsonl")

    _write_jsonl(train_examples, train_path)
    _write_jsonl(eval_examples, eval_path)

    # Also write a combined file
    _write_jsonl(examples, output_path)

    stats = {
        "total": len(examples),
        "train": len(train_examples),
        "eval": len(eval_examples),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "combined_path": str(output_path),
        "task_distribution": _count_tasks(examples),
    }

    logger.info(f"Dataset saved: {stats}")
    return stats


def _write_jsonl(examples: list[dict], path: Path) -> None:
    """Write examples to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"  Wrote {len(examples)} examples to {path}")


def _count_tasks(examples: list[dict]) -> dict[str, int]:
    """Count examples by task type."""
    counts: dict[str, int] = {}
    for ex in examples:
        task = ex.get("task", "unknown")
        counts[task] = counts.get(task, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Convert reasoning logs to SFT training data"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "reasoning",
        help="Path to reasoning log directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "finetune" / "data" / "sft_dataset.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["signal_generation", "ensemble_orchestration"],
        help="Filter by task type",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--min-signals",
        type=int,
        default=1,
        help="Minimum number of valid signals per example",
    )
    parser.add_argument(
        "--max-latency",
        type=int,
        default=None,
        help="Max latency in ms (exclude slow responses)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train/eval split ratio",
    )

    args = parser.parse_args()

    stats = prepare_dataset(
        log_dir=args.log_dir,
        output_path=args.output,
        task_filter=args.task,
        min_confidence=args.min_confidence,
        min_signals=args.min_signals,
        max_latency_ms=args.max_latency,
        train_ratio=args.train_ratio,
    )

    print("\n=== Dataset Preparation Complete ===")
    print(f"Total examples: {stats.get('total', 0)}")
    print(f"Train: {stats.get('train', 0)}")
    print(f"Eval: {stats.get('eval', 0)}")
    if stats.get("task_distribution"):
        print(f"Tasks: {stats['task_distribution']}")


if __name__ == "__main__":
    main()
