"""
HuggingFace 데이터셋 로더.

5개 데이터셋을 다운로드하고 ChatML 형식으로 통합합니다.
"""

from __future__ import annotations

import logging
import random

from finetune.config import QUANT_SYSTEM_PROMPT, DATASET_CONFIG

logger = logging.getLogger(__name__)


def load_and_normalize_all(total_samples: int) -> list[dict]:
    """
    5개 데이터셋을 모두 로드하여 ChatML 형식으로 통합하는 메인 데이터 로딩 함수.

    각 데이터셋을 DATASET_CONFIG에 정의된 비율(ratio)에 따라 샘플링한 뒤,
    전체를 하나로 합쳐서 랜덤 셔플합니다.
    """
    all_examples = []

    for name, config in DATASET_CONFIG.items():
        n_samples = int(total_samples * config["ratio"])
        logger.info(
            f"Loading {name}: {config['hf_id']} "
            f"(target: {n_samples} samples, {config['ratio']:.0%})"
        )

        try:
            examples = _load_single_dataset(name, config["hf_id"], n_samples)
            all_examples.extend(examples)
            logger.info(f"  → {len(examples)} examples loaded")
        except Exception as e:
            logger.error(f"  → Failed to load {name}: {e}")
            continue

    random.seed(42)
    random.shuffle(all_examples)

    logger.info(f"\nTotal training examples: {len(all_examples)}")
    _print_dataset_stats(all_examples)

    return all_examples


def _load_single_dataset(name: str, hf_id: str, n_samples: int) -> list[dict]:
    """데이터셋 이름에 따라 적절한 로더 함수를 호출하는 라우터."""
    loaders = {
        "finance_instruct": _load_finance_instruct,
        "sujet_finance": _load_sujet_finance,
        "krx_won": _load_krx_won,
        "trading_signals": _load_trading_signals,
        "fin_sentiment": _load_fin_sentiment,
    }
    loader = loaders.get(name)
    if not loader:
        raise ValueError(f"Unknown dataset: {name}")
    return loader(hf_id, n_samples)


def _load_finance_instruct(hf_id: str, n_samples: int) -> list[dict]:
    """Finance-Instruct-500k: system/user/assistant 컬럼."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    examples = []
    for row in ds:
        system = row.get("system", "") or QUANT_SYSTEM_PROMPT
        user = row.get("user", "")
        assistant = row.get("assistant", "")
        if not user or not assistant:
            continue
        examples.append(_make_chatml(system, user, assistant, "finance_instruct"))
    return examples


def _load_sujet_finance(hf_id: str, n_samples: int) -> list[dict]:
    """Sujet-Finance: system_prompt/user_prompt/answer 컬럼."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    examples = []
    for row in ds:
        system = row.get("system_prompt", "") or QUANT_SYSTEM_PROMPT
        user = row.get("user_prompt", "") or row.get("inputs", "")
        assistant = row.get("answer", "")
        if not user or not assistant:
            continue
        examples.append(_make_chatml(system, user, assistant, "sujet_finance"))
    return examples


def _load_krx_won(hf_id: str, n_samples: int) -> list[dict]:
    """KRX Won-Instruct: prompt/original_response 컬럼 (한국어)."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    examples = []
    for row in ds:
        user = row.get("prompt", "")
        assistant = row.get("original_response", "")
        if not user or not assistant:
            continue
        system = (
            "당신은 한국 금융시장 전문 분석가입니다. "
            "한국거래소(KRX) 데이터를 기반으로 정확하고 상세한 분석을 제공합니다."
        )
        examples.append(_make_chatml(system, user, assistant, "krx_won"))
    return examples


def _load_trading_signals(hf_id: str, n_samples: int) -> list[dict]:
    """Trading Dataset v2: instruction/input/output 컬럼."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    examples = []
    for row in ds:
        instruction = row.get("instruction", "")
        input_data = row.get("input", "")
        output = row.get("output", "")
        if not output:
            continue
        user = instruction
        if input_data:
            user = f"{instruction}\n\n{input_data}"
        system = (
            "You are a quantitative trading analyst. "
            "Analyze the given technical indicators and generate a trading signal "
            "with clear reasoning."
        )
        examples.append(_make_chatml(system, user, output, "trading_signals"))
    return examples


def _load_fin_sentiment(hf_id: str, n_samples: int) -> list[dict]:
    """FinGPT Sentiment: instruction/input/output 컬럼."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    examples = []
    for row in ds:
        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        output = row.get("output", "")
        if not input_text or not output:
            continue
        user = instruction
        if input_text:
            user = f"{instruction}\n\n{input_text}"
        system = (
            "You are a financial sentiment analyst. "
            "Classify the sentiment of financial news and provide reasoning."
        )
        examples.append(_make_chatml(system, user, output, "fin_sentiment"))
    return examples


def _make_chatml(system: str, user: str, assistant: str, source: str) -> dict:
    """ChatML 형식으로 변환합니다."""
    text = (
        f"<|im_start|>system\n{system.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{user.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant.strip()}<|im_end|>"
    )
    return {"text": text, "source": source}


def _print_dataset_stats(examples: list[dict]) -> None:
    """데이터셋 구성 통계를 로그로 출력합니다."""
    from collections import Counter

    counts = Counter(ex["source"] for ex in examples)
    total = len(examples)
    logger.info("\n=== Dataset Composition ===")
    for source, count in counts.most_common():
        pct = count / total * 100
        desc = DATASET_CONFIG.get(source, {}).get("description", "")
        logger.info(f"  {source:20s}: {count:>6,} ({pct:5.1f}%) - {desc}")
    logger.info(f"  {'TOTAL':20s}: {total:>6,}")
