"""
Evaluate Base vs Fine-tuned Model

Compares the base Qwen2.5:32B with the fine-tuned version on:
1. JSON validity rate - does the model return valid JSON?
2. Schema compliance - does the JSON match the expected structure?
3. Signal quality - are scores reasonable (-1 to 1)?
4. Reasoning quality - does the model provide Korean rationale?
5. Latency - response time comparison
6. ROUGE score - overlap with original reasoning logs (if available)

Usage:
    python finetune/evaluate.py
    python finetune/evaluate.py --base-model qwen2.5:32b --ft-model qwen2.5-kospi-ft
    python finetune/evaluate.py --n-samples 50 --output finetune/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Prompts
# =============================================================================

EVAL_SYSTEM_PROMPT = """\
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
- Provide rationale in Korean for each signal
- confidence: 0.0 to 1.0"""

EVAL_PROMPTS = [
    {
        "id": "bull_market",
        "prompt": (
            "10개 KOSPI200 종목을 분석해주세요. 날짜: 2024-12-15\n\n"
            "=== 시장 컨텍스트 ===\n"
            "최근 30일 평균 일일수익률: +0.15%, 변동성(연율): 18.2%, "
            "누적수익률: +4.5%, 상승일수: 14/20일\n\n"
            "=== 종목별 최근 데이터 ===\n"
            "ticker | close | change_1d | change_5d | volume\n"
            "-----------------------------------------------\n"
            "005930 | 72,000 | +1.2% | +3.5% | 15,234,567\n"
            "000660 | 185,000 | +2.1% | +5.2% | 3,456,789\n"
            "035420 | 215,000 | -0.5% | +1.8% | 1,234,567\n"
            "005380 | 198,000 | +0.8% | +2.1% | 2,345,678\n"
            "051910 | 420,000 | +1.5% | +4.2% | 567,890\n\n"
            "=== 현재 레짐 ===\nbull\n\n"
            "각 종목에 대해 매매 시그널을 생성하세요."
        ),
    },
    {
        "id": "bear_market",
        "prompt": (
            "10개 KOSPI200 종목을 분석해주세요. 날짜: 2024-08-05\n\n"
            "=== 시장 컨텍스트 ===\n"
            "최근 30일 평균 일일수익률: -0.25%, 변동성(연율): 28.5%, "
            "누적수익률: -7.2%, 상승일수: 7/20일\n\n"
            "=== 종목별 최근 데이터 ===\n"
            "ticker | close | change_1d | change_5d | volume\n"
            "-----------------------------------------------\n"
            "005930 | 65,000 | -2.5% | -8.1% | 25,678,901\n"
            "000660 | 155,000 | -3.2% | -12.3% | 8,901,234\n"
            "035420 | 180,000 | -1.8% | -5.5% | 3,456,789\n"
            "005380 | 175,000 | -1.2% | -4.8% | 4,567,890\n"
            "051910 | 350,000 | -2.8% | -9.5% | 1,234,567\n\n"
            "=== 현재 레짐 ===\nbear\n\n"
            "각 종목에 대해 매매 시그널을 생성하세요."
        ),
    },
    {
        "id": "sideways_market",
        "prompt": (
            "10개 KOSPI200 종목을 분석해주세요. 날짜: 2024-10-15\n\n"
            "=== 시장 컨텍스트 ===\n"
            "최근 30일 평균 일일수익률: +0.02%, 변동성(연율): 15.1%, "
            "누적수익률: +0.3%, 상승일수: 10/20일\n\n"
            "=== 종목별 최근 데이터 ===\n"
            "ticker | close | change_1d | change_5d | volume\n"
            "-----------------------------------------------\n"
            "005930 | 68,000 | +0.3% | -0.5% | 12,345,678\n"
            "000660 | 170,000 | -0.2% | +0.8% | 2,345,678\n"
            "035420 | 200,000 | +0.1% | -0.3% | 987,654\n"
            "005380 | 190,000 | -0.5% | +0.2% | 1,567,890\n"
            "051910 | 390,000 | +0.2% | -1.1% | 456,789\n\n"
            "=== 현재 레짐 ===\nsideways\n\n"
            "각 종목에 대해 매매 시그널을 생성하세요."
        ),
    },
]


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_response(response_text: str, latency_ms: int) -> dict[str, Any]:
    """Evaluate a single model response."""
    result: dict[str, Any] = {
        "latency_ms": latency_ms,
        "json_valid": False,
        "schema_valid": False,
        "n_signals": 0,
        "scores_in_range": True,
        "has_korean_rationale": False,
        "has_confidence": False,
        "has_regime": False,
        "has_analysis": False,
        "response_length": len(response_text),
    }

    # Parse JSON
    parsed = _try_parse_json(response_text)
    if parsed is None:
        return result

    result["json_valid"] = True

    # Schema check
    signals = parsed.get("signals", [])
    result["n_signals"] = len(signals)
    result["has_confidence"] = "confidence" in parsed
    result["has_regime"] = "regime_assessment" in parsed
    result["has_analysis"] = bool(parsed.get("analysis"))

    # Validate signals
    if signals:
        all_valid = True
        has_korean = False

        for sig in signals:
            if "ticker" not in sig or "score" not in sig:
                all_valid = False
                continue
            try:
                score = float(sig["score"])
                if not (-1.0 <= score <= 1.0):
                    result["scores_in_range"] = False
            except (ValueError, TypeError):
                result["scores_in_range"] = False

            rationale = sig.get("rationale", "")
            if rationale and _contains_korean(rationale):
                has_korean = True

        result["schema_valid"] = all_valid
        result["has_korean_rationale"] = has_korean
    else:
        result["schema_valid"] = "market_context" in parsed

    # Confidence value check
    conf = parsed.get("confidence")
    if conf is not None:
        try:
            result["confidence_value"] = float(conf)
        except (ValueError, TypeError):
            pass

    result["parsed_json"] = parsed
    return result


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON from response text."""
    import re

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _contains_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    return any("\uac00" <= c <= "\ud7a3" for c in text)


def compute_aggregate_metrics(
    results: list[dict],
) -> dict[str, float]:
    """Compute aggregate metrics from evaluation results."""
    n = len(results)
    if n == 0:
        return {}

    return {
        "n_samples": n,
        "json_valid_rate": sum(r["json_valid"] for r in results) / n,
        "schema_valid_rate": sum(r["schema_valid"] for r in results) / n,
        "avg_n_signals": sum(r["n_signals"] for r in results) / n,
        "scores_in_range_rate": sum(r["scores_in_range"] for r in results) / n,
        "korean_rationale_rate": sum(r["has_korean_rationale"] for r in results) / n,
        "has_confidence_rate": sum(r["has_confidence"] for r in results) / n,
        "has_regime_rate": sum(r["has_regime"] for r in results) / n,
        "has_analysis_rate": sum(r["has_analysis"] for r in results) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / n,
        "avg_response_length": sum(r["response_length"] for r in results) / n,
    }


# =============================================================================
# ROUGE Score (optional)
# =============================================================================

def compute_rouge(reference: str, hypothesis: str) -> dict[str, float] | None:
    """Compute ROUGE scores between reference and hypothesis."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False,
        )
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1_f": scores["rouge1"].fmeasure,
            "rouge2_f": scores["rouge2"].fmeasure,
            "rougeL_f": scores["rougeL"].fmeasure,
        }
    except ImportError:
        return None


# =============================================================================
# Main Evaluation
# =============================================================================

def run_evaluation(
    base_model: str,
    ft_model: str | None,
    ollama_url: str = "http://localhost:11434",
    n_samples: int = 3,
    temperature: float = 0.3,
    eval_data_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run full evaluation comparing base vs fine-tuned model.
    """
    from src.llm.ollama_client import OllamaClient

    client = OllamaClient(base_url=ollama_url, timeout=300.0)

    if not client.is_available():
        logger.error("Ollama is not running. Start with: ollama serve")
        return {}

    # Build eval prompts
    prompts = EVAL_PROMPTS[:n_samples]

    # Also load from eval dataset if available
    if eval_data_path and eval_data_path.exists():
        with open(eval_data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_samples:
                    break
                example = json.loads(line)
                messages = example.get("messages", [])
                user_msg = next(
                    (m["content"] for m in messages if m["role"] == "user"), None
                )
                ref_msg = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), None
                )
                if user_msg:
                    prompts.append({
                        "id": f"eval_{i}",
                        "prompt": user_msg,
                        "reference": ref_msg,
                    })

    results = {"base_model": base_model}

    # Evaluate base model
    logger.info(f"Evaluating base model: {base_model}")
    base_results = _evaluate_model(
        client, base_model, prompts, temperature,
    )
    results["base"] = {
        "model": base_model,
        "metrics": compute_aggregate_metrics(base_results),
        "details": [
            {k: v for k, v in r.items() if k != "parsed_json"}
            for r in base_results
        ],
    }

    # Evaluate fine-tuned model
    if ft_model:
        available_models = client.list_models()
        if ft_model not in available_models and f"{ft_model}:latest" not in available_models:
            logger.warning(
                f"Fine-tuned model '{ft_model}' not found in Ollama. "
                f"Available: {available_models}"
            )
        else:
            logger.info(f"Evaluating fine-tuned model: {ft_model}")
            ft_results = _evaluate_model(
                client, ft_model, prompts, temperature,
            )
            results["finetuned"] = {
                "model": ft_model,
                "metrics": compute_aggregate_metrics(ft_results),
                "details": [
                    {k: v for k, v in r.items() if k != "parsed_json"}
                    for r in ft_results
                ],
            }

            # Comparison
            results["comparison"] = _compare_models(
                results["base"]["metrics"],
                results["finetuned"]["metrics"],
            )

    return results


def _evaluate_model(
    client,
    model: str,
    prompts: list[dict],
    temperature: float,
) -> list[dict]:
    """Evaluate a single model on all prompts."""
    results = []

    for i, prompt_data in enumerate(prompts):
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["prompt"]
        reference = prompt_data.get("reference")

        logger.info(f"  [{i+1}/{len(prompts)}] {prompt_id}")

        start = time.monotonic()
        try:
            response = client.generate(
                prompt=prompt_text,
                model=model,
                json_mode=True,
                system=EVAL_SYSTEM_PROMPT,
                temperature=temperature,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            response_text = response.get("response", "")
        except Exception as e:
            logger.error(f"  Model call failed: {e}")
            latency_ms = int((time.monotonic() - start) * 1000)
            response_text = ""

        eval_result = evaluate_response(response_text, latency_ms)
        eval_result["prompt_id"] = prompt_id

        # ROUGE against reference if available
        if reference and response_text:
            rouge = compute_rouge(reference, response_text)
            if rouge:
                eval_result["rouge"] = rouge

        results.append(eval_result)
        logger.info(
            f"    JSON valid: {eval_result['json_valid']}, "
            f"Signals: {eval_result['n_signals']}, "
            f"Latency: {latency_ms}ms"
        )

    return results


def _compare_models(
    base_metrics: dict[str, float],
    ft_metrics: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compare base vs fine-tuned metrics."""
    comparison = {}
    for key in base_metrics:
        if key == "n_samples":
            continue
        base_val = base_metrics.get(key, 0)
        ft_val = ft_metrics.get(key, 0)
        diff = ft_val - base_val
        pct_change = (diff / base_val * 100) if base_val != 0 else 0
        comparison[key] = {
            "base": round(base_val, 4),
            "finetuned": round(ft_val, 4),
            "diff": round(diff, 4),
            "pct_change": round(pct_change, 2),
        }
    return comparison


def print_comparison(results: dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    base = results.get("base", {})
    if base:
        print(f"\n--- Base Model: {base.get('model', '?')} ---")
        _print_metrics(base.get("metrics", {}))

    ft = results.get("finetuned", {})
    if ft:
        print(f"\n--- Fine-tuned Model: {ft.get('model', '?')} ---")
        _print_metrics(ft.get("metrics", {}))

    comparison = results.get("comparison", {})
    if comparison:
        print(f"\n--- Comparison (fine-tuned - base) ---")
        print(f"{'Metric':<30} {'Base':>10} {'FT':>10} {'Diff':>10} {'%':>8}")
        print("-" * 70)
        for key, vals in comparison.items():
            print(
                f"{key:<30} {vals['base']:>10.4f} {vals['finetuned']:>10.4f} "
                f"{vals['diff']:>+10.4f} {vals['pct_change']:>+7.1f}%"
            )


def _print_metrics(metrics: dict) -> None:
    """Print metrics dict."""
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base vs fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="qwen2.5:32b",
        help="Ollama base model name",
    )
    parser.add_argument(
        "--ft-model",
        type=str,
        default="qwen2.5-kospi-ft",
        help="Ollama fine-tuned model name (skip if not available)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of evaluation prompts",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=PROJECT_ROOT / "finetune" / "data" / "sft_dataset.eval.jsonl",
        help="Path to eval JSONL for ROUGE comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "finetune" / "eval_results.json",
        help="Output results JSON path",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only evaluate base model",
    )

    args = parser.parse_args()

    ft_model = None if args.base_only else args.ft_model

    results = run_evaluation(
        base_model=args.base_model,
        ft_model=ft_model,
        ollama_url=args.ollama_url,
        n_samples=args.n_samples,
        eval_data_path=args.eval_data,
    )

    if results:
        print_comparison(results)

        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
