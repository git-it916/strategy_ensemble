"""
Stage 3 DPO Preference Pair Generator.

Stage 2의 synthetic 데이터(chosen)와 의도적으로 결함이 있는 응답(rejected)을
페어로 만들어 DPO 학습 데이터를 생성합니다.

Rejected 응답 생성 전략:
  1) schema_violation   - 필수 필드 누락, 잘못된 타입, malformed JSON
  2) strategy_violation  - regime에 맞지 않는 exposure/포지션
  3) quality_degradation - 근거 없음, 너무 짧음, 분석 부실
  4) hallucination       - 존재하지 않는 종목, 비현실적 밸류에이션
  5) verbose_response    - 장황한 응답 (chosen을 간결하게 만들고 원본을 rejected)
  6) model_generated     - Stage 2 모델의 실제 (나쁜) 출력 수집

Usage:
    # Synthetic rejected 생성
    python finetune/data/gen_dpo_pairs.py

    # Stage 2 모델 출력도 수집 (Ollama 필요)
    python finetune/data/gen_dpo_pairs.py --collect-model-rejects --model qwen2.5-kospi-ft-s2

    # 출력 건수 지정
    python finetune/data/gen_dpo_pairs.py --count 2000
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetune.config import QUANT_SYSTEM_PROMPT, KOREAN_STOCKS, REGIMES

random.seed(123)

DATA_DIR = Path(__file__).parent

# Regime → 정상 net_exposure 범위
REGIME_EXPOSURE = {
    "STRONG_BULL":       (1.0, 1.3),
    "MILD_BULL":         (0.80, 1.0),
    "WEAKENING":         (0.30, 0.60),
    "SHORT_TERM_STRESS": (0.10, 0.55),
    "BEAR":              (-0.80, -0.20),
}

# 반대 regime 매핑 (strategy violation용)
OPPOSITE_REGIME = {
    "STRONG_BULL": "BEAR",
    "MILD_BULL": "WEAKENING",
    "WEAKENING": "STRONG_BULL",
    "SHORT_TERM_STRESS": "MILD_BULL",
    "BEAR": "STRONG_BULL",
}


# =============================================================================
# Helper Functions
# =============================================================================

def _make_malformed_json(data: dict) -> str:
    """의도적으로 파싱 불가능한 JSON을 생성."""
    valid = json.dumps(data, indent=2, ensure_ascii=False)
    strategy = random.choice([
        "trailing_comma", "unclosed_bracket", "missing_quote",
        "double_comma", "truncated",
    ])
    if strategy == "trailing_comma":
        # 마지막 } 앞에 trailing comma 삽입
        idx = valid.rfind("}")
        return valid[:idx] + ",\n}" if idx > 0 else valid + ","
    elif strategy == "unclosed_bracket":
        # 마지막 닫는 괄호 제거
        if valid.rstrip().endswith("}"):
            return valid.rstrip()[:-1]
        return valid[:-1]
    elif strategy == "missing_quote":
        # 첫 번째 key의 따옴표 하나 제거
        idx = valid.find('"', 2)
        if idx > 0:
            return valid[:idx] + valid[idx+1:]
        return valid
    elif strategy == "double_comma":
        idx = valid.find(",")
        if idx > 0:
            return valid[:idx+1] + "," + valid[idx+1:]
        return valid
    else:  # truncated
        cut_point = len(valid) * random.randint(40, 70) // 100
        return valid[:cut_point]


def _make_concise_response(data: dict) -> dict:
    """응답을 간결하게 만듦. chosen으로 사용될 짧은 버전."""
    concise = copy.deepcopy(data)

    # 1. reasoning 간결화: 핵심만 남기기 (1~2문장)
    if "reasoning" in concise and isinstance(concise["reasoning"], str):
        sentences = concise["reasoning"].replace(". ", ".\n").split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 2:
            concise["reasoning"] = ". ".join(sentences[:2]).rstrip(".") + "."

    if "analysis" in concise and isinstance(concise["analysis"], str):
        sentences = concise["analysis"].replace(". ", ".\n").split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 2:
            concise["analysis"] = ". ".join(sentences[:2]).rstrip(".") + "."

    # 2. position reason 간결화
    for key in ("positions",):
        if key in concise and isinstance(concise[key], dict):
            for pos in concise[key].get("long", []):
                if "reason" in pos and len(pos["reason"]) > 80:
                    # 첫 문장만 유지
                    first = pos["reason"].split(". ")[0]
                    pos["reason"] = first.rstrip(".") + "."
            for pos in concise[key].get("short", []):
                if "reason" in pos and len(pos["reason"]) > 80:
                    first = pos["reason"].split(". ")[0]
                    pos["reason"] = first.rstrip(".") + "."

    # 3. affected_stocks도 동일 처리
    if "affected_stocks" in concise:
        for stock in concise["affected_stocks"]:
            for field in ("reason", "impact_detail", "reasoning"):
                if field in stock and isinstance(stock[field], str) and len(stock[field]) > 80:
                    first = stock[field].split(". ")[0]
                    stock[field] = first.rstrip(".") + "."

    # 4. 불필요한 반복 필드 제거
    for field in ("disclaimer", "notes", "additional_context", "market_summary"):
        concise.pop(field, None)

    return concise


def _make_verbose_response(data: dict) -> dict:
    """응답을 장황하게 만듦. rejected로 사용될 긴 버전."""
    verbose = copy.deepcopy(data)

    filler_phrases = [
        "It is important to note that ",
        "Furthermore, considering the broader macroeconomic environment, ",
        "Additionally, based on extensive historical analysis, ",
        "From a risk-adjusted perspective, it should be emphasized that ",
        "Taking into account various market microstructure factors, ",
        "In light of the current geopolitical landscape and its implications, ",
        "Moreover, when we consider the interplay between monetary policy and equity markets, ",
    ]

    # 1. reasoning에 장황한 문구 삽입
    if "reasoning" in verbose and isinstance(verbose["reasoning"], str):
        original = verbose["reasoning"]
        padding = random.sample(filler_phrases, k=min(3, len(filler_phrases)))
        verbose["reasoning"] = (
            original + " " +
            padding[0] + "this assessment remains valid under multiple scenarios. " +
            padding[1] + "we maintain our conviction in the current positioning. " +
            padding[2] + "the risk-reward profile continues to favor our thesis. " +
            "Past performance is not indicative of future results. "
            "Investors should always conduct their own due diligence. "
            "This analysis does not constitute investment advice."
        )

    # 2. 각 position에 장황한 reason 추가
    for key in ("positions",):
        if key in verbose and isinstance(verbose[key], dict):
            for pos in verbose[key].get("long", []) + verbose[key].get("short", []):
                if "reason" in pos:
                    pos["reason"] = (
                        pos["reason"] + " " +
                        random.choice(filler_phrases) +
                        "the fundamental thesis remains intact based on our proprietary multi-factor model. "
                        "Key catalysts include potential earnings beats, sector rotation dynamics, and "
                        "favorable technical formations on the daily and weekly charts."
                    )

    # 3. 불필요한 필드 추가
    verbose["disclaimer"] = (
        "This analysis is for informational purposes only and does not constitute "
        "investment advice. Past performance is not indicative of future results. "
        "Always consult with a qualified financial advisor before making investment decisions."
    )
    verbose["additional_context"] = (
        "The above recommendations are based on quantitative analysis of historical price data, "
        "technical indicators, fundamental metrics, and sentiment analysis. Market conditions "
        "may change rapidly and the analysis should be updated accordingly."
    )
    verbose["methodology_note"] = (
        "Signals generated using ensemble of RSI reversal, volatility breakout, value factor, "
        "momentum, and machine learning models. Each model's contribution is weighted by "
        "recent Sharpe ratio and correlation-adjusted to minimize redundancy."
    )

    return verbose


# =============================================================================
# Rejection Strategies
# =============================================================================

def reject_schema_violation(chosen_response: dict) -> str | None:
    """필수 필드를 제거하거나 잘못된 타입으로 변경. 항상 복합 변형 적용."""
    data = copy.deepcopy(chosen_response)

    # 20% 확률로 malformed JSON 생성 (파싱 자체가 불가능한 응답)
    if random.random() < 0.20:
        return _make_malformed_json(data)

    # 복합 변형: 2~3개 동시 적용
    mutations = random.sample([
        "remove_regime", "remove_positions", "wrong_type_exposure",
        "remove_confidence", "empty_signals", "strip_reasoning",
        "wrong_types", "flat_structure",
    ], k=random.randint(2, 3))

    for strategy in mutations:
        if strategy == "remove_regime":
            data.pop("regime", None)
            data.pop("regime_assessment", None)
        elif strategy == "remove_positions":
            data.pop("positions", None)
            data.pop("long", None)
            data.pop("short", None)
        elif strategy == "wrong_type_exposure":
            data["net_exposure"] = random.choice(["high", "aggressive", "maximum", True, [1.3]])
        elif strategy == "remove_confidence":
            data.pop("confidence", None)
        elif strategy == "empty_signals":
            if "positions" in data:
                data["positions"] = {"long": [], "short": []}
            if "signals" in data:
                data["signals"] = []
        elif strategy == "strip_reasoning":
            data.pop("reasoning", None)
            data.pop("analysis", None)
            # 종목 reason도 제거
            for key in ("positions", "affected_stocks"):
                if key in data and isinstance(data[key], dict):
                    for pos in data[key].get("long", []):
                        pos.pop("reason", None)
                    for pos in data[key].get("short", []):
                        pos.pop("reason", None)
        elif strategy == "wrong_types":
            if "confidence" in data:
                data["confidence"] = "very confident"
            if "positions" in data and isinstance(data["positions"], dict):
                for pos in data["positions"].get("long", []):
                    if "weight" in pos:
                        pos["weight"] = "large"
        elif strategy == "flat_structure":
            flat = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        flat[f"{k}_{k2}"] = v2
                else:
                    flat[k] = v
            data = flat

    return json.dumps(data, indent=2, ensure_ascii=False)


def reject_strategy_violation(chosen_response: dict, task_type: str) -> str | None:
    """Regime과 맞지 않는 전략적 결정을 생성. 항상 복합 변형."""
    data = copy.deepcopy(chosen_response)

    regime = data.get("regime", data.get("regime_assessment", ""))

    if regime not in OPPOSITE_REGIME:
        return json.dumps(data, indent=2, ensure_ascii=False)

    wrong_regime = OPPOSITE_REGIME[regime]
    wrong_lo, wrong_hi = REGIME_EXPOSURE.get(wrong_regime, (0.5, 1.0))

    # 항상 exposure 변경 + reasoning 모순 + 포지션 방향 오류 복합 적용
    # 1. Wrong exposure (항상)
    data["net_exposure"] = round(random.uniform(wrong_lo, wrong_hi), 2)
    if "recommended_net_exposure" in data:
        data["recommended_net_exposure"] = f"{wrong_lo*100:.0f}-{wrong_hi*100:.0f}%"

    # 2. Contradictory reasoning (항상)
    contradiction_templates = {
        ("BEAR", "SHORT_TERM_STRESS"): [
            "Strong bullish momentum with improving breadth. Foreign buying accelerating at +800B KRW/day. "
            "Telegram sentiment turning euphoric. Recommend aggressive long positioning with maximum leverage.",
            "Market showing classic V-bottom pattern. Institutional accumulation detected. "
            "All technical indicators flashing buy signals. Full long exposure recommended.",
            "Breadth improving rapidly to 75%. Volatility compressing. This is a textbook buying opportunity. "
            "Load up on high-beta momentum names with 130% net long.",
        ],
        ("STRONG_BULL", "MILD_BULL"): [
            "Severe deterioration in market breadth to 20%. Foreign outflows exceeding 1T KRW over 3 days. "
            "VIX spiking above 35. Recommend full liquidation and index short positioning.",
            "Crash imminent. Leading indicators all negative. Smart money exiting. "
            "Immediately cut all longs and go maximum short via inverse ETFs.",
            "Death cross forming on KOSPI daily chart. Breadth collapsed below 25%. "
            "Foreign selling accelerating. Risk-off: liquidate everything, 80% short.",
        ],
        ("WEAKENING",): [
            "Explosive breakout confirmed. Momentum surging. Foreign buying +500B KRW. "
            "Full long with 130% exposure, overweight high-beta semiconductor and battery names.",
        ],
    }
    for regimes, templates in contradiction_templates.items():
        if regime in regimes:
            data["reasoning"] = random.choice(templates)
            break

    # 3. Wrong direction (positions 있으면)
    if "positions" in data:
        if regime in ("STRONG_BULL", "MILD_BULL"):
            data["positions"]["long"] = []
            data["net_exposure"] = round(random.uniform(-0.5, -0.8), 2)
        elif regime in ("BEAR",):
            data["positions"]["short"] = []
            data["net_exposure"] = round(random.uniform(1.0, 1.3), 2)
        elif regime in ("SHORT_TERM_STRESS",):
            data["positions"]["short"] = []
            data["net_exposure"] = round(random.uniform(1.0, 1.3), 2)

    return json.dumps(data, indent=2, ensure_ascii=False)


def reject_quality_degradation(chosen_response: dict) -> str | None:
    """분석 품질을 의도적으로 저하. 항상 복합 변형 적용."""
    data = copy.deepcopy(chosen_response)

    # 복합 변형: 2~3개 동시 적용
    mutations = random.sample([
        "generic_reasoning", "strip_position_reasons", "remove_valuation",
        "confidence_wrong", "truncate_positions",
    ], k=random.randint(2, 3))

    for mutation in mutations:
        if mutation == "generic_reasoning":
            templates = [
                "Market looks okay. Buy some stocks.",
                "Based on current market conditions, we recommend the following portfolio. "
                "Please note that past performance is not indicative of future results. "
                "Always do your own research before making investment decisions.",
                "The market is showing mixed signals. We suggest a balanced approach. "
                "Diversification is key. Monitor closely for changes.",
                "Technical indicators suggest a trade. Consider the options carefully.",
                "Buy low sell high. Current conditions are favorable for trading.",
            ]
            data["reasoning"] = random.choice(templates)
            data.pop("analysis", None)

        elif mutation == "strip_position_reasons":
            generic = random.choice([
                "Good stock.", "Buy.", "Sell.", "Hold.",
                "This stock shows potential based on technical indicators.",
                "Looks promising.", "Fundamentals are strong.",
            ])
            for key in ("positions",):
                if key in data and isinstance(data[key], dict):
                    for pos in data[key].get("long", []):
                        pos["reason"] = generic
                    for pos in data[key].get("short", []):
                        pos["reason"] = generic
            if "affected_stocks" in data:
                for stock in data["affected_stocks"]:
                    if "reason" in stock:
                        stock["reason"] = generic
                    if "impact_detail" in stock:
                        stock["impact_detail"] = generic

        elif mutation == "remove_valuation":
            for key in ("positions",):
                if key in data and isinstance(data[key], dict):
                    for pos in data[key].get("long", []):
                        pos.pop("valuation", None)
                    for pos in data[key].get("short", []):
                        pos.pop("valuation", None)
            if "affected_stocks" in data:
                for stock in data["affected_stocks"]:
                    stock.pop("valuation", None)

        elif mutation == "confidence_wrong":
            data["confidence"] = round(random.uniform(0.95, 1.0), 2)  # 과도한 확신

        elif mutation == "truncate_positions":
            if "positions" in data and isinstance(data["positions"], dict):
                longs = data["positions"].get("long", [])
                if len(longs) > 1:
                    data["positions"]["long"] = longs[:1]  # 하나만 남김

    return json.dumps(data, indent=2, ensure_ascii=False)


def reject_hallucination(chosen_response: dict) -> str | None:
    """존재하지 않는 종목이나 비현실적 수치 삽입. 항상 복합 변형."""
    data = copy.deepcopy(chosen_response)

    fake_stocks = [
        {"ticker": "999999", "name": "Phantom Corp", "sector": "Unknown"},
        {"ticker": "888888", "name": "Ghost Industries", "sector": "Fantasy"},
        {"ticker": "777777", "name": "Mirage Tech", "sector": "Imaginary"},
        {"ticker": "666666", "name": "Vapor Holdings", "sector": "Nonexistent"},
        {"ticker": "555555", "name": "Illusion Capital", "sector": "Fictional"},
    ]

    # 복합 변형: 2~3개 동시 적용
    mutations = random.sample([
        "fake_tickers", "unrealistic_valuation", "impossible_leverage",
        "fabricated_reasoning", "wrong_sector",
    ], k=random.randint(2, 3))

    for mutation in mutations:
        if mutation == "fake_tickers":
            # 모든 종목을 가짜로 교체 (절반 이상)
            if "positions" in data and isinstance(data["positions"], dict):
                for pos_list in [data["positions"].get("long", []),
                                 data["positions"].get("short", [])]:
                    for pos in pos_list:
                        if random.random() < 0.7:  # 70% 확률로 교체
                            fake = random.choice(fake_stocks)
                            pos["ticker"] = fake["ticker"]
                            pos["name"] = fake["name"]
            if "affected_stocks" in data:
                for stock in data["affected_stocks"]:
                    if random.random() < 0.7:
                        fake = random.choice(fake_stocks)
                        stock["ticker"] = fake["ticker"]
                        stock["name"] = fake["name"]

        elif mutation == "unrealistic_valuation":
            for key in ("positions",):
                if key in data and isinstance(data[key], dict):
                    for pos in data[key].get("long", []):
                        if "valuation" in pos:
                            pos["valuation"]["PER"] = round(random.uniform(500, 2000), 1)
                            pos["valuation"]["PBR"] = round(random.uniform(50, 200), 1)
                            pos["valuation"]["EV/EBITDA"] = round(random.uniform(200, 800), 1)
                            pos["valuation"]["F-Score"] = random.randint(10, 15)  # max 9
            if "affected_stocks" in data:
                for stock in data["affected_stocks"]:
                    if "valuation" in stock:
                        stock["valuation"]["PER"] = round(random.uniform(500, 2000), 1)
                        stock["valuation"]["PBR"] = round(random.uniform(50, 200), 1)

        elif mutation == "impossible_leverage":
            data["net_exposure"] = round(random.uniform(3.0, 5.0), 2)
            if "max_leverage" in data:
                data["max_leverage"] = round(random.uniform(5.0, 10.0), 1)
            data["reasoning"] = (data.get("reasoning", "") +
                " Using 5x leverage for maximum returns. Risk is irrelevant when conviction is high.")

        elif mutation == "fabricated_reasoning":
            fabrications = [
                "Warren Buffett personally recommended buying Korean stocks yesterday in a CNBC interview. "
                "This confirms our thesis. All-in on high-beta names.",
                "According to leaked internal data from the Bank of Korea, interest rates will be cut by 200bp next week. "
                "Position for massive rally.",
                "Our proprietary AI model with 99.8% accuracy predicts a 15% rally within 3 days. "
                "Maximum leverage recommended. Guaranteed returns.",
                "Insider sources confirm Samsung will announce a 10:1 stock split tomorrow. "
                "Buy immediately before the announcement. This is certain information.",
            ]
            data["reasoning"] = random.choice(fabrications)

        elif mutation == "wrong_sector":
            if "positions" in data and isinstance(data["positions"], dict):
                wrong_sectors = ["Crypto Mining", "Cannabis", "Space Tourism",
                                 "NFT Platform", "Metaverse Real Estate"]
                for pos in data["positions"].get("long", []):
                    if random.random() < 0.5:
                        pos["sector"] = random.choice(wrong_sectors)

    return json.dumps(data, indent=2, ensure_ascii=False)


def reject_verbose_response(chosen_response: dict) -> tuple[str, str]:
    """장황한 응답(rejected)과 간결한 응답(chosen)을 생성.

    이 함수는 특별: chosen/rejected 모두 새로 만듦.
    - 간결한 버전 → chosen (짧고 핵심만)
    - 장황한 버전 → rejected (불필요한 반복, disclaimer, 과도한 설명)

    Returns:
        (concise_chosen, verbose_rejected)
    """
    concise = _make_concise_response(chosen_response)
    verbose = _make_verbose_response(chosen_response)

    concise_text = json.dumps(concise, indent=2, ensure_ascii=False)
    verbose_text = json.dumps(verbose, indent=2, ensure_ascii=False)

    return concise_text, verbose_text


# =============================================================================
# Pair Generation
# =============================================================================

REJECT_STRATEGIES = [
    ("schema_violation", reject_schema_violation, 0.15),
    ("strategy_violation", reject_strategy_violation, 0.25),
    ("quality_degradation", reject_quality_degradation, 0.20),
    ("hallucination", reject_hallucination, 0.15),
    ("verbose_response", None, 0.25),  # 별도 처리 (chosen도 변경)
]


def _parse_chatml(text: str) -> dict | None:
    """ChatML text를 system/user/assistant로 파싱."""
    parts = text.split("<|im_start|>")
    result = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        for role in ("system", "user", "assistant"):
            if part.startswith(role):
                content = part[len(role):].strip()
                content = content.replace("<|im_end|>", "").strip()
                result[role] = content
                break
    return result if "user" in result and "assistant" in result else None


def generate_dpo_pair(example: dict) -> dict | None:
    """하나의 synthetic 예제에서 DPO preference pair를 생성.

    Returns:
        {"prompt": str, "chosen": str, "rejected": str, "source": str, "reject_type": str}
    """
    text = example.get("text", "")
    source = example.get("source", "")

    parsed = _parse_chatml(text)
    if parsed is None:
        return None

    chosen_text = parsed["assistant"]

    # JSON 파싱 시도
    try:
        chosen_json = json.loads(chosen_text)
    except json.JSONDecodeError:
        return None

    # 확률 기반 rejection strategy 선택
    strategies = REJECT_STRATEGIES
    weights = [s[2] for s in strategies]
    selected = random.choices(strategies, weights=weights, k=1)[0]
    reject_name, reject_fn, _ = selected

    # DPO 데이터 포맷: prompt = system + user, chosen/rejected = assistant
    prompt = (
        f"<|im_start|>system\n{parsed.get('system', QUANT_SYSTEM_PROMPT)}<|im_end|>\n"
        f"<|im_start|>user\n{parsed['user']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # verbose_response: chosen/rejected 모두 새로 만듦
    if reject_name == "verbose_response":
        concise_text, verbose_text = reject_verbose_response(chosen_json)
        # 간결 버전이 원본보다 짧아야 의미 있음
        if len(concise_text) >= len(chosen_text) * 0.9:
            return None
        return {
            "prompt": prompt,
            "chosen": concise_text + "<|im_end|>",
            "rejected": verbose_text + "<|im_end|>",
            "source": source,
            "reject_type": reject_name,
        }

    # Generate rejected response
    if reject_name == "strategy_violation":
        rejected_text = reject_fn(chosen_json, source.replace("synthetic_", ""))
    else:
        rejected_text = reject_fn(chosen_json)

    if rejected_text is None or rejected_text == chosen_text:
        return None

    # 유사도 게이트: character-level 차이가 20% 미만이면 reject
    chosen_for_cmp = chosen_text.strip()
    rejected_for_cmp = rejected_text.strip()
    max_len = max(len(chosen_for_cmp), len(rejected_for_cmp))
    min_len = min(len(chosen_for_cmp), len(rejected_for_cmp))
    char_diffs = sum(1 for i in range(min_len) if chosen_for_cmp[i] != rejected_for_cmp[i])
    char_diffs += max_len - min_len
    diff_ratio = char_diffs / max_len if max_len > 0 else 0
    if diff_ratio < 0.20:
        return None  # 20% 미만 차이 → 다시 시도

    return {
        "prompt": prompt,
        "chosen": chosen_text + "<|im_end|>",
        "rejected": rejected_text + "<|im_end|>",
        "source": source,
        "reject_type": reject_name,
    }


def collect_model_rejects(
    examples: list[dict],
    model_name: str,
    ollama_url: str = "http://localhost:11434",
    max_samples: int = 200,
) -> list[dict]:
    """Stage 2 모델의 실제 출력을 rejected로 수집.

    모델이 chosen과 다른 (더 나쁜) 응답을 생성하면 rejected로 사용.
    """
    from src.llm.ollama_client import OllamaClient

    client = OllamaClient(base_url=ollama_url, timeout=120.0)
    if not client.is_available():
        print("WARNING: Ollama not available, skipping model-generated rejections")
        return []

    pairs = []
    sampled = random.sample(examples, min(max_samples, len(examples)))

    for i, ex in enumerate(sampled):
        parsed = _parse_chatml(ex.get("text", ""))
        if parsed is None:
            continue

        chosen_text = parsed["assistant"]

        try:
            resp = client.generate(
                prompt=parsed["user"],
                model=model_name,
                system=parsed.get("system", QUANT_SYSTEM_PROMPT),
                temperature=0.7,  # higher temp for diverse bad outputs
                json_mode=True,
            )
            model_output = resp.get("response", "").strip()
        except Exception as e:
            print(f"  [{i+1}] Model call failed: {e}")
            continue

        # 모델 출력이 chosen과 충분히 다르면 rejected로 사용
        if not model_output or model_output == chosen_text:
            continue

        # JSON 파싱이 안 되거나, 스키마가 불량하면 좋은 rejected 후보
        try:
            model_json = json.loads(model_output)
            # 필수 필드 체크
            has_regime = "regime" in model_json or "regime_assessment" in model_json
            has_positions = "positions" in model_json or "signals" in model_json
            has_reasoning = bool(model_json.get("reasoning") or model_json.get("analysis"))

            # 3개 중 2개 이상 없으면 rejected로 채택
            score = sum([has_regime, has_positions, has_reasoning])
            if score >= 2:
                continue  # 모델 출력이 너무 좋으면 skip
        except json.JSONDecodeError:
            pass  # JSON 파싱 실패 = 좋은 rejected

        prompt = (
            f"<|im_start|>system\n{parsed.get('system', QUANT_SYSTEM_PROMPT)}<|im_end|>\n"
            f"<|im_start|>user\n{parsed['user']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        pairs.append({
            "prompt": prompt,
            "chosen": chosen_text + "<|im_end|>",
            "rejected": model_output + "<|im_end|>",
            "source": ex.get("source", ""),
            "reject_type": "model_generated",
        })

        if (i + 1) % 20 == 0:
            print(f"  Collected {len(pairs)} model-generated rejections ({i+1}/{len(sampled)})")

    return pairs


def generate_all_pairs(
    synthetic_data: list[dict],
    count: int = 2000,
    model_reject_ratio: float = 0.0,
    model_name: str | None = None,
    ollama_url: str = "http://localhost:11434",
) -> list[dict]:
    """전체 DPO pair 생성."""
    pairs = []

    # 1. Synthetic rejected 생성
    synthetic_count = int(count * (1.0 - model_reject_ratio))
    print(f"Generating {synthetic_count} synthetic rejection pairs...")

    # 반복해서 충분한 수 확보
    attempts = 0
    max_attempts = synthetic_count * 3
    while len(pairs) < synthetic_count and attempts < max_attempts:
        ex = random.choice(synthetic_data)
        pair = generate_dpo_pair(ex)
        if pair is not None:
            pairs.append(pair)
        attempts += 1

    print(f"  Generated {len(pairs)} synthetic pairs (from {attempts} attempts)")

    # 2. Model-generated rejected 수집
    if model_reject_ratio > 0 and model_name:
        model_count = count - len(pairs)
        print(f"\nCollecting {model_count} model-generated rejections...")
        model_pairs = collect_model_rejects(
            synthetic_data, model_name, ollama_url,
            max_samples=model_count * 2,
        )
        pairs.extend(model_pairs[:model_count])
        print(f"  Added {min(len(model_pairs), model_count)} model-generated pairs")

    # Shuffle
    random.shuffle(pairs)

    # Stats
    from collections import Counter
    type_counts = Counter(p["reject_type"] for p in pairs)
    source_counts = Counter(p["source"] for p in pairs)

    print(f"\n=== DPO Pairs Summary ===")
    print(f"Total pairs: {len(pairs)}")
    print(f"\nBy rejection type:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c} ({100*c/len(pairs):.1f}%)")
    print(f"\nBy task source:")
    for s, c in source_counts.most_common(5):
        print(f"  {s}: {c}")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs for Stage 3"
    )
    parser.add_argument("--count", type=int, default=2000,
                        help="Number of DPO pairs to generate")
    parser.add_argument("--input", type=str,
                        default=str(DATA_DIR / "synthetic_data.json"),
                        help="Input synthetic data path")
    parser.add_argument("--output", type=str,
                        default=str(DATA_DIR / "dpo_pairs.json"),
                        help="Output DPO pairs path")
    parser.add_argument("--collect-model-rejects", action="store_true",
                        help="Also collect rejected outputs from Stage 2 model")
    parser.add_argument("--model", type=str, default="qwen2.5-kospi-ft-s2",
                        help="Ollama model name for model-generated rejections")
    parser.add_argument("--model-reject-ratio", type=float, default=0.15,
                        help="Fraction of pairs using model-generated rejections")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    args = parser.parse_args()

    # Load synthetic data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Synthetic data not found: {input_path}")
        print(f"Run first: python finetune/data/synthetic.py")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        synthetic_data = json.load(f)
    print(f"Loaded {len(synthetic_data)} synthetic examples")

    # Generate pairs
    model_ratio = args.model_reject_ratio if args.collect_model_rejects else 0.0
    model_name = args.model if args.collect_model_rejects else None

    pairs = generate_all_pairs(
        synthetic_data,
        count=args.count,
        model_reject_ratio=model_ratio,
        model_name=model_name,
        ollama_url=args.ollama_url,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(pairs)} DPO pairs to {output_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
