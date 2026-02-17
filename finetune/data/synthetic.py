"""
Synthetic Training Data Generator for Stage 2 Fine-tuning.

config/universe.py의 종목 유니버스를 사용하여
다양한 시장 상황(5개 regime)과 태스크 유형을 커버하는 학습 예제를 생성합니다.

태스크 10개:
  1) full_analysis        - 종합 분석 (regime + 포지션 + valuation)
  2) regime_classification - regime 분류만
  3) overnight_gap         - 오버나이트 갭 전략
  4) risk_management       - MDD/레버리지 리스크 관리
  5) regime_transition     - regime 전환 포지션 조정
  6) sector_analysis       - US 이벤트 → 한국 종목 영향
  7) opening_momentum      - 시초가 갭업 모멘텀
  8) fx_impact             - 환율 변동 → 수출/내수주 영향
  9) stop_loss_management  - 기존 포지션 손절/익절 판단
  10) intraday_rebalancing - 장중 포트폴리오 리밸런싱

Usage:
    python finetune/data/synthetic.py
    python finetune/data/synthetic.py --count 2000
    python finetune/data/synthetic.py --use-real-data --start-date 2020-01-01
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetune.config import (
    QUANT_SYSTEM_PROMPT,
    KOREAN_STOCKS,
    KRX_ETFS,
    US_CATALYSTS_BULLISH,
    US_CATALYSTS_BEARISH,
    TELEGRAM_THEMES_BULLISH,
    TELEGRAM_THEMES_BEARISH,
    REGIMES,
)

random.seed(42)

DATA_DIR = Path(__file__).parent

# Real data snapshot store (initialized when --use-real-data is set)
_snapshot_store = None  # type: ignore


# =============================================================================
# 밸류에이션 데이터 생성기
# =============================================================================

# 섹터별 현실적 밸류에이션 범위
SECTOR_VALUATION = {
    "Semiconductor":          {"per": (8, 25),  "pbr": (1.0, 3.5), "ev_ebitda": (5, 15),  "roe": (8, 25)},
    "Semiconductor Equipment":{"per": (10, 35), "pbr": (2.0, 8.0), "ev_ebitda": (8, 25),  "roe": (12, 35)},
    "Semiconductor Test":     {"per": (12, 30), "pbr": (2.5, 7.0), "ev_ebitda": (10, 22), "roe": (15, 30)},
    "Semiconductor Chemical": {"per": (10, 25), "pbr": (1.5, 5.0), "ev_ebitda": (7, 18),  "roe": (10, 22)},
    "Semiconductor Parts":    {"per": (12, 30), "pbr": (2.0, 6.0), "ev_ebitda": (8, 20),  "roe": (12, 28)},
    "Semiconductor Probe":    {"per": (15, 35), "pbr": (3.0, 8.0), "ev_ebitda": (10, 25), "roe": (15, 30)},
    "Semiconductor Substrate":{"per": (12, 30), "pbr": (2.0, 6.0), "ev_ebitda": (8, 20),  "roe": (12, 25)},
    "Battery":                {"per": (15, 60), "pbr": (1.5, 5.0), "ev_ebitda": (10, 35), "roe": (3, 15)},
    "Battery Cathode":        {"per": (20, 80), "pbr": (3.0, 12.0),"ev_ebitda": (15, 50), "roe": (5, 20)},
    "Battery Materials Holding":{"per": (25, 90),"pbr": (4.0, 15.0),"ev_ebitda": (18, 55),"roe": (3, 18)},
    "Battery Copper Foil":    {"per": (20, 50), "pbr": (2.0, 6.0), "ev_ebitda": (12, 30), "roe": (5, 15)},
    "Battery Anode/Cathode":  {"per": (15, 45), "pbr": (1.5, 5.0), "ev_ebitda": (10, 28), "roe": (5, 18)},
    "Battery Housing":        {"per": (10, 30), "pbr": (1.5, 5.0), "ev_ebitda": (8, 20),  "roe": (8, 20)},
    "Battery Recycling":      {"per": (25, 80), "pbr": (3.0, 10.0),"ev_ebitda": (15, 40), "roe": (3, 12)},
    "Bio/CDMO":               {"per": (30, 80), "pbr": (3.0, 10.0),"ev_ebitda": (20, 50), "roe": (5, 15)},
    "Bio/Biosimilar":         {"per": (20, 50), "pbr": (2.0, 6.0), "ev_ebitda": (12, 30), "roe": (8, 18)},
    "Bio/Oncology":           {"per": (50, 200),"pbr": (5.0, 20.0),"ev_ebitda": (30, 100),"roe": (-5, 10)},
    "Bio/Aesthetics":         {"per": (20, 45), "pbr": (3.0, 8.0), "ev_ebitda": (15, 30), "roe": (15, 30)},
    "Bio/Platform":           {"per": (40, 120),"pbr": (5.0, 15.0),"ev_ebitda": (25, 60), "roe": (3, 15)},
    "Bio/Vaccine":            {"per": (25, 60), "pbr": (2.0, 7.0), "ev_ebitda": (12, 35), "roe": (5, 15)},
    "Bio/Stem Cell":          {"per": (50, 150),"pbr": (4.0, 12.0),"ev_ebitda": (30, 80), "roe": (-3, 8)},
    "Pharma":                 {"per": (12, 30), "pbr": (1.5, 4.0), "ev_ebitda": (8, 20),  "roe": (10, 20)},
    "Pharma/CNS":             {"per": (25, 60), "pbr": (2.0, 6.0), "ev_ebitda": (15, 35), "roe": (5, 15)},
    "CDMO/Pharma":            {"per": (15, 40), "pbr": (1.5, 5.0), "ev_ebitda": (10, 25), "roe": (8, 18)},
    "Auto":                   {"per": (5, 12),  "pbr": (0.5, 1.5), "ev_ebitda": (3, 8),   "roe": (10, 20)},
    "Auto Parts":             {"per": (6, 15),  "pbr": (0.5, 1.8), "ev_ebitda": (4, 10),  "roe": (8, 15)},
    "Auto Semiconductor":     {"per": (12, 30), "pbr": (2.0, 6.0), "ev_ebitda": (8, 20),  "roe": (12, 25)},
    "Finance":                {"per": (4, 8),   "pbr": (0.3, 0.7), "ev_ebitda": (3, 7),   "roe": (8, 14)},
    "Regional Finance":       {"per": (3, 7),   "pbr": (0.2, 0.5), "ev_ebitda": (2, 6),   "roe": (6, 10)},
    "Securities":             {"per": (5, 12),  "pbr": (0.4, 0.9), "ev_ebitda": (4, 10),  "roe": (8, 15)},
    "Insurance":              {"per": (5, 10),  "pbr": (0.3, 0.6), "ev_ebitda": (3, 8),   "roe": (8, 12)},
    "Internet/Platform":      {"per": (15, 40), "pbr": (1.5, 5.0), "ev_ebitda": (10, 25), "roe": (8, 20)},
    "Gaming":                 {"per": (12, 35), "pbr": (1.0, 4.0), "ev_ebitda": (8, 22),  "roe": (8, 22)},
    "Gaming/Mobile":          {"per": (10, 30), "pbr": (1.0, 3.5), "ev_ebitda": (7, 20),  "roe": (5, 18)},
    "Gaming/Blockchain":      {"per": (15, 40), "pbr": (1.5, 5.0), "ev_ebitda": (10, 25), "roe": (5, 15)},
    "Defense/Aerospace":      {"per": (15, 35), "pbr": (2.0, 6.0), "ev_ebitda": (10, 25), "roe": (10, 22)},
    "Defense/Missile":        {"per": (12, 28), "pbr": (1.5, 4.5), "ev_ebitda": (8, 20),  "roe": (10, 20)},
    "Defense/IT":             {"per": (15, 35), "pbr": (2.0, 5.5), "ev_ebitda": (10, 22), "roe": (8, 18)},
    "Defense/Rail":           {"per": (10, 25), "pbr": (1.5, 4.0), "ev_ebitda": (7, 18),  "roe": (8, 15)},
    "Transformer/Grid":       {"per": (10, 30), "pbr": (1.5, 5.0), "ev_ebitda": (7, 20),  "roe": (10, 22)},
    "Transformer/Switchgear": {"per": (12, 30), "pbr": (2.0, 6.0), "ev_ebitda": (8, 22),  "roe": (12, 25)},
    "Transformer":            {"per": (10, 28), "pbr": (1.5, 5.0), "ev_ebitda": (7, 18),  "roe": (10, 22)},
    "Electrical Equipment":   {"per": (10, 25), "pbr": (1.5, 4.5), "ev_ebitda": (7, 18),  "roe": (10, 20)},
    "Generator/Engine":       {"per": (8, 22),  "pbr": (1.0, 4.0), "ev_ebitda": (5, 15),  "roe": (8, 18)},
    "Nuclear/Energy":         {"per": (10, 30), "pbr": (1.5, 5.0), "ev_ebitda": (7, 20),  "roe": (5, 15)},
    "Nuclear Equipment":      {"per": (12, 35), "pbr": (2.0, 6.0), "ev_ebitda": (8, 22),  "roe": (5, 15)},
    "Shipbuilding":           {"per": (8, 20),  "pbr": (1.0, 3.0), "ev_ebitda": (5, 15),  "roe": (8, 18)},
    "Shipbuilding/Holding":   {"per": (7, 18),  "pbr": (0.8, 2.5), "ev_ebitda": (5, 12),  "roe": (8, 15)},
    "Steel/Holding":          {"per": (5, 12),  "pbr": (0.4, 1.0), "ev_ebitda": (3, 8),   "roe": (5, 12)},
    "Non-ferrous Metals":     {"per": (6, 15),  "pbr": (0.5, 1.5), "ev_ebitda": (4, 10),  "roe": (8, 15)},
    "Chemical/Battery":       {"per": (8, 20),  "pbr": (0.8, 2.0), "ev_ebitda": (5, 12),  "roe": (5, 12)},
    "Petrochemical":          {"per": (5, 15),  "pbr": (0.4, 1.2), "ev_ebitda": (3, 10),  "roe": (3, 10)},
    "Electronics":            {"per": (8, 18),  "pbr": (0.8, 2.0), "ev_ebitda": (5, 12),  "roe": (8, 15)},
    "Electronic Components":  {"per": (10, 25), "pbr": (1.0, 3.0), "ev_ebitda": (6, 15),  "roe": (10, 20)},
    "IT Services":            {"per": (10, 22), "pbr": (1.0, 3.0), "ev_ebitda": (6, 15),  "roe": (10, 18)},
    "IT Services/AI":         {"per": (12, 30), "pbr": (1.5, 5.0), "ev_ebitda": (8, 20),  "roe": (10, 22)},
    "Telecom":                {"per": (8, 14),  "pbr": (0.5, 1.0), "ev_ebitda": (4, 8),   "roe": (6, 10)},
    "Utility":                {"per": (8, 15),  "pbr": (0.3, 0.7), "ev_ebitda": (5, 10),  "roe": (2, 8)},
    "Tobacco/Consumer":       {"per": (8, 14),  "pbr": (0.8, 1.5), "ev_ebitda": (5, 10),  "roe": (10, 18)},
    "Cosmetics":              {"per": (15, 35), "pbr": (1.5, 5.0), "ev_ebitda": (10, 22), "roe": (8, 18)},
    "Food/Bio":               {"per": (8, 18),  "pbr": (0.8, 2.0), "ev_ebitda": (5, 12),  "roe": (8, 14)},
    "Retail/Department":      {"per": (6, 15),  "pbr": (0.4, 1.2), "ev_ebitda": (4, 10),  "roe": (5, 12)},
    "Construction":           {"per": (5, 12),  "pbr": (0.4, 1.0), "ev_ebitda": (3, 8),   "roe": (6, 12)},
    "Construction/Holding":   {"per": (6, 14),  "pbr": (0.5, 1.2), "ev_ebitda": (4, 10),  "roe": (6, 12)},
    "Trading/LNG":            {"per": (5, 12),  "pbr": (0.5, 1.2), "ev_ebitda": (4, 9),   "roe": (8, 15)},
    "Airline":                {"per": (5, 15),  "pbr": (0.5, 2.0), "ev_ebitda": (3, 10),  "roe": (5, 18)},
    "Media/Content":          {"per": (10, 30), "pbr": (1.0, 3.5), "ev_ebitda": (6, 18),  "roe": (5, 15)},
    "K-Pop/Entertainment":    {"per": (15, 40), "pbr": (2.0, 6.0), "ev_ebitda": (10, 25), "roe": (10, 25)},
    "Entertainment":          {"per": (15, 40), "pbr": (2.0, 6.0), "ev_ebitda": (10, 25), "roe": (8, 20)},
    "Holding":                {"per": (8, 18),  "pbr": (0.4, 1.0), "ev_ebitda": (5, 12),  "roe": (5, 10)},
    "Solar Module":           {"per": (10, 30), "pbr": (1.0, 4.0), "ev_ebitda": (7, 20),  "roe": (5, 15)},
    "Bio Equipment":          {"per": (15, 35), "pbr": (2.0, 6.0), "ev_ebitda": (10, 22), "roe": (8, 18)},
    "Tire":                   {"per": (5, 12),  "pbr": (0.5, 1.2), "ev_ebitda": (3, 8),   "roe": (8, 14)},
}

# 기본 범위 (매핑 안 된 섹터용)
_DEFAULT_VAL = {"per": (8, 25), "pbr": (0.8, 3.0), "ev_ebitda": (5, 15), "roe": (5, 15)}


def _gen_valuation(sector: str) -> dict:
    """섹터에 맞는 현실적 밸류에이션 데이터를 생성합니다."""
    v = SECTOR_VALUATION.get(sector, _DEFAULT_VAL)
    per = round(random.uniform(*v["per"]), 1)
    pbr = round(random.uniform(*v["pbr"]), 2)
    ev_ebitda = round(random.uniform(*v["ev_ebitda"]), 1)
    roe = round(random.uniform(*v["roe"]), 1)

    # F-Score (간이 Piotroski 0-9)
    f_score = 0
    if roe > 0: f_score += 2          # 수익성
    if per > 0 and per < 20: f_score += 2  # 합리적 밸류
    if pbr < 3.0: f_score += 2        # 저PBR
    debt_equity = round(random.uniform(0.1, 2.5), 2)
    if debt_equity < 1.0: f_score += 2  # 저부채
    f_score += 1  # 시총 1000억+ 보너스
    f_score = min(f_score, 9)

    return {
        "per": per, "pbr": pbr, "ev_ebitda": ev_ebitda,
        "roe": roe, "debt_equity": debt_equity, "f_score": f_score,
    }


# =============================================================================
# 시장 데이터 생성기
# =============================================================================


def _gen_market_data(regime: str) -> dict:
    """주어진 regime에 맞는 현실적인 시장 데이터를 생성합니다."""
    params = {
        "STRONG_BULL": {
            "kospi": (0.8, 2.5), "kosdaq": (0.5, 3.0), "mom": (2.0, 6.0),
            "vol": (8.0, 15.0), "tg": (0.65, 0.90), "br": (65, 82),
            "foreign": (200, 800), "inst": (100, 500),
            "usdkrw": (1280, 1330), "vix": (10, 16),
        },
        "MILD_BULL": {
            "kospi": (0.2, 1.2), "kosdaq": (-0.3, 1.5), "mom": (0.5, 3.0),
            "vol": (10.0, 18.0), "tg": (0.50, 0.70), "br": (52, 65),
            "foreign": (-100, 400), "inst": (-50, 300),
            "usdkrw": (1300, 1360), "vix": (14, 20),
        },
        "WEAKENING": {
            "kospi": (-0.8, 0.3), "kosdaq": (-1.2, 0.2), "mom": (-2.0, 0.5),
            "vol": (15.0, 25.0), "tg": (0.30, 0.50), "br": (38, 52),
            "foreign": (-400, 50), "inst": (-200, 100),
            "usdkrw": (1340, 1400), "vix": (18, 26),
        },
        "SHORT_TERM_STRESS": {
            "kospi": (-2.5, -0.8), "kosdaq": (-3.5, -1.0), "mom": (-4.0, -1.0),
            "vol": (20.0, 35.0), "tg": (0.15, 0.35), "br": (25, 40),
            "foreign": (-800, -200), "inst": (-300, 100),
            "usdkrw": (1370, 1430), "vix": (22, 35),
        },
        "BEAR": {
            "kospi": (-3.5, -1.5), "kosdaq": (-4.5, -2.0), "mom": (-8.0, -3.0),
            "vol": (28.0, 50.0), "tg": (0.05, 0.20), "br": (12, 30),
            "foreign": (-1200, -500), "inst": (-600, -100),
            "usdkrw": (1400, 1480), "vix": (28, 50),
        },
    }
    p = params[regime]
    return {
        "kospi_change_pct": round(random.uniform(*p["kospi"]), 2),
        "kosdaq_change_pct": round(random.uniform(*p["kosdaq"]), 2),
        "momentum_10d_pct": round(random.uniform(*p["mom"]), 2),
        "realized_vol_10d_pct": round(random.uniform(*p["vol"]), 1),
        "telegram_sentiment": round(random.uniform(*p["tg"]), 2),
        "market_breadth_pct": round(random.uniform(*p["br"]), 1),
        "foreign_net_buying_bn_krw": int(random.uniform(*p["foreign"])),
        "institutional_net_buying_bn_krw": int(random.uniform(*p["inst"])),
        "usdkrw": round(random.uniform(*p["usdkrw"]), 1),
        "vix": round(random.uniform(*p["vix"]), 1),
    }


def _gen_market_data_real(snapshot) -> dict:
    """실제 MarketSnapshot에서 시장 데이터를 생성합니다.

    실제 값: market_return, momentum, volatility, breadth
    랜덤 (regime 기반): telegram, foreign/inst, USD/KRW, VIX
    """
    regime = snapshot.regime
    kospi_pct = round(snapshot.market_return_pct, 2)
    kosdaq_pct = round(kospi_pct + random.uniform(-0.5, 0.5), 2)

    rp = {
        "STRONG_BULL":       {"tg": (0.65, 0.90), "foreign": (200, 800), "inst": (100, 500), "usdkrw": (1280, 1330), "vix": (10, 16)},
        "MILD_BULL":         {"tg": (0.50, 0.70), "foreign": (-100, 400), "inst": (-50, 300), "usdkrw": (1300, 1360), "vix": (14, 20)},
        "WEAKENING":         {"tg": (0.30, 0.50), "foreign": (-400, 50), "inst": (-200, 100), "usdkrw": (1340, 1400), "vix": (18, 26)},
        "SHORT_TERM_STRESS": {"tg": (0.15, 0.35), "foreign": (-800, -200), "inst": (-300, 100), "usdkrw": (1370, 1430), "vix": (22, 35)},
        "BEAR":              {"tg": (0.05, 0.20), "foreign": (-1200, -500), "inst": (-600, -100), "usdkrw": (1400, 1480), "vix": (28, 50)},
    }
    p = rp[regime]

    return {
        "kospi_change_pct": kospi_pct,
        "kosdaq_change_pct": kosdaq_pct,
        "momentum_10d_pct": round(snapshot.momentum_10d_pct, 2),
        "realized_vol_10d_pct": round(snapshot.realized_vol_10d_pct, 1),
        "telegram_sentiment": round(random.uniform(*p["tg"]), 2),
        "market_breadth_pct": round(snapshot.market_breadth_pct, 1),
        "foreign_net_buying_bn_krw": int(random.uniform(*p["foreign"])),
        "institutional_net_buying_bn_krw": int(random.uniform(*p["inst"])),
        "usdkrw": round(random.uniform(*p["usdkrw"]), 1),
        "vix": round(random.uniform(*p["vix"]), 1),
    }


def _gen_valuation_real(snapshot, ticker: str, sector: str) -> dict:
    """실제 PE/PBR 데이터를 사용하여 밸류에이션을 생성합니다.

    실제 값: PE, PBR (범위 내일 때)
    랜덤: EV/EBITDA, ROE, debt_equity, F-Score
    """
    stock = snapshot.stocks.get(ticker) if snapshot else None
    v = SECTOR_VALUATION.get(sector, _DEFAULT_VAL)

    # PE: 실제값 사용 (0.5 < PE < 200 범위 내)
    if stock and stock.pe_ratio is not None and 0.5 < stock.pe_ratio < 200:
        per = round(stock.pe_ratio, 1)
    else:
        per = round(random.uniform(*v["per"]), 1)

    # PBR: 실제값 사용 (0.01 < PBR < 50 범위 내)
    if stock and stock.pb_ratio is not None and 0.01 < stock.pb_ratio < 50:
        pbr = round(stock.pb_ratio, 2)
    else:
        pbr = round(random.uniform(*v["pbr"]), 2)

    ev_ebitda = round(random.uniform(*v["ev_ebitda"]), 1)
    roe = round(random.uniform(*v["roe"]), 1)

    f_score = 0
    if roe > 0: f_score += 2
    if per > 0 and per < 20: f_score += 2
    if pbr < 3.0: f_score += 2
    debt_equity = round(random.uniform(0.1, 2.5), 2)
    if debt_equity < 1.0: f_score += 2
    f_score += 1
    f_score = min(f_score, 9)

    return {
        "per": per, "pbr": pbr, "ev_ebitda": ev_ebitda,
        "roe": roe, "debt_equity": debt_equity, "f_score": f_score,
    }


def _format_market_data_text(data: dict, extras: str = "") -> str:
    """시장 데이터를 다양한 형식으로 변환합니다 (bullet / sentence / compact / table)."""
    fmt = random.choice(["bullet", "bullet", "sentence", "compact", "table"])

    if fmt == "bullet":
        text = (
            f"Market data:\n"
            f"- KOSPI: {data['kospi_change_pct']:+.2f}%\n"
            f"- KOSDAQ: {data['kosdaq_change_pct']:+.2f}%\n"
            f"- 10-day momentum: {data['momentum_10d_pct']:+.2f}%\n"
            f"- 10-day realized volatility: {data['realized_vol_10d_pct']:.1f}%\n"
            f"- Telegram sentiment: {data['telegram_sentiment']:.2f}\n"
            f"- Market breadth (advancing %): {data['market_breadth_pct']:.1f}%\n"
            f"- Foreign net buying: {data['foreign_net_buying_bn_krw']:+,} B KRW\n"
            f"- Institutional net buying: {data['institutional_net_buying_bn_krw']:+,} B KRW\n"
            f"- USD/KRW: {data['usdkrw']:.1f}\n"
            f"- VIX: {data['vix']:.1f}"
        )
    elif fmt == "sentence":
        text = (
            f"KOSPI is {data['kospi_change_pct']:+.2f}% and KOSDAQ {data['kosdaq_change_pct']:+.2f}% today. "
            f"10-day momentum stands at {data['momentum_10d_pct']:+.2f}% with realized vol at {data['realized_vol_10d_pct']:.1f}%. "
            f"Telegram sentiment reads {data['telegram_sentiment']:.2f}, breadth at {data['market_breadth_pct']:.1f}% advancing. "
            f"Foreign flow: {data['foreign_net_buying_bn_krw']:+,}B KRW, institutional: {data['institutional_net_buying_bn_krw']:+,}B KRW. "
            f"USD/KRW at {data['usdkrw']:.1f}, VIX at {data['vix']:.1f}."
        )
    elif fmt == "compact":
        text = (
            f"KOSPI {data['kospi_change_pct']:+.2f}% | KOSDAQ {data['kosdaq_change_pct']:+.2f}% | "
            f"Mom {data['momentum_10d_pct']:+.2f}% | Vol {data['realized_vol_10d_pct']:.1f}% | "
            f"TG {data['telegram_sentiment']:.2f} | Breadth {data['market_breadth_pct']:.1f}% | "
            f"Foreign {data['foreign_net_buying_bn_krw']:+,}B | Inst {data['institutional_net_buying_bn_krw']:+,}B | "
            f"USD/KRW {data['usdkrw']:.1f} | VIX {data['vix']:.1f}"
        )
    else:  # table
        text = (
            f"| Indicator      | Value          |\n"
            f"|----------------|----------------|\n"
            f"| KOSPI          | {data['kospi_change_pct']:+.2f}%       |\n"
            f"| KOSDAQ         | {data['kosdaq_change_pct']:+.2f}%       |\n"
            f"| 10d Momentum   | {data['momentum_10d_pct']:+.2f}%       |\n"
            f"| 10d Vol        | {data['realized_vol_10d_pct']:.1f}%         |\n"
            f"| TG Sentiment   | {data['telegram_sentiment']:.2f}          |\n"
            f"| Breadth        | {data['market_breadth_pct']:.1f}%         |\n"
            f"| Foreign Flow   | {data['foreign_net_buying_bn_krw']:+,}B KRW |\n"
            f"| Inst Flow      | {data['institutional_net_buying_bn_krw']:+,}B KRW |\n"
            f"| USD/KRW        | {data['usdkrw']:.1f}        |\n"
            f"| VIX            | {data['vix']:.1f}          |"
        )

    if extras:
        text += f"\n\n{extras}"
    return text


# =============================================================================
# User 질문 다양화 & Context 변주
# =============================================================================

# 태스크별 질문 템플릿 (8~12개씩)
QUERY_TEMPLATES = {
    "full_analysis": [
        "Classify the current regime and recommend positions with valuation analysis.",
        "What's the current regime? Build a portfolio with valuation-backed reasoning.",
        "Regime assessment + full position recommendation needed.",
        "Read the market data and give me regime classification, target positions, and valuation justification.",
        "Based on these indicators, what regime are we in and what should our book look like?",
        "Full analysis please — regime, net exposure, and individual stock picks with valuations.",
        "Where are we in the cycle? Recommend longs and shorts with PER/PBR support.",
        "Analyze market conditions. Output regime, conviction level, and position sizing.",
        "Market snapshot below. Classify regime and construct an optimal portfolio.",
        "Give me a complete regime + portfolio recommendation with F-Score analysis.",
        "Run full analysis: regime determination, risk assessment, and position construction.",
        "Process these market signals and return regime classification with actionable positions.",
    ],
    "regime_classification": [
        "Classify the current market regime.",
        "What regime are we in right now?",
        "Based on these numbers, where are we in the market cycle?",
        "Regime check — is this bull, bear, or transitional?",
        "Determine the current market regime from this data.",
        "Quick regime classification with confidence level.",
        "What's your regime call based on these indicators?",
        "Assess current market conditions and classify the regime.",
        "Read the data and tell me: which of the 5 regimes fits best?",
        "Market regime determination needed. What's your read?",
    ],
    "overnight_gap": [
        "What overnight positions should we take before the Korean close?",
        "How should we position for the overnight gap based on US signals?",
        "Pre-close positioning: what trades to put on before 15:30 KST?",
        "US futures are moving — what overnight exposure should we take?",
        "Plan overnight strategy. What to hold through the close?",
        "Should we take overnight US-linked positions? If so, what?",
        "Recommend overnight gap strategy based on these US pre-market signals.",
        "Korean market closing soon. How to position for tomorrow's open?",
    ],
    "risk_management": [
        "Evaluate risk level and recommend leverage adjustment.",
        "Are we within risk limits? What leverage changes are needed?",
        "Risk check: is our current leverage appropriate for this regime?",
        "Assess portfolio risk and recommend position sizing changes.",
        "MDD and leverage review — should we reduce exposure?",
        "Is our risk budget being used efficiently? Recommend adjustments.",
        "Portfolio risk assessment. Are we overleveraged for current conditions?",
        "Check if leverage is within NAV/MDD limits and advise accordingly.",
    ],
    "regime_transition": [
        "Plan the transition. Execute within 1-2 trading days.",
        "How should we adjust the portfolio for this regime change?",
        "Regime shift detected. What's the execution plan?",
        "Design a 2-day transition plan for the new regime.",
        "Portfolio needs restructuring for the new regime. Prioritize actions.",
        "Map out the transition trades. What to sell first, what to buy?",
        "Execute regime transition: timeline, priority sells, and new positions.",
        "We're shifting regimes. Plan the rotation with urgency assessment.",
    ],
    "sector_analysis": [
        "Which Korean stocks are affected and how should we adjust?",
        "How does this event impact Korean equities? Recommend trades.",
        "Sector impact analysis — who benefits and who gets hurt?",
        "Map this global event to Korean stock exposure. What to do?",
        "Which names in our universe are affected? Long or short?",
        "Analyze the downstream impact on Korean companies and recommend action.",
        "Break down the sector implications for our portfolio.",
        "Given this catalyst, which Korean stocks should we add or trim?",
    ],
    "opening_momentum": [
        "Analyze and recommend.",
        "Should we chase this gap-up or fade it?",
        "Opening momentum detected. Is this a real move or a trap?",
        "These stocks are gapping up — which ones to ride, which to avoid?",
        "Evaluate gap-up candidates. Quality gap or short squeeze?",
        "Morning momentum analysis. Worth entering at open?",
        "Pre-market gaps detected. Analyze sustainability and recommend.",
        "Which of these gap-ups have follow-through potential?",
    ],
    "fx_impact": [
        "How does this affect our portfolio? Which sectors benefit or suffer?",
        "USD/KRW move — impact on exporters vs importers?",
        "FX analysis needed. Who wins and who loses from this move?",
        "Korean won is moving. How should we adjust sector exposure?",
        "Analyze currency impact on our universe. Rotation needed?",
        "Break down the FX effect on portfolio holdings and recommend trades.",
        "How does this USD/KRW shift change the sector landscape?",
        "Currency move detected. Which names benefit and which are at risk?",
    ],
    "stop_loss_management": [
        "Evaluate each position for stop-loss or take-profit.",
        "Review all positions — any stops or profit-taking needed?",
        "Position-by-position P&L review. What to cut, what to hold?",
        "Run stop-loss check across the portfolio.",
        "Which positions should we exit and which still have thesis intact?",
        "Portfolio health check: flag any positions for exit.",
        "Assess each holding. Any stop-loss triggers or take-profit targets hit?",
        "Review P&L and fundamentals for each position. Recommend actions.",
    ],
    "intraday_rebalancing": [
        "Should we rebalance? What changes to make?",
        "Intraday event just hit. Do we need to adjust positions?",
        "Mid-session review: is rebalancing warranted?",
        "React to this intraday development. What trades to execute?",
        "Event-driven rebalancing check. Should we act now or wait?",
        "Assess the intraday move and recommend portfolio adjustments.",
        "Do we need to rebalance based on this event? If so, how?",
        "Intraday alert: evaluate if position changes are needed.",
    ],
}


def _gen_context_prefix(snapshot=None) -> str:
    """날짜, 포트폴리오 이력, 리스크 예산 등 컨텍스트를 랜덤 생성합니다."""
    parts = []

    # 날짜 (snapshot이 있으면 실제 날짜 사용)
    if snapshot and hasattr(snapshot, "date"):
        parts.append(f"Date: {snapshot.date.strftime('%Y-%m-%d')}")
    elif random.random() < 0.6:
        y = random.randint(2020, 2025)
        m = random.randint(1, 12)
        d = random.randint(1, 28)
        parts.append(f"Date: {y}-{m:02d}-{d:02d}")

    # 포트폴리오 컨텍스트 (50% 확률)
    if random.random() < 0.5:
        ctx = random.choice([
            f"Portfolio NAV: {round(random.uniform(1.0, 15.0), 1)}B KRW",
            f"AUM: {round(random.uniform(0.5, 10.0), 1)}B KRW, inception 2023",
            f"Fund size: {round(random.uniform(2.0, 20.0), 1)}B KRW",
        ])
        parts.append(ctx)

    # 최근 이력 (30% 확률)
    if random.random() < 0.3:
        hist = random.choice([
            f"Previous regime held for {random.randint(3, 25)} trading days",
            f"Portfolio was {random.choice(['up', 'down'])} {round(random.uniform(0.5, 5.0), 1)}% this week",
            f"Last rebalance: {random.randint(1, 10)} days ago",
            f"Trailing 30d return: {round(random.uniform(-8, 12), 1):+.1f}%",
            f"YTD return: {round(random.uniform(-15, 25), 1):+.1f}%",
        ])
        parts.append(hist)

    # 리스크 파라미터 (25% 확률)
    if random.random() < 0.25:
        risk = random.choice([
            f"Max drawdown budget: {random.choice([10, 12, 15, 20])}%",
            f"Risk limit: max leverage {round(random.uniform(1.2, 2.0), 1)}x",
            f"Mandate: {random.choice(['long-only', 'long-short', 'market-neutral', 'absolute return'])}",
            f"Volatility target: {random.choice([10, 12, 15, 18, 20])}%",
        ])
        parts.append(risk)

    if not parts:
        return ""
    return "\n".join(parts) + "\n\n"


# =============================================================================
# 종목별 다양한 reason 생성 (#3 해결)
# =============================================================================


def _filter_stocks(min_beta: float = 0, max_beta: float = 99,
                   us_corr: str | None = None,
                   cap_tiers: list[str] | None = None,
                   sectors: list[str] | None = None) -> list[str]:
    """조건에 맞는 종목 티커를 필터링합니다."""
    result = []
    for t, s in KOREAN_STOCKS.items():
        if s["beta"] < min_beta or s["beta"] > max_beta:
            continue
        if us_corr and s["us_corr"] != us_corr:
            continue
        if cap_tiers and s["cap_tier"] not in cap_tiers:
            continue
        if sectors and s["sector"] not in sectors:
            continue
        result.append(t)
    return result


def _gen_stock_reason(stock: dict, val: dict, regime: str) -> str:
    """종목 속성 + 밸류에이션 수치를 조합하여 매번 다른 reason을 생성합니다."""
    parts = []

    # 밸류에이션 기반 reason (구체적 수치 포함)
    if val["per"] < 10:
        parts.append(random.choice([
            f"deeply undervalued at PER {val['per']}x",
            f"trading at bargain PER {val['per']}x below sector average",
            f"PER {val['per']}x implies significant earnings upside",
        ]))
    elif val["per"] < 15:
        parts.append(random.choice([
            f"attractive PER {val['per']}x with earnings growth potential",
            f"reasonably valued at PER {val['per']}x",
            f"PER {val['per']}x offers margin of safety",
        ]))
    elif val["per"] > 35:
        parts.append(random.choice([
            f"premium valuation PER {val['per']}x justified by growth trajectory",
            f"high PER {val['per']}x but expanding TAM",
            f"PER {val['per']}x pricing in multi-year growth runway",
        ]))

    if val["pbr"] < 1.0:
        parts.append(random.choice([
            f"PBR {val['pbr']}x below book value (asset play)",
            f"deep value at PBR {val['pbr']}x",
        ]))
    elif val["pbr"] > 5.0:
        parts.append(f"PBR {val['pbr']}x reflects high ROE ({val['roe']:.1f}%)")

    if val["roe"] > 20:
        parts.append(random.choice([
            f"strong profitability with ROE {val['roe']:.1f}%",
            f"high-quality earnings (ROE {val['roe']:.1f}%)",
            f"best-in-class ROE {val['roe']:.1f}% in sector",
        ]))

    if val["f_score"] >= 7:
        parts.append(random.choice([
            f"strong F-Score {val['f_score']}/9 (quality + value)",
            f"Piotroski F-Score {val['f_score']}/9 signals fundamental strength",
        ]))
    elif val["f_score"] <= 3:
        parts.append(f"weak F-Score {val['f_score']}/9, trade as momentum only")

    if val["debt_equity"] < 0.3:
        parts.append(random.choice([
            f"near-zero debt (D/E {val['debt_equity']:.2f}x)",
            f"clean balance sheet with D/E {val['debt_equity']:.2f}x",
        ]))

    # regime 기반 전략적 reason
    if regime == "STRONG_BULL":
        r = random.choice([
            f"beta {stock['beta']:.1f} amplifies upside in bull regime",
            f"momentum leader with beta {stock['beta']:.1f}",
            f"overweight for max upside capture (beta {stock['beta']:.1f})",
            f"{stock['sector']} sector benefiting from regime tailwind",
            f"institutional accumulation pattern detected",
        ])
        parts.append(r)
    elif regime == "MILD_BULL":
        r = random.choice([
            f"balanced risk-reward with beta {stock['beta']:.1f}",
            f"{stock['sector']} offers value + momentum combination",
            f"quality name for balanced portfolio construction",
            f"sector rotation beneficiary in mild bull environment",
            f"steady earnings growth supporting current valuation",
        ])
        parts.append(r)
    elif regime == "WEAKENING":
        r = random.choice([
            f"low-beta ({stock['beta']:.1f}) provides downside protection",
            f"defensive {stock['sector']} sector holds up in weak markets",
            f"stable dividend yield + low volatility profile",
            f"counter-cyclical earnings expected to be resilient",
            f"beta {stock['beta']:.1f} reduces portfolio volatility",
        ])
        parts.append(r)
    elif regime == "SHORT_TERM_STRESS":
        r = random.choice([
            f"holding through stress - fundamental value unchanged",
            f"expected recovery within 5 days based on mean-reversion pattern",
            f"temporary selloff creates opportunity (intrinsic value intact)",
            f"institutional support near current levels historically",
            f"oversold on technical indicators, maintaining position",
            f"{stock['sector']} sector stress is technical, not fundamental",
        ])
        parts.append(r)

    # cap_tier 기반
    if stock["cap_tier"] == "mid_cap" and regime in ("STRONG_BULL", "MILD_BULL"):
        parts.append(random.choice([
            "mid-cap sweet spot: institutional discovery + retail momentum",
            "3000억-1조 range optimal for value + momentum capture",
            "mid-cap alpha: under-covered by sell-side analysts",
        ]))

    # US 연동성
    if stock["us_corr"] == "high" and regime in ("STRONG_BULL", "MILD_BULL"):
        parts.append(random.choice([
            "benefits from US tech cycle tailwind",
            f"high US correlation: {stock['sector']} tracks NASDAQ closely",
            "overnight gap strategy candidate due to US linkage",
        ]))

    # 최소 2개 이상 보장, 최대 4개로 제한
    if not parts:
        parts.append(f"{stock['sector']} sector exposure with beta {stock['beta']:.1f}")
    random.shuffle(parts)
    return ". ".join(parts[:random.randint(2, 4)])


def _pick_stocks(regime: str, data: dict, snapshot=None) -> dict:
    """regime에 맞는 종목 포지션을 생성합니다 (밸류에이션 포함)."""
    all_tickers = list(KOREAN_STOCKS.keys())

    if regime == "STRONG_BULL":
        pool = _filter_stocks(min_beta=1.1, cap_tiers=["mid_cap", "mid_large", "small_mid"])
        if len(pool) < 5:
            pool = _filter_stocks(min_beta=1.0)
        selected = random.sample(pool, min(random.randint(6, 10), len(pool)))
        net_exposure = round(random.uniform(1.0, 1.3), 2)
        longs = _build_long_positions(selected, net_exposure, regime, snapshot)
        return {"longs": longs, "shorts": [], "net_exposure": net_exposure}

    elif regime == "MILD_BULL":
        mid = _filter_stocks(cap_tiers=["mid_cap", "mid_large"])
        large = _filter_stocks(cap_tiers=["large_cap"])
        pool = mid + large
        selected = random.sample(pool, min(random.randint(6, 10), len(pool)))
        net_exposure = round(random.uniform(0.80, 1.0), 2)
        longs = _build_long_positions(selected, net_exposure, regime, snapshot)
        return {"longs": longs, "shorts": [], "net_exposure": net_exposure}

    elif regime == "WEAKENING":
        pool = _filter_stocks(max_beta=0.9)
        if len(pool) < 4:
            pool = _filter_stocks(max_beta=1.0)
        selected = random.sample(pool, min(random.randint(4, 7), len(pool)))
        short_pct = round(random.uniform(0.10, 0.20), 2)
        long_total = round(random.uniform(0.50, 0.70), 2)
        net_exposure = round(long_total - short_pct, 2)
        longs = _build_long_positions(selected, long_total, regime, snapshot)
        short_etf = random.choice(KRX_ETFS["short_kospi"])
        shorts = [{"ticker": short_etf["ticker"], "name": short_etf["name"],
                    "weight": short_pct,
                    "reason": random.choice([
                        f"index hedge: reduce net exposure to {net_exposure:.0%} in weakening regime",
                        f"KOSPI put equivalent via inverse ETF, targeting {short_pct:.0%} of NAV",
                        f"systematic hedge against further downside, {data['realized_vol_10d_pct']:.1f}% vol warrants protection",
                    ])}]
        return {"longs": longs, "shorts": shorts, "net_exposure": net_exposure}

    elif regime == "SHORT_TERM_STRESS":
        selected = random.sample(all_tickers, min(random.randint(5, 8), len(all_tickers)))
        short_pct = round(random.uniform(0.30, 0.50), 2)
        long_total = round(random.uniform(0.60, 0.85), 2)
        net_exposure = round(long_total - short_pct, 2)
        longs = _build_long_positions(selected, long_total, regime, snapshot)
        short_etfs = random.sample(KRX_ETFS["short_kospi"], 2)
        shorts = [
            {"ticker": etf["ticker"], "name": etf["name"],
             "weight": round(short_pct / 2, 3),
             "reason": random.choice([
                 f"MDD protection: limit drawdown while waiting for recovery (target recovery in ~{random.randint(2,5)} days)",
                 f"Sharpe improvement overlay: index short reduces portfolio vol from {data['realized_vol_10d_pct']:.1f}%",
                 f"Tactical hedge at {round(short_pct/2, 1):.0%} of NAV, will unwind on regime improvement",
             ])}
            for etf in short_etfs
        ]
        return {"longs": longs, "shorts": shorts, "net_exposure": net_exposure}

    else:  # BEAR
        short_pct = round(random.uniform(0.50, 0.80), 2)
        short_etfs = random.sample(KRX_ETFS["short_kospi"], 2)
        shorts = [
            {"ticker": etf["ticker"], "name": etf["name"],
             "weight": round(short_pct / 2, 3),
             "reason": random.choice([
                 f"full bear positioning: momentum {data['momentum_10d_pct']:+.2f}%, no recovery signal",
                 f"index short at {round(short_pct/2, 1):.0%} NAV, foreign selling {data['foreign_net_buying_bn_krw']:+,}B KRW",
                 f"maximize short exposure in sustained downturn, breadth collapsed to {data['market_breadth_pct']:.1f}%",
             ])}
            for etf in short_etfs
        ]
        return {"longs": [], "shorts": shorts, "net_exposure": round(-short_pct, 2)}


def _build_long_positions(tickers: list[str], total_weight: float, regime: str, snapshot=None) -> list[dict]:
    """종목 리스트에서 포지션을 구성합니다 (밸류에이션 포함)."""
    longs = []
    remaining = total_weight
    for i, t in enumerate(tickers):
        s = KOREAN_STOCKS[t]
        w = round(remaining / (len(tickers) - i) * random.uniform(0.7, 1.3), 3)
        w = min(w, 0.18)
        remaining -= w
        if snapshot:
            val = _gen_valuation_real(snapshot, t, s["sector"])
        else:
            val = _gen_valuation(s["sector"])
        reason = _gen_stock_reason(s, val, regime)
        longs.append({
            "ticker": t, "name": s["name"], "weight": round(w, 3),
            "valuation": {"PER": val["per"], "PBR": val["pbr"],
                          "EV/EBITDA": val["ev_ebitda"], "ROE": val["roe"],
                          "F-Score": val["f_score"]},
            "reason": reason,
        })
    return longs


def _gen_reasoning(regime: str, data: dict) -> str:
    """regime 판단 reasoning을 생성합니다. 매번 표현을 다르게 합니다."""
    parts = []
    mom = data["momentum_10d_pct"]
    vol = data["realized_vol_10d_pct"]
    foreign = data["foreign_net_buying_bn_krw"]
    inst = data["institutional_net_buying_bn_krw"]
    tg = data["telegram_sentiment"]
    br = data["market_breadth_pct"]
    usdkrw = data["usdkrw"]
    vix = data["vix"]

    # 모멘텀 (4가지 표현 변형)
    if mom > 2:
        parts.append(random.choice([
            f"10-day momentum at {mom:+.2f}% confirms strong uptrend",
            f"Sustained buying pressure with {mom:+.2f}% momentum",
            f"KOSPI/KOSDAQ momentum {mom:+.2f}% indicates broad rally phase",
            f"Positive momentum ({mom:+.2f}%) with no reversal signal yet",
        ]))
    elif mom > 0:
        parts.append(random.choice([
            f"Mildly positive momentum ({mom:+.2f}%) - uptrend intact but weakening",
            f"Momentum {mom:+.2f}%: cautious optimism warranted",
            f"Trend still positive at {mom:+.2f}%, but conviction fading",
        ]))
    elif mom > -2:
        parts.append(random.choice([
            f"Momentum turning negative ({mom:+.2f}%) - watch for regime shift",
            f"Fading momentum at {mom:+.2f}% suggests distribution phase",
            f"Negative momentum ({mom:+.2f}%) but not yet at stress levels",
        ]))
    else:
        parts.append(random.choice([
            f"Severe negative momentum ({mom:+.2f}%) confirms bear trend",
            f"Momentum collapse to {mom:+.2f}%: capitulation underway",
            f"Deep downtrend at {mom:+.2f}%, no bottom signals visible",
        ]))

    # 변동성
    if vol < 15:
        parts.append(random.choice([
            f"Low realized vol ({vol:.1f}%) supports leveraged positioning",
            f"Vol at {vol:.1f}% - risk-on environment with stable price action",
        ]))
    elif vol < 25:
        parts.append(random.choice([
            f"Elevated vol ({vol:.1f}%) warrants position size reduction",
            f"Vol rising to {vol:.1f}% - transitional market conditions",
        ]))
    else:
        parts.append(random.choice([
            f"Extreme vol ({vol:.1f}%): crisis-level uncertainty",
            f"Vol at {vol:.1f}% signals market stress - reduce exposure immediately",
        ]))

    # 수급 (다양한 표현)
    if foreign > 200:
        parts.append(random.choice([
            f"Foreign institutions net buying {foreign:+,}B KRW - strong tailwind",
            f"Sustained foreign buying ({foreign:+,}B KRW) supporting index",
        ]))
    elif foreign < -200:
        parts.append(random.choice([
            f"Foreign selling pressure at {foreign:+,}B KRW creates headwind",
            f"Heavy foreign outflow ({foreign:+,}B KRW) weighing on market",
        ]))

    if inst > 200:
        parts.append(f"Institutional buying ({inst:+,}B KRW) adds conviction")
    elif inst < -200:
        parts.append(f"Institutional net selling ({inst:+,}B KRW) signals caution")

    # 텔레그램
    if tg > 0.6:
        parts.append(random.choice([
            f"Telegram sentiment bullish ({tg:.2f}) - retail narrative positive",
            f"Social sentiment at {tg:.2f} confirms market optimism",
        ]))
    elif tg < 0.3:
        parts.append(random.choice([
            f"Telegram sentiment bearish ({tg:.2f}) - fear/panic dominant",
            f"Social sentiment collapsed to {tg:.2f}, reflecting widespread pessimism",
        ]))

    # 브레드쓰
    if br > 60:
        parts.append(f"Market breadth healthy at {br:.1f}% advancing")
    elif br < 35:
        parts.append(f"Breadth collapsed to {br:.1f}% - very narrow market")

    # 환율/VIX (새로 추가)
    if usdkrw > 1400:
        parts.append(f"Won weakness (USD/KRW {usdkrw:.0f}) pressuring import-heavy sectors")
    elif usdkrw < 1300:
        parts.append(f"Won strength (USD/KRW {usdkrw:.0f}) favorable for domestic demand")

    if vix > 30:
        parts.append(f"VIX at {vix:.1f} signals global risk-off")
    elif vix < 15:
        parts.append(f"VIX low at {vix:.1f} - global risk appetite strong")

    return ". ".join(parts) + "."


# =============================================================================
# 섹터 분석 시나리오 15개 (#5 해결)
# =============================================================================

SECTOR_SCENARIOS = [
    {"event": "NVIDIA reported record revenue with 200% YoY data center growth",
     "sector": "Semiconductor", "tickers": ["005930", "000660", "042700", "095340", "403870"],
     "direction": "positive",
     "why": "Samsung/SK Hynix supply HBM3E memory, Hanmi Semi test equipment for HBM stacking, ISC test sockets, HPSP advanced packaging equipment"},
    {"event": "TSMC raised capex guidance 30%, citing AI chip demand",
     "sector": "Semiconductor Equipment", "tickers": ["042700", "240810", "185750", "357780", "166090"],
     "direction": "positive",
     "why": "Korean equipment makers (Hanmi, Wonik IPS, Jusung) supply deposition/test tools to TSMC. Solbrain/Hana Materials supply chemicals and parts"},
    {"event": "US expanded EV tax credits under IRA",
     "sector": "Battery", "tickers": ["373220", "006400", "247540", "066970", "336370"],
     "direction": "positive",
     "why": "LG Energy/Samsung SDI have US plants. Ecopro BM, L&F supply cathode materials. Solus supplies copper foil for battery cells"},
    {"event": "Tesla cut Model Y prices globally by 5-8%",
     "sector": "Auto/Battery", "tickers": ["005380", "000270", "373220", "012330"],
     "direction": "mixed",
     "why": "EV price war pressures Hyundai/Kia margins (negative). But volume increase could benefit battery supplier LG Energy and parts maker Hyundai Mobis"},
    {"event": "Fed unexpectedly raised rates by 50bp, 10Y yield at 5.2%",
     "sector": "Financials/Defensive", "tickers": ["105560", "055550", "033780", "017670", "032830"],
     "direction": "defensive_outperform",
     "why": "KB/Shinhan benefit from NIM expansion. Defensive names KT&G and SK Telecom outperform as growth stocks de-rate. Samsung Life gains from higher bond yields"},
    {"event": "US power grid infrastructure spending bill passed ($50B package)",
     "sector": "Transformer/Grid", "tickers": ["267260", "298040", "103590", "010120", "071970"],
     "direction": "positive",
     "why": "HD Hyundai Electric, Hyosung Heavy, Ilhin Electric are top-3 transformer exporters to US. LS Electric supplies switchgear. STX Heavy for generators"},
    {"event": "NATO announced 3% GDP defense spending target",
     "sector": "Defense", "tickers": ["012450", "047810", "014880", "012750", "064350"],
     "direction": "positive",
     "why": "Hanwha Aerospace (K9 howitzer exports to Poland), KAI (FA-50 trainer jets), LIG Nex1 (missiles), Hanwha Systems (C4I), Hyundai Rotem (K2 tanks)"},
    {"event": "Japan restarted 10 nuclear reactors, Korea reversed anti-nuclear policy",
     "sector": "Nuclear", "tickers": ["034020", "950160", "071970"],
     "direction": "positive",
     "why": "Doosan Enerbility is Korea's sole nuclear reactor builder. Koltec supplies nuclear equipment. STX Heavy provides turbine generators for nuclear plants"},
    {"event": "Global shipping rates spiked 40% on Red Sea disruption",
     "sector": "Shipbuilding", "tickers": ["009540", "329180", "267250"],
     "direction": "positive",
     "why": "HD Korea Shipbuilding, HD Hyundai Heavy receive accelerated LNG carrier/container ship orders. HD Hyundai holding benefits from subsidiary valuation uplift"},
    {"event": "China travel to Korea surged 50% after visa-free entry policy",
     "sector": "Consumer/Cosmetics", "tickers": ["090430", "004170", "069960", "003490"],
     "direction": "positive",
     "why": "Amorepacific (cosmetics duty-free sales), Shinsegae/Hyundai Dept Store (luxury retail), Korean Air (inbound travel capacity)"},
    {"event": "Celltrion received FDA approval for new biosimilar",
     "sector": "Bio/Pharma", "tickers": ["068270", "207940", "196170", "145020", "326030"],
     "direction": "positive",
     "why": "Celltrion FDA approval validates Korean biosimilar pipeline. Samsung Biologics benefits as CDMO partner. Alteogen/Hugel gain on platform licensing. SK Bio on CNS pipeline"},
    {"event": "USD/KRW broke above 1450, won at 2-year low",
     "sector": "FX Sensitive", "tickers": ["005930", "000660", "005380", "090430", "004170"],
     "direction": "mixed",
     "why": "Exporters (Samsung, SK Hynix, Hyundai Motor) benefit from won weakness via translation gains. Importers and domestic consumption (cosmetics, retail) hurt by weak won"},
    {"event": "HYBE's BTS announced world tour, K-pop streaming records broken",
     "sector": "Entertainment", "tickers": ["352820", "041510", "035760"],
     "direction": "positive",
     "why": "HYBE direct beneficiary. SM Entertainment gains from sector re-rating. CJ ENM content licensing revenue expected to grow"},
    {"event": "Korea PF (project financing) default fears spread across construction sector",
     "sector": "Construction", "tickers": ["006360", "000720", "086790", "138930", "139130"],
     "direction": "negative",
     "why": "GS E&C and Hyundai E&C face PF exposure concerns. Regional banks (BNK, DGB) hold PF loans. Hana Financial has lower exposure but sentiment contagion"},
    {"event": "AI datacenter electricity demand projected to triple by 2027",
     "sector": "Power/Nuclear", "tickers": ["015760", "034020", "267260", "103590", "298040"],
     "direction": "positive",
     "why": "KEPCO benefits from electricity demand growth. Doosan Enerbility for nuclear. HD Hyundai Electric/Ilhin for transformers. Hyosung Heavy for grid infrastructure"},
    # --- 16~25: 추가 시나리오 ---
    {"event": "Apple announced iPhone production shift from China to Vietnam and India",
     "sector": "Electronics/Components", "tickers": ["005930", "009150", "006400", "010120"],
     "direction": "mixed",
     "why": "Samsung Electronics may gain share as alternative supplier. Samsung Electro-Mechanics (MLCC) could see order shifts. LG Electronics and LS Electric may benefit from supply chain diversification"},
    {"event": "China imposed rare earth export restrictions targeting semiconductor materials",
     "sector": "Materials/Supply Chain", "tickers": ["000660", "042700", "036830", "166090"],
     "direction": "negative",
     "why": "SK Hynix and Hanmi Semi face supply chain disruption risk. Soulbrain Holdings and Hana Materials need to secure alternative sourcing for critical chemicals"},
    {"event": "US FDA accelerated approval pathway for GLP-1 obesity drugs",
     "sector": "Bio/Pharma", "tickers": ["068270", "207940", "326030", "141080", "145020"],
     "direction": "positive",
     "why": "Celltrion developing GLP-1 biosimilar. Samsung Biologics positioned as CDMO partner. SK Bio and Regen Bio gain from obesity drug pipeline. Hugel benefits from aesthetic/weight-loss convergence"},
    {"event": "EU carbon border tax (CBAM) implementation accelerated",
     "sector": "Green/Industrial", "tickers": ["005490", "009540", "034020", "373220"],
     "direction": "mixed",
     "why": "POSCO Holdings faces higher costs but leads in green steel (HyREX). HD Korea Shipbuilding benefits from LNG/ammonia fuel ship orders. Doosan Enerbility for hydrogen turbines. LG Energy for ESS"},
    {"event": "BOK (Bank of Korea) unexpectedly cut rates by 50bp to support economy",
     "sector": "Rate Sensitive", "tickers": ["105560", "055550", "086790", "138930", "032830"],
     "direction": "mixed",
     "why": "KB Financial and Shinhan benefit from credit growth but NIM compression. Regional banks (BNK, DGB) see faster loan growth. Samsung Life hurt by lower reinvestment rates"},
    {"event": "Samsung Electronics announced $50B investment in Texas chip fab (3nm)",
     "sector": "Semiconductor Ecosystem", "tickers": ["005930", "042700", "240810", "185750", "403870"],
     "direction": "positive",
     "why": "Samsung foundry expansion benefits equipment suppliers Hanmi Semi, Wonik IPS, Jusung Engineering. HPSP benefits from advanced packaging demand"},
    {"event": "Global copper prices hit all-time high on AI datacenter demand",
     "sector": "Copper/Cable", "tickers": ["010120", "267260", "298040", "336370"],
     "direction": "mixed",
     "why": "LS Electric benefits as copper cable/busbar maker. HD Hyundai Electric and Hyosung Heavy face input cost pressure on transformers. Solus copper foil margins impacted"},
    {"event": "Korea-Japan semiconductor alliance announced for next-gen chip packaging",
     "sector": "Advanced Packaging", "tickers": ["005930", "042700", "403870", "095340", "357780"],
     "direction": "positive",
     "why": "Samsung and Hanmi Semi direct beneficiaries. HPSP and ISC gain from advanced packaging test demand. Solbrain supplies specialty chemicals for CoWoS process"},
    {"event": "Hyundai Motor Group announced solid-state battery mass production timeline (2027)",
     "sector": "Battery/Auto", "tickers": ["005380", "000270", "012330", "373220", "247540"],
     "direction": "mixed",
     "why": "Hyundai Motor and Kia positive on technology leadership. Hyundai Mobis integrates SSB packs. LG Energy may face disruption risk. Ecopro BM cathode demand shifts with new chemistry"},
    {"event": "Korean pension fund (NPS) announced 5% increase in domestic equity allocation",
     "sector": "Large Cap/Index", "tickers": ["005930", "000660", "005380", "105560", "068270"],
     "direction": "positive",
     "why": "NPS buying supports large-cap names disproportionately. Samsung Electronics, SK Hynix, Hyundai Motor as top-weight index names. KB Financial and Celltrion as dividend + growth plays"},
]


# =============================================================================
# 태스크별 예제 생성기
# =============================================================================


def gen_full_analysis(regime: str, snapshot=None) -> dict:
    """태스크 1: 종합 분석 (시장 데이터 + valuation → regime + 포지션 JSON)."""
    if snapshot:
        data = _gen_market_data_real(snapshot)
        regime = snapshot.regime
    else:
        data = _gen_market_data(regime)
    positions = _pick_stocks(regime, data, snapshot)
    reasoning = _gen_reasoning(regime, data)

    extras = ""
    if random.random() < 0.5:
        themes = TELEGRAM_THEMES_BULLISH if regime in ("STRONG_BULL", "MILD_BULL") else TELEGRAM_THEMES_BEARISH
        extras = f"Telegram buzz: {random.choice(themes)}"

    ctx = _gen_context_prefix(snapshot)
    user = ctx + _format_market_data_text(data, extras)
    user += "\n\n" + random.choice(QUERY_TEMPLATES["full_analysis"])

    conf_range = {
        "STRONG_BULL": (0.75, 0.95), "MILD_BULL": (0.60, 0.80),
        "WEAKENING": (0.60, 0.85), "SHORT_TERM_STRESS": (0.65, 0.85),
        "BEAR": (0.70, 0.90),
    }

    response = {
        "regime": regime,
        "confidence": round(random.uniform(*conf_range[regime]), 2),
        "reasoning": reasoning,
        "net_exposure": positions["net_exposure"],
        "positions": {"long": positions["longs"], "short": positions["shorts"]},
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2, ensure_ascii=False),
            "task_type": "full_analysis"}


def gen_regime_classification(regime: str, snapshot=None) -> dict:
    """태스크 2: Regime 분류만."""
    if snapshot:
        data = _gen_market_data_real(snapshot)
        regime = snapshot.regime
    else:
        data = _gen_market_data(regime)
    reasoning = _gen_reasoning(regime, data)

    ctx = _gen_context_prefix(snapshot)
    user = ctx + _format_market_data_text(data)
    user += "\n\n" + random.choice(QUERY_TEMPLATES["regime_classification"])

    exposure_map = {
        "STRONG_BULL": f"{random.uniform(1.0, 1.3):.0%}",
        "MILD_BULL": f"{random.uniform(0.8, 1.0):.0%}",
        "WEAKENING": f"{random.uniform(0.4, 0.7):.0%}",
        "SHORT_TERM_STRESS": f"{random.uniform(0.2, 0.5):.0%}",
        "BEAR": f"{random.uniform(-0.8, -0.5):.0%}",
    }

    response = {
        "regime": regime,
        "confidence": round(random.uniform(0.60, 0.95), 2),
        "reasoning": reasoning,
        "recommended_net_exposure": exposure_map[regime],
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2),
            "task_type": "regime_classification"}


def gen_overnight_gap(us_direction: str) -> dict:
    """태스크 3: Overnight Gap 전략 (VIX 추가)."""
    vix = round(random.uniform(12, 20) if us_direction == "bullish" else random.uniform(18, 35), 1)

    if us_direction == "bullish":
        catalyst = random.choice(US_CATALYSTS_BULLISH)
        sp = round(random.uniform(0.3, 1.5), 2)
        nq = round(random.uniform(0.4, 2.0), 2)

        if random.random() < 0.5:
            etf = random.choice(KRX_ETFS["long_us"])
            overnight_pos = {
                "direction": "long", "instrument": etf["name"], "ticker": etf["ticker"],
                "weight": round(random.uniform(0.10, 0.20), 2),
                "reason": random.choice([
                    f"Capture expected US rally via KRX ETF. S&P futures {sp:+.2f}% with VIX at {vix}",
                    f"US momentum trade: {catalyst}. Low VIX ({vix}) supports overnight carry",
                    f"Tactical long US exposure at {sp:+.2f}% futures signal. Position sizing conservative at 10-20% NAV",
                ]),
            }
        else:
            us_stocks = _filter_stocks(us_corr="high")
            picks = random.sample(us_stocks, min(3, len(us_stocks)))
            overnight_pos = {
                "direction": "long",
                "instruments": [
                    {"ticker": t, "name": KOREAN_STOCKS[t]["name"],
                     "weight": round(random.uniform(0.03, 0.07), 3),
                     "sector": KOREAN_STOCKS[t]["sector"]}
                    for t in picks
                ],
                "reason": random.choice([
                    f"Hold US-correlated Korean names overnight. {catalyst}",
                    f"Korean {KOREAN_STOCKS[picks[0]]['sector']} stocks to gap up on US strength ({sp:+.2f}% futures)",
                    f"Overnight carry in high-US-correlation names. NASDAQ futures {nq:+.2f}%",
                ]),
            }
    else:
        catalyst = random.choice(US_CATALYSTS_BEARISH)
        sp = round(random.uniform(-1.5, -0.3), 2)
        nq = round(random.uniform(-2.0, -0.4), 2)

        etf = random.choice(KRX_ETFS["inverse_us"])
        overnight_pos = {
            "direction": "short_via_inverse", "instrument": etf["name"],
            "ticker": etf["ticker"],
            "weight": round(random.uniform(0.10, 0.18), 2),
            "reason": random.choice([
                f"Hedge overnight via inverse ETF. VIX spiked to {vix}, S&P futures {sp:+.2f}%",
                f"Protect against US decline: {catalyst}. Inverse position sizes at 10-18% NAV",
                f"Risk-off overnight: {catalyst}. Also reducing high-beta Korean exposure",
            ]),
            "additional_action": random.choice([
                "Reduce overnight exposure in high-beta Korean names (semi equipment, battery cathode)",
                "Trim KOSDAQ momentum names before close to limit overnight gap risk",
                "Set stop-loss on remaining overnight longs at -3% from close",
            ]),
        }

    user = (
        f"It is 15:20 KST (Korean market closing soon).\n"
        f"US pre-market signals:\n"
        f"- S&P 500 futures: {sp:+.2f}%\n"
        f"- NASDAQ futures: {nq:+.2f}%\n"
        f"- VIX: {vix}\n"
        f"- Catalyst: {catalyst}\n\n"
        f"{random.choice(QUERY_TEMPLATES['overnight_gap'])}"
    )
    response = {
        "overnight_strategy": overnight_pos,
        "unwind_plan": random.choice([
            "Unwind at Korean open 09:00-09:10 KST once gap is realized",
            "Monitor pre-market and unwind within first 30 minutes of Korean session",
            "Set limit orders at gap target level for automatic unwind at open",
        ]),
        "max_overnight_nav_pct": "10-20%",
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2, ensure_ascii=False),
            "task_type": "overnight_gap"}


def gen_risk_management(snapshot=None) -> dict:
    """태스크 4: 리스크 관리."""
    current_mdd = round(random.uniform(5, 30), 1)
    current_leverage = round(random.uniform(1.0, 2.2), 2)
    nav = round(random.uniform(1.0, 10.0), 1)
    trailing_sharpe = round(random.uniform(-0.5, 2.5), 2)
    regime = random.choice(REGIMES)
    if snapshot:
        data = _gen_market_data_real(snapshot)
        regime = snapshot.regime
    else:
        data = _gen_market_data(regime)
    max_lev = min(2.0, round(100 / current_mdd, 2)) if current_mdd > 0 else 2.0
    should_reduce = current_leverage > max_lev

    ctx = _gen_context_prefix(snapshot)
    user = (
        f"{ctx}Portfolio status:\n"
        f"- NAV: {nav:.1f}B KRW\n"
        f"- Current leverage: {current_leverage:.2f}x\n"
        f"- Trailing MDD: {current_mdd:.1f}%\n"
        f"- Trailing Sharpe: {trailing_sharpe:.2f}\n"
        f"- Current regime: {regime}\n\n"
        f"{_format_market_data_text(data)}\n\n"
        f"{random.choice(QUERY_TEMPLATES['risk_management'])}"
    )

    if should_reduce:
        action = f"REDUCE leverage from {current_leverage:.2f}x to {max_lev:.2f}x"
        reasoning = random.choice([
            f"Current leverage ({current_leverage:.2f}x) exceeds NAV/MDD limit (100/{current_mdd:.1f}% = {max_lev:.2f}x). Must reduce exposure by selling {(current_leverage - max_lev) / current_leverage * 100:.0f}% of positions",
            f"Risk limit breach: {current_leverage:.2f}x vs max {max_lev:.2f}x. Prioritize selling high-beta names first to reduce portfolio vol simultaneously",
            f"MDD at {current_mdd:.1f}% requires max leverage {max_lev:.2f}x but currently at {current_leverage:.2f}x. Execute reduction over today's session",
        ])
    elif regime in ("STRONG_BULL", "MILD_BULL") and current_leverage < max_lev * 0.8:
        action = f"MAINTAIN or INCREASE (room up to {max_lev:.2f}x)"
        reasoning = random.choice([
            f"Leverage ({current_leverage:.2f}x) well within limits ({max_lev:.2f}x). {regime} regime with Sharpe {trailing_sharpe:.2f} supports maintaining/increasing exposure",
            f"Favorable risk-reward: current {current_leverage:.2f}x vs max {max_lev:.2f}x, bullish regime. Can add {(max_lev - current_leverage):.2f}x if conviction rises",
            f"Risk budget available: {current_leverage:.2f}x/{max_lev:.2f}x utilized. Consider adding momentum mid-caps with remaining capacity",
        ])
    else:
        action = f"MAINTAIN at {current_leverage:.2f}x"
        reasoning = random.choice([
            f"Leverage ({current_leverage:.2f}x) within max ({max_lev:.2f}x). Regime {regime} does not warrant increase. Monitor for regime change signals",
            f"Current positioning appropriate for {regime} regime at {current_leverage:.2f}x. No adjustment needed until regime shifts",
            f"Risk-neutral: {current_leverage:.2f}x leverage, {current_mdd:.1f}% MDD, {regime} regime. Hold steady and wait for clearer directional signal",
        ])

    response = {
        "risk_assessment": {
            "max_allowed_leverage": max_lev,
            "current_leverage": current_leverage,
            "mdd_rule": f"100 / {current_mdd:.1f}% = {max_lev:.2f}x",
            "within_limits": not should_reduce,
            "sharpe": trailing_sharpe,
        },
        "action": action, "reasoning": reasoning, "regime": regime,
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2),
            "task_type": "risk_management"}


def gen_regime_transition(snapshot=None) -> dict:
    """태스크 5: Regime 전환 시 포지션 조정."""
    transitions = [
        ("MILD_BULL", "WEAKENING"), ("MILD_BULL", "SHORT_TERM_STRESS"),
        ("STRONG_BULL", "WEAKENING"), ("WEAKENING", "BEAR"),
        ("WEAKENING", "MILD_BULL"), ("SHORT_TERM_STRESS", "BEAR"),
        ("SHORT_TERM_STRESS", "MILD_BULL"), ("BEAR", "SHORT_TERM_STRESS"),
        ("BEAR", "WEAKENING"), ("STRONG_BULL", "MILD_BULL"),
    ]
    from_r, to_r = random.choice(transitions)
    data_before = _gen_market_data(from_r)
    if snapshot:
        data_after = _gen_market_data_real(snapshot)
        to_r = snapshot.regime
    else:
        data_after = _gen_market_data(to_r)

    user = (
        f"Regime transition detected:\n"
        f"- Previous: {from_r}\n- New: {to_r}\n\n"
        f"Previous: KOSPI {data_before['kospi_change_pct']:+.2f}%, "
        f"momentum {data_before['momentum_10d_pct']:+.2f}%, vol {data_before['realized_vol_10d_pct']:.1f}%\n"
        f"Current: KOSPI {data_after['kospi_change_pct']:+.2f}%, "
        f"momentum {data_after['momentum_10d_pct']:+.2f}%, vol {data_after['realized_vol_10d_pct']:.1f}%\n\n"
        f"{random.choice(QUERY_TEMPLATES['regime_transition'])}"
    )

    pos = _pick_stocks(to_r, data_after, snapshot)
    if to_r in ("BEAR", "SHORT_TERM_STRESS"):
        urgency = "HIGH - execute within 1 trading day"
        day1_pct = round(random.uniform(0.70, 0.90), 2)
    else:
        urgency = "MODERATE - execute over 1-2 trading days"
        day1_pct = round(random.uniform(0.50, 0.70), 2)

    response = {
        "transition": {"from": from_r, "to": to_r},
        "urgency": urgency,
        "execution_plan": {
            "day_1": {"pct": f"{day1_pct:.0%}", "actions": _transition_actions(from_r, to_r, "day1")},
            "day_2": {"pct": f"{1 - day1_pct:.0%}", "actions": _transition_actions(from_r, to_r, "day2")},
        },
        "target_net_exposure": pos["net_exposure"],
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2),
            "task_type": "regime_transition"}


def _transition_actions(from_r: str, to_r: str, day: str) -> list[str]:
    """전환 시 구체적 액션을 생성합니다. 매번 다르게."""
    sell_candidates = random.choice([
        "battery cathode (Ecopro BM, L&F)", "semi equipment (Hanmi, Wonik IPS, HPSP)",
        "high-beta mid-caps above 1.5 beta", "KOSDAQ momentum names with stretched valuations",
        "growth names trading above PER 30x",
    ])
    buy_candidates = random.choice([
        "KB Financial, Shinhan (low-beta financials)", "KT&G, SK Telecom (defensive yield)",
        "Samsung Electronics, POSCO Holdings (value anchors)", "large-cap quality with F-Score 7+",
        "low-vol utilities and telecom names",
    ])

    if to_r in ("WEAKENING", "SHORT_TERM_STRESS", "BEAR"):
        if day == "day1":
            actions = [f"Sell {sell_candidates}"]
            if to_r != "WEAKENING":
                actions.append(random.choice([
                    "Initiate KODEX 200 Inverse position at 15-20% of NAV",
                    "Buy TIGER 200 Inverse as index hedge (20-30% of NAV)",
                ]))
            if to_r == "BEAR":
                actions.append("Begin liquidating all individual stock longs via TWAP")
        else:
            actions = [f"Rotate remaining longs to {buy_candidates}"]
            if to_r in ("SHORT_TERM_STRESS", "BEAR"):
                actions.append("Increase index short to full target weight")
    else:
        if day == "day1":
            actions = [
                random.choice(["Close KODEX 200 Inverse position", "Reduce index short by 70%"]),
                f"Begin building long in {sell_candidates.replace('high-beta ', '')}",
            ]
        else:
            actions = ["Complete long portfolio construction", "Close any remaining hedges"]
    return actions


def gen_sector_analysis(snapshot=None) -> dict:
    """태스크 6: US 이벤트 → 한국 종목 영향 분석 (15개 시나리오)."""
    sc = random.choice(SECTOR_SCENARIOS)
    user = f"US/Global event: {sc['event']}\n\n{random.choice(QUERY_TEMPLATES['sector_analysis'])}"

    affected = []
    for t in sc["tickers"]:
        if t not in KOREAN_STOCKS:
            continue
        s = KOREAN_STOCKS[t]
        if snapshot:
            val = _gen_valuation_real(snapshot, t, s["sector"])
        else:
            val = _gen_valuation(s["sector"])
        affected.append({
            "ticker": t, "name": s["name"], "sector": s["sector"],
            "cap_tier": s["cap_tier"], "impact": sc["direction"],
            "valuation": {"PER": val["per"], "PBR": val["pbr"], "F-Score": val["f_score"]},
        })

    response = {
        "event": sc["event"], "primary_sector": sc["sector"], "direction": sc["direction"],
        "affected_stocks": affected, "reasoning": sc["why"],
        "action": "Increase weight in affected names" if "positive" in sc["direction"]
                  else "Rotate to defensive / reduce affected exposure",
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2, ensure_ascii=False),
            "task_type": "sector_analysis"}


def gen_opening_momentum(snapshot=None) -> dict:
    """태스크 7: 시초가 모멘텀 분석."""
    mid_caps = _filter_stocks(cap_tiers=["mid_cap", "small_mid"])
    gap_tickers = random.sample(mid_caps, min(random.randint(2, 4), len(mid_caps)))

    gap_data = []
    for t in gap_tickers:
        s = KOREAN_STOCKS[t]
        if snapshot:
            val = _gen_valuation_real(snapshot, t, s["sector"])
        else:
            val = _gen_valuation(s["sector"])
        gap_data.append({
            "ticker": t, "name": s["name"],
            "gap_pct": round(random.uniform(2.0, 8.0), 2),
            "vol_ratio": round(random.uniform(2.0, 6.0), 1),
            "sector": s["sector"], "val": val,
        })

    catalysts = [
        f"Overnight US {random.choice(['tech', 'semiconductor', 'EV/battery'])} rally ({round(random.uniform(1.5, 3.0), 1):+.1f}%) driving Korean names",
        f"Pre-market institutional block buying {round(random.uniform(10, 50))},000 shares detected",
        f"Positive Q{random.randint(1,4)} earnings surprise: operating profit +{random.randint(20,80)}% YoY",
        f"Trending in Korean finance Telegram channels since 07:00 KST ({random.randint(500, 2000)} mentions)",
        f"Related US peer ({random.choice(['NVIDIA', 'Tesla', 'Broadcom', 'ASML'])}) announced strong guidance",
        f"Upgraded to Overweight by {random.choice(['Morgan Stanley', 'Goldman Sachs', 'JP Morgan', 'Samsung Securities', 'Mirae Asset'])} with TP raised {random.randint(20,50)}%",
    ]
    catalyst = random.choice(catalysts)

    gaps_text = "\n".join([
        f"  - {g['name']} ({g['ticker']}): gap +{g['gap_pct']:.1f}%, vol {g['vol_ratio']:.1f}x avg"
        for g in gap_data
    ])
    user = (
        f"Opening momentum alert (09:05 KST):\n{gaps_text}\n\n"
        f"Catalyst: {catalyst}\n\n{random.choice(QUERY_TEMPLATES['opening_momentum'])}"
    )

    analysis = []
    for g in gap_data:
        score = 0
        reasons = []
        if g["gap_pct"] > 3:
            score += 1
            reasons.append(f"strong gap {g['gap_pct']:+.1f}%")
        if g["vol_ratio"] > 3:
            score += 1
            reasons.append(f"volume {g['vol_ratio']:.1f}x average")
        if KOREAN_STOCKS[g["ticker"]]["us_corr"] == "high":
            score += 1
            reasons.append("high US correlation - gap likely sustained")
        if KOREAN_STOCKS[g["ticker"]]["cap_tier"] == "mid_cap":
            score += 1
            reasons.append("mid-cap (3000억-1조) momentum sweet spot")
        if g["val"]["f_score"] >= 7:
            score += 1
            reasons.append(f"F-Score {g['val']['f_score']}/9 (strong fundamentals)")
        if g["val"]["per"] < 15:
            reasons.append(f"PER {g['val']['per']}x still attractive post-gap")

        analysis.append({
            "ticker": g["ticker"], "name": g["name"],
            "signal": "STRONG" if score >= 3 else ("MODERATE" if score >= 2 else "WEAK"),
            "action": "ADD" if score >= 2 else "MONITOR",
            "weight": round(random.uniform(0.03, 0.08), 3) if score >= 2 else 0,
            "reasons": reasons,
            "stop_loss": f"{g['gap_pct'] * -0.5:.1f}% (gap fill level)" if score >= 2 else "N/A",
        })

    response = {
        "opening_analysis": analysis,
        "recommendation": random.choice([
            "Add to strong-signal names within first 30 minutes. Set stop-loss at gap fill level. Max 3-8% NAV per name",
            "Selectively enter high-conviction gaps. Avoid chasing weak signals. Monitor volume follow-through by 09:30",
            "Focus on mid-cap names with fundamental backing (F-Score 7+). Scale in over 09:00-09:30 window",
        ]),
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2, ensure_ascii=False),
            "task_type": "opening_momentum"}


# =============================================================================
# 신규 태스크 (#6)
# =============================================================================


def gen_fx_impact() -> dict:
    """태스크 8: 환율 변동 → 수출/내수 영향 분석."""
    usdkrw = round(random.uniform(1280, 1480), 1)
    usdkrw_chg = round(random.choice([-1, 1]) * random.uniform(5, 25), 1)
    direction = "weakening" if usdkrw_chg > 0 else "strengthening"

    user = (
        f"USD/KRW moved to {usdkrw} ({usdkrw_chg:+.1f} over past week).\n"
        f"Won is {direction} against the dollar.\n\n"
        f"{random.choice(QUERY_TEMPLATES['fx_impact'])}"
    )

    exporters = _filter_stocks(us_corr="high")
    domestic = _filter_stocks(us_corr="low", max_beta=0.9)

    if direction == "weakening":
        beneficiaries = random.sample(exporters, min(4, len(exporters)))
        losers = random.sample(domestic, min(3, len(domestic)))
        bene_list = [{"ticker": t, "name": KOREAN_STOCKS[t]["name"],
                      "sector": KOREAN_STOCKS[t]["sector"],
                      "fx_impact": random.choice([
                          f"Won weakness boosts export revenue translation by ~{random.randint(2,5)}%",
                          f"USD-denominated revenue gains from {usdkrw_chg:+.1f} won move",
                          f"Competitive pricing advantage in global markets",
                      ])} for t in beneficiaries]
        loser_list = [{"ticker": t, "name": KOREAN_STOCKS[t]["name"],
                       "sector": KOREAN_STOCKS[t]["sector"],
                       "fx_impact": random.choice([
                           f"Import cost pressure from won weakness",
                           f"Domestic consumption dampened by currency depreciation",
                           f"Raw material import costs rise with weak won",
                       ])} for t in losers]
        action = "Overweight exporters (semiconductor, auto, battery). Underweight import-heavy names"
    else:
        beneficiaries = random.sample(domestic, min(4, len(domestic)))
        losers = random.sample(exporters, min(3, len(exporters)))
        bene_list = [{"ticker": t, "name": KOREAN_STOCKS[t]["name"],
                      "sector": KOREAN_STOCKS[t]["sector"],
                      "fx_impact": random.choice([
                          f"Strong won reduces import costs, boosting margins",
                          f"Domestic purchasing power increases",
                          f"Consumer sentiment improves with won strength",
                      ])} for t in beneficiaries]
        loser_list = [{"ticker": t, "name": KOREAN_STOCKS[t]["name"],
                       "sector": KOREAN_STOCKS[t]["sector"],
                       "fx_impact": random.choice([
                           f"Export competitiveness hurt by strong won",
                           f"Translated revenue declines by ~{random.randint(2,5)}%",
                           f"Margin compression from strong won",
                       ])} for t in losers]
        action = "Overweight domestic consumption names. Underweight export-heavy semiconductor/auto"

    response = {
        "fx_analysis": {
            "usdkrw": usdkrw,
            "weekly_change": usdkrw_chg,
            "direction": f"Won {direction}",
        },
        "beneficiaries": bene_list,
        "negatively_affected": loser_list,
        "portfolio_action": action,
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2, ensure_ascii=False),
            "task_type": "fx_impact"}


def gen_stop_loss_management(snapshot=None) -> dict:
    """태스크 9: 기존 포지션 손절/익절 판단."""
    n_positions = random.randint(4, 7)
    tickers = random.sample(list(KOREAN_STOCKS.keys()), n_positions)

    positions = []
    for t in tickers:
        s = KOREAN_STOCKS[t]
        entry_price = round(random.uniform(10000, 500000), 0)
        pnl_pct = round(random.uniform(-15, 25), 2)
        current_price = round(entry_price * (1 + pnl_pct / 100), 0)
        hold_days = random.randint(1, 20)
        if snapshot:
            val = _gen_valuation_real(snapshot, t, s["sector"])
        else:
            val = _gen_valuation(s["sector"])
        positions.append({
            "ticker": t, "name": s["name"], "sector": s["sector"],
            "entry_price": int(entry_price), "current_price": int(current_price),
            "pnl_pct": pnl_pct, "hold_days": hold_days,
            "weight": round(random.uniform(0.03, 0.12), 3),
            "per": val["per"], "f_score": val["f_score"],
        })

    regime = random.choice(REGIMES)
    pos_text = "\n".join([
        f"  {p['name']} ({p['ticker']}): entry {p['entry_price']:,}, "
        f"current {p['current_price']:,} ({p['pnl_pct']:+.2f}%), "
        f"{p['hold_days']}d held, PER {p['per']}x, weight {p['weight']:.1%}"
        for p in positions
    ])

    user = (
        f"Current regime: {regime}\n"
        f"Current positions:\n{pos_text}\n\n"
        f"{random.choice(QUERY_TEMPLATES['stop_loss_management'])}"
    )

    decisions = []
    for p in positions:
        if p["pnl_pct"] < -8:
            action = "STOP_LOSS"
            reason = random.choice([
                f"Loss exceeds -8% threshold at {p['pnl_pct']:+.2f}%. Cut to preserve capital",
                f"Momentum broken: {p['pnl_pct']:+.2f}% loss over {p['hold_days']} days. No recovery catalyst",
                f"Risk management rule: exit at {p['pnl_pct']:+.2f}% to limit portfolio MDD impact",
            ])
        elif p["pnl_pct"] > 15:
            action = "TAKE_PROFIT"
            reason = random.choice([
                f"Lock in {p['pnl_pct']:+.2f}% gain. Consider trailing stop at {p['pnl_pct']*0.5:+.1f}%",
                f"Target reached at {p['pnl_pct']:+.2f}%. Sell 50%, hold rest with trailing stop",
                f"Strong gain ({p['pnl_pct']:+.2f}%) in {p['hold_days']} days. Valuation stretched at PER {p['per']}x",
            ])
        elif p["pnl_pct"] < -3 and regime in ("WEAKENING", "BEAR"):
            action = "REDUCE"
            reason = random.choice([
                f"Bearish regime + loss ({p['pnl_pct']:+.2f}%). Reduce by 50% to limit further downside",
                f"{regime} regime increases loss probability. Trim from {p['weight']:.1%} to {p['weight']/2:.1%}",
            ])
        elif p["f_score"] >= 7 and p["pnl_pct"] > -3:
            action = "HOLD"
            reason = f"F-Score {p['f_score']}/9 supports thesis. P&L {p['pnl_pct']:+.2f}% within tolerance. Hold"
        else:
            action = "HOLD"
            reason = f"P&L {p['pnl_pct']:+.2f}% within stop-loss range. Monitoring for {regime} regime signal"

        decisions.append({"ticker": p["ticker"], "name": p["name"], "action": action, "reason": reason})

    response = {"position_review": decisions, "regime": regime}
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2, ensure_ascii=False),
            "task_type": "stop_loss_management"}


def gen_intraday_rebalancing(snapshot=None) -> dict:
    """태스크 10: 장중 포트폴리오 리밸런싱."""
    regime = random.choice(REGIMES)
    if snapshot:
        data = _gen_market_data_real(snapshot)
        regime = snapshot.regime
    else:
        data = _gen_market_data(regime)
    time_kst = f"{random.randint(10, 14)}:{random.choice(['00', '15', '30', '45'])} KST"

    n_current = random.randint(5, 8)
    current_tickers = random.sample(list(KOREAN_STOCKS.keys()), n_current)

    current_positions = []
    for t in current_tickers:
        s = KOREAN_STOCKS[t]
        intraday_pnl = round(random.uniform(-5, 5), 2)
        current_positions.append({
            "ticker": t, "name": s["name"], "sector": s["sector"],
            "weight": round(random.uniform(0.05, 0.15), 3),
            "intraday_pnl_pct": intraday_pnl,
        })

    pos_text = "\n".join([
        f"  {p['name']} ({p['ticker']}): weight {p['weight']:.1%}, "
        f"intraday {p['intraday_pnl_pct']:+.2f}%"
        for p in current_positions
    ])

    # 장중 이벤트
    events = [
        f"Foreign selling accelerated: {abs(data['foreign_net_buying_bn_krw'])}B KRW sold in last hour",
        f"KOSDAQ suddenly dropped {abs(data['kosdaq_change_pct']):.1f}% on margin call fears",
        f"Samsung Electronics broke key support level at KOSPI {round(random.uniform(2400, 2800))}",
        f"Telegram channels reporting institutional block buy in {random.choice(['battery', 'semiconductor', 'defense'])} names",
        f"USD/KRW just broke {int(data['usdkrw'])} level, won weakening sharply",
        f"Program selling spike: {random.randint(200, 500)}B KRW in last 30 minutes",
        f"Positive economic data released: GDP growth {round(random.uniform(2.0, 3.5), 1)}% (above {round(random.uniform(1.5, 2.5), 1)}% consensus)",
    ]
    event = random.choice(events)

    user = (
        f"Time: {time_kst}\n"
        f"Current regime: {regime}\n\n"
        f"Intraday event: {event}\n\n"
        f"{_format_market_data_text(data)}\n\n"
        f"Current positions:\n{pos_text}\n\n"
        f"{random.choice(QUERY_TEMPLATES['intraday_rebalancing'])}"
    )

    # 리밸런싱 결정
    actions = []
    for p in current_positions:
        if p["intraday_pnl_pct"] < -3 and regime in ("WEAKENING", "BEAR"):
            actions.append({
                "ticker": p["ticker"], "name": p["name"],
                "action": "REDUCE",
                "from_weight": p["weight"],
                "to_weight": round(p["weight"] * 0.5, 3),
                "reason": f"Intraday loss {p['intraday_pnl_pct']:+.2f}% in {regime} regime. Cut exposure",
            })
        elif p["intraday_pnl_pct"] > 3 and KOREAN_STOCKS[p["ticker"]]["beta"] > 1.3:
            actions.append({
                "ticker": p["ticker"], "name": p["name"],
                "action": "TRIM",
                "from_weight": p["weight"],
                "to_weight": round(p["weight"] * 0.7, 3),
                "reason": f"Lock partial profit at {p['intraday_pnl_pct']:+.2f}%. High-beta ({KOREAN_STOCKS[p['ticker']]['beta']:.1f}) vulnerable to reversal",
            })

    if not actions:
        actions.append({
            "action": "NO_CHANGE",
            "reason": f"Intraday moves within normal range for {regime} regime. No rebalancing needed. Next review at market close",
        })

    response = {
        "rebalancing_decision": actions,
        "overall_assessment": random.choice([
            f"Intraday event is significant - execute rebalancing via TWAP over next 30 minutes",
            f"Minor adjustment needed - current positioning broadly appropriate for {regime}",
            f"Defensive adjustment: reduce high-beta names and increase cash buffer",
            f"No immediate action needed. Monitor for further deterioration before close",
        ]),
        "regime": regime,
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2, ensure_ascii=False),
            "task_type": "intraday_rebalancing"}


# =============================================================================
# 신규 태스크: What-if 시나리오 & 멀티스텝
# =============================================================================


def gen_what_if(snapshot=None) -> dict:
    """태스크 11: What-if 시나리오 분석 — 가상 상황 변화에 따른 대응."""
    regime = random.choice(REGIMES)
    if snapshot:
        data = _gen_market_data_real(snapshot)
        regime = snapshot.regime
    else:
        data = _gen_market_data(regime)

    # 시나리오 유형 선택
    scenario_type = random.choice([
        "market_drop", "market_rally", "regime_shift",
        "fx_shock", "vol_spike", "sector_crash", "foreign_exit",
    ])

    if scenario_type == "market_drop":
        drop = round(random.uniform(2, 5), 1)
        scenario = f"What if KOSPI drops another {drop}% from here?"
        hypothetical_data = data.copy()
        hypothetical_data["kospi_change_pct"] = round(data["kospi_change_pct"] - drop, 2)
        hypothetical_data["market_breadth_pct"] = max(5, round(data["market_breadth_pct"] - random.uniform(10, 25), 1))
        new_regime = "BEAR" if drop > 3 else "SHORT_TERM_STRESS"
    elif scenario_type == "market_rally":
        rally = round(random.uniform(1.5, 4), 1)
        scenario = f"What if KOSPI rallies {rally}% tomorrow on stimulus announcement?"
        hypothetical_data = data.copy()
        hypothetical_data["kospi_change_pct"] = round(data["kospi_change_pct"] + rally, 2)
        hypothetical_data["market_breadth_pct"] = min(90, round(data["market_breadth_pct"] + random.uniform(10, 20), 1))
        new_regime = "STRONG_BULL"
    elif scenario_type == "regime_shift":
        target = random.choice([r for r in REGIMES if r != regime])
        scenario = f"If regime shifts from {regime} to {target}, how should we reposition?"
        hypothetical_data = _gen_market_data(target)
        new_regime = target
    elif scenario_type == "fx_shock":
        fx_move = round(random.uniform(20, 50), 0)
        direction = random.choice(["up", "down"])
        scenario = f"What if USD/KRW moves {fx_move} won {'higher' if direction == 'up' else 'lower'} this week?"
        hypothetical_data = data.copy()
        hypothetical_data["usdkrw"] = round(data["usdkrw"] + (fx_move if direction == "up" else -fx_move), 1)
        new_regime = regime
    elif scenario_type == "vol_spike":
        new_vol = round(random.uniform(30, 50), 1)
        scenario = f"What if realized vol spikes to {new_vol}% from current {data['realized_vol_10d_pct']:.1f}%?"
        hypothetical_data = data.copy()
        hypothetical_data["realized_vol_10d_pct"] = new_vol
        hypothetical_data["vix"] = round(random.uniform(25, 40), 1)
        new_regime = "SHORT_TERM_STRESS" if new_vol > 35 else regime
    elif scenario_type == "sector_crash":
        sector = random.choice(["semiconductor", "battery", "bio/pharma", "construction", "financials"])
        scenario = f"What if the {sector} sector drops 8-10% on negative catalyst?"
        hypothetical_data = data.copy()
        hypothetical_data["kospi_change_pct"] = round(data["kospi_change_pct"] - random.uniform(1, 3), 2)
        new_regime = "WEAKENING" if regime in ("MILD_BULL", "STRONG_BULL") else regime
    else:  # foreign_exit
        outflow = random.randint(800, 2000)
        scenario = f"What if foreigners sell {outflow}B KRW in a single session?"
        hypothetical_data = data.copy()
        hypothetical_data["foreign_net_buying_bn_krw"] = -outflow
        hypothetical_data["kospi_change_pct"] = round(data["kospi_change_pct"] - random.uniform(1, 3), 2)
        new_regime = "SHORT_TERM_STRESS"

    ctx = _gen_context_prefix(snapshot)
    user = (
        f"{ctx}Current state:\n{_format_market_data_text(data)}\n\n"
        f"Scenario: {scenario}\n\n"
        f"How should we adjust the portfolio?"
    )

    positions = _pick_stocks(new_regime, hypothetical_data, snapshot)
    reasoning = _gen_reasoning(new_regime, hypothetical_data)

    response = {
        "scenario_analysis": {
            "current_regime": regime,
            "hypothetical_regime": new_regime,
            "scenario": scenario,
        },
        "recommended_action": {
            "regime_shift": regime != new_regime,
            "new_net_exposure": positions["net_exposure"],
            "reasoning": reasoning,
        },
        "positions": {"long": positions["longs"], "short": positions["shorts"]},
        "contingency": random.choice([
            "Set stop-losses at -3% on all new positions. Re-evaluate if scenario doesn't materialize within 2 days",
            "Execute 50% immediately, remaining 50% contingent on scenario confirmation",
            "Use TWAP execution over 2 sessions to minimize market impact",
            "Pre-set limit orders at target levels. Monitor for reversal signals before full commitment",
            "Implement via options overlay if available, direct positions otherwise",
        ]),
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2, ensure_ascii=False),
            "task_type": "what_if"}


def gen_multi_step(snapshot=None) -> dict:
    """태스크 12: 멀티스텝 분석 — regime 판단 → 포지션 → 리스크 체크를 순차적으로."""
    regime = random.choice(REGIMES)
    if snapshot:
        data = _gen_market_data_real(snapshot)
        regime = snapshot.regime
    else:
        data = _gen_market_data(regime)

    # 다양한 멀티스텝 질문 형식
    multi_queries = [
        "Step 1: Classify the regime.\nStep 2: Recommend positions with valuations.\nStep 3: Set risk parameters (max leverage, stop-loss levels).",
        "First, determine the market regime. Then, build a portfolio. Finally, define exit criteria for each position.",
        "Three tasks:\n1) Regime classification with confidence\n2) Long/short portfolio construction\n3) Risk limits and position sizing rules",
        "Analyze in order: (a) regime assessment, (b) sector allocation, (c) individual stock picks with PER/PBR, (d) risk management plan.",
        "Do a full top-down analysis:\n- Macro regime call\n- Sector rotation view\n- Stock-level positions\n- Risk budget allocation",
        "Walk me through: What's the regime? → What should we own? → How much risk can we take? → When do we exit?",
        "Sequential analysis needed:\n1. Market regime and confidence level\n2. Target net exposure and hedging\n3. Top 5-7 long picks with valuation support\n4. Stop-loss and take-profit levels",
        "Provide a complete investment memo:\n- Market assessment\n- Conviction level\n- Recommended portfolio\n- Risk controls\n- Review triggers",
    ]

    ctx = _gen_context_prefix(snapshot)
    extras = ""
    if random.random() < 0.5:
        themes = TELEGRAM_THEMES_BULLISH if regime in ("STRONG_BULL", "MILD_BULL") else TELEGRAM_THEMES_BEARISH
        extras = f"Telegram buzz: {random.choice(themes)}"

    user = ctx + _format_market_data_text(data, extras) + "\n\n" + random.choice(multi_queries)

    positions = _pick_stocks(regime, data, snapshot)
    reasoning = _gen_reasoning(regime, data)

    # 리스크 파라미터 생성
    current_mdd = round(random.uniform(5, 25), 1)
    max_lev = min(2.0, round(100 / current_mdd, 2)) if current_mdd > 0 else 2.0

    response = {
        "step_1_regime": {
            "regime": regime,
            "confidence": round(random.uniform(0.60, 0.95), 2),
            "reasoning": reasoning,
        },
        "step_2_portfolio": {
            "net_exposure": positions["net_exposure"],
            "positions": {"long": positions["longs"], "short": positions["shorts"]},
        },
        "step_3_risk": {
            "max_leverage": max_lev,
            "position_stop_loss": f"-{random.choice([5, 7, 8, 10])}% per position",
            "portfolio_stop_loss": f"-{random.choice([3, 5, 7])}% daily NAV",
            "review_trigger": random.choice([
                "Regime change signal or 2-day consecutive drawdown > 3%",
                "VIX above 30 or foreign selling > 500B KRW/day for 3 days",
                "Momentum reversal (10d momentum sign flip) or breadth < 25%",
                "Net exposure drifts more than 20% from target",
            ]),
        },
    }
    return {"system": QUANT_SYSTEM_PROMPT, "user": user,
            "assistant": json.dumps(response, indent=2, ensure_ascii=False),
            "task_type": "multi_step"}


# =============================================================================
# 메인 생성 함수
# =============================================================================


def _maybe_snapshot(use_real: bool, regime: str | None = None, ratio: float = 0.7):
    """use_real이면 snapshot을 반환, 아니면 None."""
    if use_real and _snapshot_store and random.random() < ratio:
        return _snapshot_store.sample_snapshot(regime=regime)
    return None


def generate_all(
    count: int = 5000,
    use_real_data: bool = False,
    real_data_ratio: float = 0.7,
) -> list[dict]:
    """전체 synthetic 데이터셋을 생성합니다 (12개 태스크)."""
    examples = []
    task_counts = {
        "full_analysis": int(count * 0.22),
        "regime_classification": int(count * 0.10),
        "overnight_gap": int(count * 0.10),
        "risk_management": int(count * 0.07),
        "regime_transition": int(count * 0.07),
        "sector_analysis": int(count * 0.08),
        "opening_momentum": int(count * 0.06),
        "fx_impact": int(count * 0.06),
        "stop_loss_management": int(count * 0.06),
        "intraday_rebalancing": int(count * 0.06),
        "what_if": int(count * 0.06),
        "multi_step": int(count * 0.06),
    }

    r = real_data_ratio

    for _ in range(task_counts["full_analysis"]):
        regime = random.choice(REGIMES)
        snap = _maybe_snapshot(use_real_data, regime, r)
        examples.append(gen_full_analysis(regime, snap))
    for _ in range(task_counts["regime_classification"]):
        regime = random.choice(REGIMES)
        snap = _maybe_snapshot(use_real_data, regime, r)
        examples.append(gen_regime_classification(regime, snap))
    for _ in range(task_counts["overnight_gap"]):
        examples.append(gen_overnight_gap(random.choice(["bullish", "bearish"])))
    for _ in range(task_counts["risk_management"]):
        snap = _maybe_snapshot(use_real_data, ratio=r)
        examples.append(gen_risk_management(snap))
    for _ in range(task_counts["regime_transition"]):
        snap = _maybe_snapshot(use_real_data, ratio=r)
        examples.append(gen_regime_transition(snap))
    for _ in range(task_counts["sector_analysis"]):
        snap = _maybe_snapshot(use_real_data, ratio=r)
        examples.append(gen_sector_analysis(snap))
    for _ in range(task_counts["opening_momentum"]):
        snap = _maybe_snapshot(use_real_data, ratio=r)
        examples.append(gen_opening_momentum(snap))
    for _ in range(task_counts["fx_impact"]):
        examples.append(gen_fx_impact())
    for _ in range(task_counts["stop_loss_management"]):
        snap = _maybe_snapshot(use_real_data, ratio=r)
        examples.append(gen_stop_loss_management(snap))
    for _ in range(task_counts["intraday_rebalancing"]):
        snap = _maybe_snapshot(use_real_data, ratio=r)
        examples.append(gen_intraday_rebalancing(snap))
    for _ in range(task_counts["what_if"]):
        snap = _maybe_snapshot(use_real_data, ratio=r)
        examples.append(gen_what_if(snap))
    for _ in range(task_counts["multi_step"]):
        snap = _maybe_snapshot(use_real_data, ratio=r)
        examples.append(gen_multi_step(snap))

    random.shuffle(examples)
    return examples


def to_chatml(examples: list[dict]) -> list[dict]:
    """ChatML 형식으로 변환합니다."""
    return [
        {
            "text": (
                f"<|im_start|>system\n{ex['system'].strip()}<|im_end|>\n"
                f"<|im_start|>user\n{ex['user'].strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['assistant'].strip()}<|im_end|>"
            ),
            "source": "synthetic_" + ex["task_type"],
        }
        for ex in examples
    ]


def main():
    global _snapshot_store

    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "synthetic_data.json"))
    parser.add_argument("--use-real-data", action="store_true",
                        help="Use real market_daily data for realistic snapshots")
    parser.add_argument("--start-date", type=str, default="2020-01-01",
                        help="Start date for real data (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-12-31",
                        help="End date for real data (YYYY-MM-DD)")
    parser.add_argument("--real-data-ratio", type=float, default=0.7,
                        help="Fraction of examples using real data (0.0-1.0)")
    args = parser.parse_args()

    if args.use_real_data:
        from finetune.data.market_snapshot import MarketSnapshotStore
        print(f"Loading real market data ({args.start_date} ~ {args.end_date})...")
        _snapshot_store = MarketSnapshotStore(
            start_date=args.start_date,
            end_date=args.end_date,
            universe_tickers=list(KOREAN_STOCKS.keys()),
        )
        n_snapshots = _snapshot_store.load()
        regime_dist = _snapshot_store.get_regime_distribution()
        print(f"  Snapshots: {n_snapshots}")
        print(f"  Regime distribution: {regime_dist}")
        print(f"  Real data ratio: {args.real_data_ratio:.0%}")

    print(f"\nGenerating {args.count} synthetic examples...")
    print(f"Stock universe: {len(KOREAN_STOCKS)} stocks")

    examples = generate_all(
        args.count,
        use_real_data=args.use_real_data,
        real_data_ratio=args.real_data_ratio,
    )

    from collections import Counter
    task_counts = Counter(ex["task_type"] for ex in examples)

    # reason 다양성 체크
    all_reasons = []
    for ex in examples:
        assistant = json.loads(ex["assistant"])
        for pos in assistant.get("positions", {}).get("long", []):
            if "reason" in pos:
                all_reasons.append(pos["reason"])

    unique_reasons = len(set(all_reasons))
    total_reasons = len(all_reasons)

    print(f"\n=== Task Distribution ===")
    for task, cnt in task_counts.most_common():
        print(f"  {task:25s}: {cnt:>5}")
    print(f"  {'TOTAL':25s}: {len(examples):>5}")

    if total_reasons:
        print(f"\n=== Reason Diversity ===")
        print(f"  Total reasons: {total_reasons}")
        print(f"  Unique reasons: {unique_reasons}")
        print(f"  Uniqueness: {unique_reasons / total_reasons * 100:.1f}%")

    chatml = to_chatml(examples)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chatml, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {args.output}")
    print(f"File size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
