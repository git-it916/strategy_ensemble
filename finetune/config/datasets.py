"""HuggingFace 데이터셋 설정."""

DATASET_CONFIG = {
    "finance_instruct": {
        "hf_id": "Josephgflowers/Finance-Instruct-500k",
        "ratio": 0.40,
        "description": "금융 전반 지식 (정책, 투자, 경제이론)",
    },
    "sujet_finance": {
        "hf_id": "sujet-ai/Sujet-Finance-Instruct-177k",
        "ratio": 0.25,
        "description": "금융 NLP 18개 태스크 통합",
    },
    "krx_won": {
        "hf_id": "KRX-Data/Won-Instruct",
        "ratio": 0.20,
        "description": "KRX 한국어 금융 (핵심)",
    },
    "trading_signals": {
        "hf_id": "darkknight25/trading_dataset_v2",
        "ratio": 0.10,
        "description": "기술적 지표 → 매매 시그널",
    },
    "fin_sentiment": {
        "hf_id": "FinGPT/fingpt-sentiment-train",
        "ratio": 0.05,
        "description": "금융 뉴스 감성 분석",
    },
}
