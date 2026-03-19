"""
StackingMetaModel — CLAUDE.md 섹션 5-2 기준.

Ridge Regression (alpha=1.0)으로 알파 시그널 → forward return 예측.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config.settings import (
    MODELS_DIR,
    STACKING_MIN_HOURS,
    STACKING_RETRAIN_INTERVAL_DAYS,
    STACKING_TRAIN_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)

STACKING_PATH = MODELS_DIR / "stacking" / "stacking_latest.pkl"


class StackingMetaModel:
    """
    Ridge 메타모델.

    피처: [10개 alpha score, 10개 alpha confidence, vol_regime_confidence]
    타겟: 5일 후 수익률
    학습: 90일 윈도우, 30일마다 재학습
    """

    def __init__(self):
        self._ridge: Ridge | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._last_train_date: datetime | None = None
        self._buffer: List[Dict[str, Any]] = []

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def record(self, symbol: str, features: Dict[str, float], timestamp: datetime) -> None:
        """시그널 버퍼에 기록 (재학습용)."""
        self._buffer.append({
            "symbol": symbol,
            "timestamp": timestamp,
            **features,
        })
        # 메모리 관리
        max_rows = STACKING_TRAIN_WINDOW_DAYS * 20  # 90일 × ~20심볼
        if len(self._buffer) > max_rows * 2:
            self._buffer = self._buffer[-max_rows:]

    def should_retrain(self) -> bool:
        # 첫 학습: 16시간치 데이터 (192 사이클 × 10코인 = ~1920행)
        min_samples = int(STACKING_MIN_HOURS * 12 * 10)  # 시간 × (60/5분) × 10코인
        if self._last_train_date is None:
            return len(self._buffer) >= min_samples
        elapsed = (datetime.utcnow() - self._last_train_date).days
        return elapsed >= STACKING_RETRAIN_INTERVAL_DAYS

    def fit(
        self,
        prices_1d: Dict[str, pd.DataFrame],
        prices_5m: Dict[str, pd.DataFrame] | None = None,
    ) -> Dict[str, Any]:
        """
        다중 horizon 학습.

        1차 (16시간 후): 5분봉으로 1시간 후 수익률 학습
        2차 (5일 후): 일봉으로 5일 후 수익률 학습 (더 정확)
        이후: 일봉 5일 기준 재학습

        둘 다 시도해서 더 많은 타겟이 나오는 쪽으로 학습.
        """
        min_samples = int(STACKING_MIN_HOURS * 12 * 10)
        if len(self._buffer) < min_samples:
            return {"status": "insufficient_data", "n": len(self._buffer)}

        df = pd.DataFrame(self._buffer)

        # --- 5일 후 수익률 (일봉 기준, 정확도 높음) ---
        rows_5d = self._calc_forward_returns(df, prices_1d, lookahead=5)

        # --- 1시간 후 수익률 (5분봉 기준, 빠른 학습) ---
        rows_1h = []
        if prices_5m:
            rows_1h = self._calc_forward_returns(df, prices_5m, lookahead=12)

        # 더 많은 타겟이 있는 쪽 선택
        if len(rows_5d) >= 100:
            rows = rows_5d
            horizon_label = "5d"
        elif len(rows_1h) >= 100:
            rows = rows_1h
            horizon_label = "1h"
        else:
            return {
                "status": "insufficient_targets",
                "n_5d": len(rows_5d),
                "n_1h": len(rows_1h),
            }

        train_df = pd.DataFrame(rows)
        feature_cols = [c for c in train_df.columns if c.endswith("_score") or c.endswith("_conf")]
        if not feature_cols:
            return {"status": "no_features"}

        self._feature_names = feature_cols
        X = train_df[feature_cols].fillna(0).values
        y = train_df["fwd_return"].values

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._ridge = Ridge(alpha=1.0)
        self._ridge.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = datetime.utcnow()

        from scipy.stats import spearmanr
        y_pred = self._ridge.predict(X_scaled)
        ic, _ = spearmanr(y_pred, y)

        logger.info(
            f"Stacking fit ({horizon_label}): {len(train_df)} samples, IC={ic:.4f}"
        )
        self.save()

        return {
            "status": "fitted",
            "horizon": horizon_label,
            "n_samples": len(train_df),
            "ic": float(ic),
        }

    @staticmethod
    def _calc_forward_returns(
        df: pd.DataFrame,
        prices: Dict[str, pd.DataFrame],
        lookahead: int,
    ) -> List[Dict]:
        """버퍼 데이터에 forward return을 붙여서 반환."""
        rows = []
        for _, row in df.iterrows():
            sym = row.get("symbol", "")
            ts = row.get("timestamp")
            if sym not in prices or ts is None:
                continue
            pdf = prices[sym]
            if pdf.empty:
                continue
            past = pdf[pdf["timestamp"] <= ts]
            future = pdf[pdf["timestamp"] > ts]
            if past.empty or len(future) < lookahead:
                continue
            close_now = float(past["close"].iloc[-1])
            close_fwd = float(future.iloc[lookahead - 1]["close"])
            if close_now > 0:
                fwd_ret = (close_fwd / close_now) - 1
                row_dict = row.to_dict()
                row_dict["fwd_return"] = np.clip(fwd_ret, -0.3, 0.3)
                rows.append(row_dict)
        return rows

    def predict_one(self, features: Dict[str, float]) -> float:
        """단일 심볼 예측."""
        if not self._is_fitted or self._ridge is None:
            return 0.0

        row = [features.get(f, 0.0) for f in self._feature_names]
        X = np.array([row])
        X_scaled = self._scaler.transform(X)
        pred = float(self._ridge.predict(X_scaled)[0])
        return float(np.tanh(pred * 20))

    def save(self) -> None:
        STACKING_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "ridge": self._ridge,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
            "last_train_date": self._last_train_date,
            "buffer": self._buffer[-3000:],
        }
        with open(STACKING_PATH, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls) -> "StackingMetaModel":
        obj = cls()
        if STACKING_PATH.exists():
            with open(STACKING_PATH, "rb") as f:
                state = pickle.load(f)
            obj._ridge = state.get("ridge")
            obj._scaler = state.get("scaler")
            obj._feature_names = state.get("feature_names", [])
            obj._is_fitted = state.get("is_fitted", False)
            obj._last_train_date = state.get("last_train_date")
            obj._buffer = state.get("buffer", [])
            if obj._is_fitted:
                logger.info("Stacking model loaded from disk")
        return obj
