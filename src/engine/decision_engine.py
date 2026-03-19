"""
DecisionEngine — CLAUDE.md 섹션 6 기준.

순수 규칙 기반 진입/청산. AI 없음.
롱/숏 비대칭 임계값 + 비대칭 SL/TP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from config.settings import (
    COOLDOWN_MINUTES,
    ENTRY_CONFIRM_CYCLES,
    ENTRY_MIN_SCORE_INCREASE,
    ENTRY_REQUIRE_RISING,
    FADE_DURATION_MIN,
    FADE_THRESHOLD,
    LONG_ENTRY_THRESHOLD,
    LONG_SL_PCT,
    LONG_TP_PCT,
    MAX_SAME_SYMBOL_LOSSES,
    MAX_TRADES_PER_DAY,
    MIN_HOLD_MINUTES,
    SAME_SYMBOL_BAN_HOURS,
    SHORT_ENTRY_THRESHOLD,
    SHORT_SL_PCT,
    SHORT_TP_PCT,
    SWITCH_COOLDOWN_MINUTES,
    SWITCH_MIN_HOLD_MINUTES,
    SWITCH_REVERSE_SCORE_DROP,
    SWITCH_SAME_DIR_GAP,
    WEAK_DURATION_MIN,
    WEAK_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class Order:
    symbol: str
    direction: str   # "LONG" or "SHORT"
    sl_pct: float
    tp_pct: float
    action: str = "OPEN"  # "OPEN" or "CLOSE"
    reason: str = ""


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    sl_price: float
    tp_price: float
    trailing_active: bool = False
    peak_pnl: float = 0.0
    entry_score: float = 0.0
    fade_since: datetime | None = None  # score가 FADE 임계값 아래로 떨어진 시각
    weak_since: datetime | None = None  # score가 WEAK 임계값 아래로 떨어진 시각


class DecisionEngine:
    """
    규칙 기반 의사결정.

    진입: LONG >= +0.30, SHORT <= -0.50 (비대칭)
    청산: 시그널 반전 (30분 이상 보유), 시그널 소멸 (2시간 이상)
    SL/TP: 롱 -5%/+10%, 숏 -3%/+8% (비대칭)
    """

    def __init__(self):
        self._last_exit_time: datetime | None = None
        self._last_exit_symbol: str = ""
        self._last_switch_time: datetime | None = None  # SWITCH cooldown용
        self._pending_entry: Dict[str, list] = {}  # {symbol: [score1, score2, ...]} 연속 확인용
        self._symbol_losses: Dict[str, list] = {}  # {symbol: [loss_time1, ...]} 연속 손실 추적
        self._daily_trade_count: int = 0
        self._daily_trade_date: str = ""  # "YYYY-MM-DD"

    def decide(
        self,
        ensemble_scores: Dict[str, float],
        current_position: Position | None,
        market_data: Dict | None = None,
    ) -> Optional[Order]:
        """
        매매 결정.

        Args:
            ensemble_scores: {symbol: score}
            current_position: 현재 포지션
            market_data: {"btc_1h_ret": float, "coin_1h_rets": {sym: float}, "coin_vol_ratios": {sym: float}}

        Returns:
            Order(OPEN/CLOSE) or None
        """
        if market_data is None:
            market_data = {}

        if current_position is not None:
            return self._check_exit(current_position, ensemble_scores)

        return self._check_entry(ensemble_scores, market_data)

    def _check_entry(self, scores: Dict[str, float], market_data: Dict = None) -> Optional[Order]:
        """진입 판단 — 필터 적용."""
        if market_data is None:
            market_data = {}
        now = datetime.now(timezone.utc)

        btc_1h_ret = market_data.get("btc_1h_ret", 0)
        coin_1h_rets = market_data.get("coin_1h_rets", {})
        coin_vol_ratios = market_data.get("coin_vol_ratios", {})

        # 롱/숏 후보
        long_candidates = [
            (sym, sc) for sym, sc in scores.items()
            if sc > LONG_ENTRY_THRESHOLD
        ]
        short_candidates = [
            (sym, sc) for sym, sc in scores.items()
            if sc < SHORT_ENTRY_THRESHOLD
        ]

        all_candidates = (
            [(s, sc, "LONG") for s, sc in long_candidates]
            + [(s, sc, "SHORT") for s, sc in short_candidates]
        )

        # === 필터 적용 ===
        filtered = []
        for sym, sc, direction in all_candidates:
            # 필터 1: 과열 방지 — 이미 3% 이상 움직인 코인 제외
            coin_1h = coin_1h_rets.get(sym, 0)
            if direction == "LONG" and coin_1h > 0.03:
                logger.info(f"FILTER: {sym.split('/')[0]} LONG blocked — already +{coin_1h:.1%} in 1h")
                continue
            if direction == "SHORT" and coin_1h < -0.03:
                logger.info(f"FILTER: {sym.split('/')[0]} SHORT blocked — already {coin_1h:.1%} in 1h")
                continue

            # 필터 2: BTC 방향 확인 — BTC 역방향 진입 차단
            if direction == "LONG" and btc_1h_ret < -0.01:
                logger.info(f"FILTER: {sym.split('/')[0]} LONG blocked — BTC dropping {btc_1h_ret:.1%}")
                continue
            if direction == "SHORT" and btc_1h_ret > 0.01:
                logger.info(f"FILTER: {sym.split('/')[0]} SHORT blocked — BTC rising {btc_1h_ret:.1%}")
                continue

            # 필터 3: 거래량 확인 — 거래량 없는 움직임 제외
            vol_ratio = coin_vol_ratios.get(sym, 1.0)
            if vol_ratio < 0.7:
                logger.info(f"FILTER: {sym.split('/')[0]} blocked — low volume ({vol_ratio:.1f}x)")
                continue

            filtered.append((sym, sc, direction))

        all_candidates = filtered

        if not all_candidates:
            return None

        # 절대값 기준 가장 강한 시그널
        best = max(all_candidates, key=lambda x: abs(x[1]))
        symbol, score, direction = best

        # 쿨다운: 같은 코인 30분
        if (
            self._last_exit_time
            and self._last_exit_symbol == symbol
            and (now - self._last_exit_time) < timedelta(minutes=COOLDOWN_MINUTES)
        ):
            logger.info(f"Cooldown active for {symbol}, skipping")
            remaining = [c for c in all_candidates if c[0] != symbol]
            if remaining:
                best = max(remaining, key=lambda x: abs(x[1]))
                symbol, score, direction = best
            else:
                return None

        # 연속 손실 차단: 같은 코인 N번 연속 손실시 일정 시간 차단
        ban_times = self._symbol_losses.get(symbol, [])
        if len(ban_times) >= MAX_SAME_SYMBOL_LOSSES:
            latest_loss = ban_times[-1]
            if (now - latest_loss) < timedelta(hours=SAME_SYMBOL_BAN_HOURS):
                logger.info(f"BANNED: {symbol.split('/')[0]} — {len(ban_times)} consecutive losses, {SAME_SYMBOL_BAN_HOURS}h ban")
                remaining = [c for c in all_candidates if c[0] != symbol]
                if remaining:
                    best = max(remaining, key=lambda x: abs(x[1]))
                    symbol, score, direction = best
                else:
                    return None

        # N사이클 연속 확인: 이전 N-1사이클에서도 같은 코인이 임계값을 넘었어야 진입
        threshold = LONG_ENTRY_THRESHOLD if direction == "LONG" else abs(SHORT_ENTRY_THRESHOLD)
        prev_scores = self._pending_entry.get(symbol, [])

        confirmed = False
        reject_reason = ""
        if len(prev_scores) >= (ENTRY_CONFIRM_CYCLES - 1):
            all_scores = prev_scores + [score]  # [cycle1, cycle2, cycle3(현재)]

            if ENTRY_REQUIRE_RISING:
                # 상승 추세 확인: 마지막 - 첫번째 >= MIN_SCORE_INCREASE
                score_delta = abs(all_scores[-1]) - abs(all_scores[0])
                is_rising = score_delta >= ENTRY_MIN_SCORE_INCREASE
                # 마지막 스코어가 임계값 이상
                last_above_threshold = abs(all_scores[-1]) >= threshold
                confirmed = is_rising and last_above_threshold

                if not confirmed:
                    trajectory = "→".join(f"{s:+.3f}" for s in all_scores)
                    if not is_rising:
                        reject_reason = f"NOT_RISING delta={score_delta:+.3f}<{ENTRY_MIN_SCORE_INCREASE} [{trajectory}]"
                    else:
                        reject_reason = f"BELOW_THRESH |{score:+.3f}|<{threshold:.2f} [{trajectory}]"
            else:
                # 기존 방식 (폴백)
                confirmed = all(abs(s) >= threshold for s in prev_scores)

        if not confirmed:
            prev_scores.append(score)
            # 최근 N-1개만 유지
            self._pending_entry = {symbol: prev_scores[-(ENTRY_CONFIRM_CYCLES - 1):]}
            trajectory = "→".join(f"{s:+.3f}" for s in prev_scores)
            if reject_reason:
                # 3사이클 모였지만 조건 미달로 차단
                logger.info(
                    f"REJECTED: {symbol.split('/')[0]} {direction} — {reject_reason}"
                )
            else:
                # 아직 사이클 수집 중
                logger.info(
                    f"PENDING: {symbol.split('/')[0]} {direction} score={score:+.3f} "
                    f"— {len(prev_scores)}/{ENTRY_CONFIRM_CYCLES} cycles [{trajectory}]"
                )
            return None

        # N사이클 연속 확인 완료 → 진입!
        entry_trajectory = prev_scores + [score]
        entry_delta = abs(entry_trajectory[-1]) - abs(entry_trajectory[0])
        self._pending_entry = {}

        # 일일 거래 횟수 제한
        today_str = now.strftime("%Y-%m-%d")
        if self._daily_trade_date != today_str:
            self._daily_trade_date = today_str
            self._daily_trade_count = 0
        if self._daily_trade_count >= MAX_TRADES_PER_DAY:
            logger.info(f"DAILY LIMIT: {self._daily_trade_count}/{MAX_TRADES_PER_DAY} trades today, blocking new entry")
            return None
        self._daily_trade_count += 1

        # SL/TP 비대칭
        if direction == "LONG":
            sl_pct, tp_pct = LONG_SL_PCT, LONG_TP_PCT
        else:
            sl_pct, tp_pct = SHORT_SL_PCT, SHORT_TP_PCT

        trajectory_str = "→".join(f"{s:+.3f}" for s in entry_trajectory)
        logger.info(
            f"ENTRY: {symbol.split('/')[0]} {direction} score={score:+.3f} "
            f"delta={entry_delta:+.3f} [{trajectory_str}] "
            f"SL={sl_pct:+.1%} TP={tp_pct:+.1%}"
        )

        return Order(
            symbol=symbol,
            direction=direction,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            action="OPEN",
            reason=f"score={score:+.3f}",
        )

    def _check_exit(
        self, pos: Position, scores: Dict[str, float]
    ) -> Optional[Order]:
        """
        점진적 청산 로직 (임계값은 settings.py 참조).

        우선순위:
          1. 최소 보유 시간 미달 → 유지
          2. 시그널 반전 (REVERSAL) → 즉시 청산
          3. WEAK: effective_score < WEAK_THRESHOLD → WEAK_DURATION_MIN 지속시 청산
          4. FADE: effective_score < FADE_THRESHOLD → FADE_DURATION_MIN 지속시 청산
          5. SWITCH: 더 강한 코인 → 엄격 조건 하 교체
        """
        now = datetime.now(timezone.utc)
        entry_time = pos.entry_time if pos.entry_time.tzinfo else pos.entry_time.replace(tzinfo=timezone.utc)
        hold_duration = now - entry_time
        hold_min = hold_duration.total_seconds() / 60
        short_sym = pos.symbol.split('/')[0]

        current_score = scores.get(pos.symbol, 0)
        # 방향 보정: 롱이면 양수가 유리, 숏이면 음수가 유리 → 절대값으로 통일
        effective_score = current_score if pos.direction == "LONG" else -current_score

        # --- 1. 최소 보유 시간 ---
        if hold_duration < timedelta(minutes=MIN_HOLD_MINUTES):
            return None

        # --- 2. 시그널 반전 (REVERSAL) → 즉시 청산 ---
        if pos.direction == "LONG" and current_score < SHORT_ENTRY_THRESHOLD:
            return self._make_close_order(pos, "REVERSAL", current_score)
        if pos.direction == "SHORT" and current_score > LONG_ENTRY_THRESHOLD:
            return self._make_close_order(pos, "REVERSAL", current_score)

        # --- 3. WEAK: |score| < WEAK_THRESHOLD → WEAK_DURATION_MIN 지속시 청산 ---
        is_weak = effective_score < WEAK_THRESHOLD
        if is_weak:
            if pos.weak_since is None:
                pos.weak_since = now
                logger.info(
                    f"EXIT_WEAK_START: {short_sym} {pos.direction} "
                    f"score={current_score:+.3f} < {WEAK_THRESHOLD} — timer started"
                )
            weak_min = (now - pos.weak_since).total_seconds() / 60
            if weak_min >= WEAK_DURATION_MIN:
                return self._make_close_order(pos, "WEAK", current_score)
        else:
            if pos.weak_since is not None:
                logger.info(
                    f"EXIT_WEAK_RESET: {short_sym} score={current_score:+.3f} "
                    f"recovered above {WEAK_THRESHOLD}"
                )
            pos.weak_since = None

        # --- 4. FADE: |score| < FADE_THRESHOLD → FADE_DURATION_MIN 지속시 청산 ---
        is_fading = effective_score < FADE_THRESHOLD
        if is_fading:
            if pos.fade_since is None:
                pos.fade_since = now
                logger.info(
                    f"EXIT_FADE_START: {short_sym} {pos.direction} "
                    f"score={current_score:+.3f} < {FADE_THRESHOLD} — timer started"
                )
            fade_min = (now - pos.fade_since).total_seconds() / 60
            if fade_min >= FADE_DURATION_MIN:
                return self._make_close_order(pos, "FADE", current_score)
        else:
            if pos.fade_since is not None:
                logger.info(
                    f"EXIT_FADE_RESET: {short_sym} score={current_score:+.3f} "
                    f"recovered above {FADE_THRESHOLD}"
                )
            pos.fade_since = None

        # --- 5. 상태 로그 (튜닝용) ---
        fade_info = ""
        if pos.fade_since:
            fade_elapsed = (now - pos.fade_since).total_seconds() / 60
            fade_info = f"fade={fade_elapsed:.0f}m/{FADE_DURATION_MIN}"
        weak_info = ""
        if pos.weak_since:
            weak_elapsed = (now - pos.weak_since).total_seconds() / 60
            weak_info = f"weak={weak_elapsed:.0f}m/{WEAK_DURATION_MIN}"
        if fade_info or weak_info:
            timers = " ".join(filter(None, [fade_info, weak_info]))
            logger.info(
                f"EXIT_CHECK: {short_sym} {pos.direction} score={current_score:+.3f} "
                f"hold={hold_min:.0f}min | {timers}"
            )

        # --- 6. SWITCH: 더 강한 코인 교체 (엄격 조건) ---
        if not scores:
            return None

        if hold_duration < timedelta(minutes=SWITCH_MIN_HOLD_MINUTES):
            return None

        if self._last_switch_time and (now - self._last_switch_time).total_seconds() / 60 < SWITCH_COOLDOWN_MINUTES:
            return None

        # 스코어가 아직 FADE 수준 이상이면 SWITCH 차단 (포지션 보호)
        if effective_score >= FADE_THRESHOLD:
            return None

        best_sym, best_score = max(scores.items(), key=lambda x: abs(x[1]))
        if best_sym != pos.symbol:
            diff = abs(best_score) - abs(current_score)
            same_dir = (pos.direction == "LONG" and best_score > 0) or (pos.direction == "SHORT" and best_score < 0)
            logger.info(
                f"SWITCH check: {short_sym}({current_score:+.3f}) "
                f"vs {best_sym.split('/')[0]}({best_score:+.3f}) "
                f"diff={diff:.3f} same_dir={same_dir}"
            )

            best_is_entry = best_score > LONG_ENTRY_THRESHOLD or best_score < SHORT_ENTRY_THRESHOLD
            if best_is_entry:
                same_direction = (
                    (pos.direction == "LONG" and best_score > 0)
                    or (pos.direction == "SHORT" and best_score < 0)
                )
                if same_direction and diff >= SWITCH_SAME_DIR_GAP:
                    return self._make_close_order(pos, "SWITCH", current_score)

                entry_thresh = LONG_ENTRY_THRESHOLD if pos.direction == "LONG" else abs(SHORT_ENTRY_THRESHOLD)
                if not same_direction and abs(current_score) < (entry_thresh - SWITCH_REVERSE_SCORE_DROP):
                    return self._make_close_order(pos, "SWITCH", current_score)

        return None

    def _make_close_order(
        self, pos: Position, reason: str, score: float
    ) -> Order:
        logger.info(
            f"EXIT: {pos.symbol} {pos.direction} reason={reason} "
            f"score={score:+.3f}"
        )
        return Order(
            symbol=pos.symbol,
            direction=pos.direction,
            sl_pct=0,
            tp_pct=0,
            action="CLOSE",
            reason=reason,
        )

    def record_exit(self, symbol: str, reason: str = "", pnl_pct: float = 0.0) -> None:
        """청산 기록 (쿨다운용)."""
        now = datetime.now(timezone.utc)

        self._last_exit_time = now
        self._last_exit_symbol = symbol
        if reason == "SWITCH":
            self._last_switch_time = now  # SWITCH cooldown 기록

        # 연속 손실 추적
        if pnl_pct < 0:
            losses = self._symbol_losses.get(symbol, [])
            losses.append(now)
            # 최근 N개만 유지
            self._symbol_losses[symbol] = losses[-MAX_SAME_SYMBOL_LOSSES:]
        else:
            # 수익이면 연속 손실 리셋
            self._symbol_losses.pop(symbol, None)
