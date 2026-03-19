#!/usr/bin/env python3
"""
Strategy Ensemble Daemon — CLAUDE.md 섹션 11 기준.

asyncio 기반 메인 루프:
    - 5분마다 리밸런스 (시그널 → 앙상블 → 의사결정 → 실행)
    - 5초마다 SL/TP/트레일링 체크
    - 60초마다 오더북 수집
    - 4시간마다 일봉+펀딩 갱신
    - 1시간마다 유니버스 갱신
    - 30일마다 Stacking 재학습

Usage:
    python scripts/run_daemon.py
    python scripts/run_daemon.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    BALANCE_USAGE_RATIO,
    COOLDOWN_MINUTES,
    DAILY_REFRESH_INTERVAL_SEC,
    LEVERAGE,
    LOGS_DIR,
    ORDERBOOK_INTERVAL_SEC,
    REBALANCE_INTERVAL_SEC,
    UNIVERSE_REFRESH_INTERVAL_SEC,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daemon")


def load_keys() -> dict:
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    with open(keys_path) as f:
        return yaml.safe_load(f)


async def main_loop(dry_run: bool = False):
    keys = load_keys()
    binance_cfg = keys.get("binance", {})
    tg_cfg = keys.get("telegram", {})

    # --- 초기화 ---
    from src.data.data_manager import DataManager
    from src.alphas.v2 import ALL_ALPHAS
    from src.alphas.base_alpha_v2 import AlphaSignal
    from src.engine.signal_aggregator import SignalAggregator
    from src.engine.decision_engine import DecisionEngine
    from src.engine.sltp_monitor import SLTPMonitor
    from src.engine.daily_drawdown import DailyDrawdownGuard
    from src.engine.position_store import PositionStore
    from src.execution.binance_executor import BinanceExecutor
    from src.ensemble.stacking import StackingMetaModel
    from src.notifications.telegram_notifier import TelegramNotifier

    data_mgr = DataManager(
        api_key=binance_cfg.get("api_key", ""),
        api_secret=binance_cfg.get("api_secret", ""),
    )
    executor = BinanceExecutor(
        api_key=binance_cfg.get("api_key", ""),
        api_secret=binance_cfg.get("api_secret", ""),
    )
    alphas = [cls() for cls in ALL_ALPHAS]
    aggregator = SignalAggregator()
    engine = DecisionEngine()
    monitor = SLTPMonitor()
    drawdown = DailyDrawdownGuard()
    store = PositionStore()

    chat_id = str(tg_cfg.get("chat_id", ""))
    if not chat_id:
        bcast = tg_cfg.get("broadcast_chat_ids", [])
        chat_id = str(bcast[0]) if bcast else ""
    telegram = TelegramNotifier(
        bot_token=tg_cfg.get("bot_token", ""),
        chat_id=chat_id,
        enabled=bool(tg_cfg.get("bot_token")),
    )

    # Stacking 비활성화 — 가중합 전용
    # stacking = StackingMetaModel.load()
    # if stacking.is_fitted:
    #     aggregator.set_stacking(stacking)

    # --- 시계 동기화 (WSL 필수) ---
    if not dry_run:
        from src.execution.time_sync import sync_time_offset
        sync_time_offset(data_mgr._exchange)
        sync_time_offset(executor._exchange)

    # --- 초기 데이터 수집 ---
    logger.info("=" * 60)
    logger.info(f"Strategy Ensemble Daemon {'[DRY RUN]' if dry_run else '[LIVE]'}")
    logger.info("=" * 60)

    await data_mgr.initial_fetch()
    logger.info(f"Universe: {data_mgr.universe}")
    logger.info(f"Alphas: {[a.name for a in alphas]}")

    if not dry_run:
        for symbol in data_mgr.universe:
            await executor.set_leverage(symbol)

    balance = await executor.get_balance() if not dry_run else 1000.0
    drawdown.set_balance(balance)
    logger.info(f"Balance: ${balance:.2f}")

    # 바이낸스 실제 포지션과 position_store 동기화
    if not dry_run:
        await _sync_positions(executor, store)
    elif store.current:
        executor.current_position = store.current
        logger.info(f"Position restored (dry-run): {store.current.symbol}")

    # --- SL/TP 모니터 (별도 태스크) ---
    if not dry_run:
        async def sltp_task():
            await monitor.run_loop(
                executor, telegram,
                lambda: store.current if not monitor._closing else None,
                lambda sym: executor.get_price(sym),
                store=store,
                engine=engine,
                drawdown=drawdown,
            )
        asyncio.create_task(sltp_task())

    # --- 오더북 수집 (별도 태스크) ---
    async def orderbook_task():
        while True:
            try:
                await data_mgr.refresh_orderbooks()
            except Exception as e:
                logger.debug(f"Orderbook task error: {e}")
            await asyncio.sleep(ORDERBOOK_INTERVAL_SEC)
    asyncio.create_task(orderbook_task())

    # --- WSL 시계 동기화 (30초마다, 백그라운드) ---
    if not dry_run:
        async def time_sync_task():
            while True:
                try:
                    from src.execution.time_sync import sync_time_offset
                    sync_time_offset(data_mgr._exchange)
                    sync_time_offset(executor._exchange)
                except Exception:
                    pass
                await asyncio.sleep(30)
        asyncio.create_task(time_sync_task())

    # --- 타이머 ---
    last_daily = time.time()
    last_universe = time.time()

    # --- 메인 루프 ---
    while True:
        try:
            cycle_start = time.time()
            now = datetime.now(timezone.utc)
            logger.info(f"{'='*30} REBALANCE {'='*30}")

            # 1. 인트라데이 데이터 갱신
            await data_mgr.refresh_intraday()

            # 2. 4시간마다 일봉+펀딩
            if time.time() - last_daily > DAILY_REFRESH_INTERVAL_SEC:
                await data_mgr.refresh_daily()
                last_daily = time.time()

            # 3. 1시간마다 유니버스
            if time.time() - last_universe > UNIVERSE_REFRESH_INTERVAL_SEC:
                await data_mgr.refresh_universe()
                last_universe = time.time()

            # 4. DataBundle 생성
            bundle = data_mgr.to_bundle()

            # 5. 10개 알파 compute
            signals = {}
            for symbol in bundle.universe:
                signals[symbol] = {}
                for alpha in alphas:
                    try:
                        sig = await alpha.compute(symbol, bundle)
                        signals[symbol][alpha.name] = sig
                    except Exception as e:
                        logger.error(f"Alpha {alpha.name} failed {symbol}: {e}")
                        signals[symbol][alpha.name] = AlphaSignal()

            # 6. 앙상블
            ensemble_scores = aggregator.aggregate(signals)

            top3 = sorted(ensemble_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            logger.info(f"Top: {[(s.split('/')[0], f'{sc:+.3f}') for s, sc in top3]}")

            # 7. 일일 손실 한도
            if not dry_run:
                balance = await executor.get_balance()
            drawdown.set_balance(balance)
            if drawdown.is_blocked():
                logger.warning("DAILY LOSS LIMIT — blocked")
                await asyncio.sleep(REBALANCE_INTERVAL_SEC)
                continue

            # 7.5. 바이낸스 실제 포지션과 동기화
            if not dry_run:
                await _sync_positions(executor, store)

            # 8. 시장 데이터 구성 + 의사결정
            market_data = {}
            btc_sym = next((s for s in bundle.universe if "BTC" in s), None)
            if btc_sym and btc_sym in bundle.ohlcv_5m and not bundle.ohlcv_5m[btc_sym].empty:
                btc_5m = bundle.ohlcv_5m[btc_sym]["close"].values
                if len(btc_5m) >= 12:
                    market_data["btc_1h_ret"] = float(btc_5m[-1] / btc_5m[-12] - 1)

            coin_1h_rets = {}
            coin_vol_ratios = {}
            for sym in bundle.universe:
                if sym in bundle.ohlcv_5m and not bundle.ohlcv_5m[sym].empty:
                    df5 = bundle.ohlcv_5m[sym]
                    closes = df5["close"].values
                    volumes = df5["volume"].values
                    if len(closes) >= 12:
                        coin_1h_rets[sym] = float(closes[-1] / closes[-12] - 1)
                    if len(volumes) >= 12:
                        recent_vol = float(volumes[-6:].mean())
                        baseline_vol = float(volumes.mean())
                        coin_vol_ratios[sym] = recent_vol / baseline_vol if baseline_vol > 0 else 1.0

            market_data["coin_1h_rets"] = coin_1h_rets
            market_data["coin_vol_ratios"] = coin_vol_ratios

            # SLTP 모니터가 청산 중이면 이번 사이클 스킵
            if monitor._closing:
                logger.info("SLTP monitor closing position — skipping decision")
                await asyncio.sleep(REBALANCE_INTERVAL_SEC)
                continue

            order = engine.decide(ensemble_scores, store.current, market_data)

            # 시그널 로그 (의사결정 후 — order 정보 포함)
            _log_signals(now, ensemble_scores, signals, bundle, store, drawdown,
                         aggregator=aggregator, engine=engine, order=order)

            # 9. 실행
            if order and order.action == "OPEN":
                logger.info(f"OPEN {order.symbol} {order.direction} ({order.reason})")
                if dry_run:
                    logger.info("[DRY RUN] skipped")
                else:
                    pos = await executor.open_position(order, balance)
                    if pos:
                        pos.entry_score = ensemble_scores.get(order.symbol, 0)
                        store.open(pos)
                        contribs = {
                            a.name: signals.get(order.symbol, {}).get(a.name, AlphaSignal()).score
                            for a in alphas
                        }
                        telegram.send_entry(
                            coin=order.symbol,
                            direction=order.direction,
                            entry_price=pos.entry_price,
                            stop_loss_price=pos.sl_price,
                            take_profit_price=pos.tp_price,
                            alpha_contributions=contribs,
                            ensemble_score=ensemble_scores.get(order.symbol, 0),
                        )

            elif order and order.action == "CLOSE" and store.current:
                logger.info(f"CLOSE {order.symbol} ({order.reason})")
                if dry_run:
                    logger.info("[DRY RUN] skipped")
                else:
                    # 청산 전에 포지션 정보 저장 (sync 후 store.current가 None이 되므로)
                    closing_symbol = store.current.symbol
                    closing_direction = store.current.direction
                    closing_entry_price = store.current.entry_price
                    closing_entry_time = store.current.entry_time
                    closing_entry_score = store.current.entry_score

                    # 청산 실행
                    close_success = False
                    pnl_pct = 0.0
                    try:
                        pnl_pct = await executor.close_position(store.current, order.reason)
                        await _sync_positions(executor, store)
                        if store.current is None:
                            close_success = True
                        else:
                            logger.error(f"Close failed: position still exists on Binance")
                    except Exception as e:
                        logger.error(f"Close execution error: {e}")

                    if close_success:
                        exit_price = await executor.get_price(closing_symbol)
                        fresh_balance = await executor.get_balance()
                        # pnl_usdt = balance × usage × leverage × pnl_pct
                        position_size = balance * BALANCE_USAGE_RATIO * LEVERAGE
                        pnl_usdt = position_size * pnl_pct
                        drawdown.record_pnl(pnl_usdt)
                        engine.record_exit(closing_symbol, order.reason, pnl_pct=pnl_pct)

                        # hold_min: UTC 통일
                        now_utc = datetime.now(timezone.utc)
                        if closing_entry_time.tzinfo:
                            hold_min = (now_utc - closing_entry_time).total_seconds() / 60
                        else:
                            hold_min = (now_utc.replace(tzinfo=None) - closing_entry_time).total_seconds() / 60

                        telegram.send_exit(
                            coin=closing_symbol,
                            direction=closing_direction,
                            entry_price=closing_entry_price,
                            exit_price=exit_price,
                            reason=order.reason,
                            pnl_usdt=pnl_usdt,
                            pnl_pct=pnl_pct * 100,
                            hold_duration_min=hold_min,
                        )

                        # store.close() 호출 — 거래 기록 저장
                        # (store.current은 이미 _sync에서 None이지만 trade log는 여기서 기록)
                        from src.engine.position_store import PositionStore
                        # 진입/청산 시점 컨텍스트 포함 (학습용)
                        exit_score = ensemble_scores.get(closing_symbol, 0)
                        alpha_at_exit = {}
                        for a in alphas:
                            sig = signals.get(closing_symbol, {}).get(a.name)
                            if sig:
                                alpha_at_exit[a.name] = round(sig.score, 4)
                        PositionStore._append_trade_log({
                            "symbol": closing_symbol,
                            "direction": closing_direction,
                            "entry_price": closing_entry_price,
                            "exit_price": exit_price,
                            "entry_time": closing_entry_time.isoformat() if hasattr(closing_entry_time, 'isoformat') else str(closing_entry_time),
                            "exit_time": datetime.now(timezone.utc).isoformat(),
                            "reason": order.reason,
                            "pnl_usdt": round(pnl_usdt, 4),
                            "pnl_pct": round(pnl_pct * 100, 2),
                            "hold_min": round(hold_min, 1),
                            "entry_score": round(closing_entry_score, 4),
                            "exit_score": round(exit_score, 4),
                            "exit_alphas": alpha_at_exit,
                            "balance": round(fresh_balance, 2),
                        })

                        # SWITCH: 청산 성공 확인 후에만 새 코인 진입
                        if order.reason == "SWITCH":
                            new_order = engine.decide(ensemble_scores, None, market_data)
                            if new_order and new_order.action == "OPEN":
                                logger.info(f"SWITCH → OPEN {new_order.symbol} {new_order.direction} ({new_order.reason})")
                                new_balance = await executor.get_balance()
                                if new_balance > 10:
                                    new_pos = await executor.open_position(new_order, new_balance)
                                    if new_pos:
                                        new_pos.entry_score = ensemble_scores.get(new_order.symbol, 0)
                                        store.open(new_pos)
                                        new_contribs = {
                                            a.name: signals.get(new_order.symbol, {}).get(a.name, AlphaSignal()).score
                                            for a in alphas
                                        }
                                        telegram.send_entry(
                                            coin=new_order.symbol,
                                            direction=new_order.direction,
                                            entry_price=new_pos.entry_price,
                                            stop_loss_price=new_pos.sl_price,
                                            take_profit_price=new_pos.tp_price,
                                            alpha_contributions=new_contribs,
                                            ensemble_score=ensemble_scores.get(new_order.symbol, 0),
                                        )
                                else:
                                    logger.warning(f"SWITCH aborted: insufficient balance ${new_balance:.2f}")
                    else:
                        logger.error(f"CLOSE failed — keeping position, will retry next cycle")
            else:
                status = "HOLDING" if store.current else "WATCHING"
                logger.info(f"No action ({status})")

            # 10. Stacking 비활성화
            # if stacking.should_retrain():
            #     result = stacking.fit(data_mgr._ohlcv_1d, data_mgr._ohlcv_5m)
            #     if result.get("status") == "fitted":
            #         aggregator.set_stacking(stacking)

            # 시장 레짐 피처 추출
            market_feats = {}
            btc_sym = next((s for s in bundle.universe if "BTC" in s), None)
            if btc_sym and btc_sym in bundle.ohlcv_1d and not bundle.ohlcv_1d[btc_sym].empty:
                btc_closes = bundle.ohlcv_1d[btc_sym]["close"].values
                if len(btc_closes) >= 8:
                    market_feats["btc_ret_7d"] = float(btc_closes[-1] / btc_closes[-8] - 1)
                if len(btc_closes) >= 21:
                    import numpy as np
                    dr = np.diff(btc_closes[-21:]) / btc_closes[-21:-1]
                    market_feats["btc_vol_20d"] = float(np.std(dr) * np.sqrt(365))
            # VolatilityRegime confidence
            for sym_sigs in signals.values():
                vs = sym_sigs.get("VolatilityRegime")
                if vs:
                    market_feats["vol_regime"] = vs.confidence
                    break

            # 시그널 버퍼 기록 (로그 데이터용, stacking 비활성)
            # for symbol, asigs in signals.items():
            #     feats = {}
            #     for aname, sig in asigs.items():
            #         feats[f"{aname}_score"] = sig.score
            #         feats[f"{aname}_conf"] = sig.confidence
            #     feats.update(market_feats)
            #     stacking.record(symbol, feats, now)

            logger.info(f"Cycle: {time.time() - cycle_start:.1f}s")

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)

        await asyncio.sleep(REBALANCE_INTERVAL_SEC)

    await data_mgr.close()
    await executor.close()


async def _sync_positions(executor, store):
    """바이낸스 실제 포지션과 position_store 동기화."""
    try:
        positions = await executor._exchange.fetch_positions()
        # 실제 열린 포지션 찾기
        real_pos = None
        for p in positions:
            contracts = abs(float(p.get("contracts", 0)))
            if contracts > 0:
                real_pos = p
                break

        if real_pos and store.current:
            # 둘 다 있음 — 심볼 일치 확인
            if real_pos["symbol"] != store.current.symbol:
                logger.warning(
                    f"Position mismatch: binance={real_pos['symbol']} "
                    f"vs store={store.current.symbol} — clearing store"
                )
                store._position = None
                store._save()
                executor.current_position = None
        elif real_pos and not store.current:
            # 바이낸스에만 있고 store에 없음 — 청산 실패로 남은 포지션일 수 있음
            # store에 등록해서 관리 대상으로 편입
            from src.engine.decision_engine import Position
            side = real_pos.get("side", "long")
            direction = "LONG" if side == "long" else "SHORT"
            entry_price = float(real_pos.get("entryPrice", 0))
            logger.warning(
                f"Unmanaged position found: {real_pos['symbol']} {direction} "
                f"— registering in store for management"
            )
            orphan = Position(
                symbol=real_pos["symbol"],
                direction=direction,
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc),
                sl_price=entry_price * (0.95 if direction == "LONG" else 1.03),
                tp_price=entry_price * (1.10 if direction == "LONG" else 0.92),
                entry_score=0.0,
            )
            store.open(orphan)
            executor.current_position = orphan
        elif not real_pos and store.current:
            # store에만 있고 바이낸스에 없음 — 이미 청산됨 (수동/SL/TP)
            logger.warning(
                f"Position {store.current.symbol} no longer exists on Binance — clearing store"
            )
            store._position = None
            store._save()
            executor.current_position = None
        # 둘 다 없으면 정상 (현금)

    except Exception as e:
        logger.warning(f"Position sync skipped: {e}")


def _log_signals(now, scores, signals, bundle=None, store=None, drawdown=None,
                  aggregator=None, engine=None, order=None):
    """
    고도화된 시그널 로그.

    한 행에 담기는 정보:
      1. timestamp — 시그널 생성 시각 (UTC)
      2. scores — 앙상블 최종 점수 (심볼별)
      3. alphas — 개별 알파 score/confidence + 내부 메타데이터
      4. prices — 시그널 시점 가격 (나중에 forward return 계산용)
      5. market — 시장 컨텍스트 (BTC 가격/수익률, 변동성)
      6. funding — 최신 펀딩비율
      7. oi — OI 변화율
      8. position — 현재 포지션 상태
      9. daily_pnl — 당일 누적 손익
    """
    sig_dir = LOGS_DIR / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    path = sig_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"

    # 1. 개별 알파 (score + confidence + metadata)
    alphas_data = {}
    for symbol, alpha_sigs in signals.items():
        alphas_data[symbol] = {}
        for alpha_name, sig in alpha_sigs.items():
            alpha_entry = {
                "s": round(sig.score, 4),
                "c": round(sig.confidence, 3),
            }
            # 알파별 내부 피처 (RSI값, z-score 등)
            if sig.metadata:
                for k, v in sig.metadata.items():
                    if isinstance(v, (int, float)) and k != "error":
                        alpha_entry[k] = round(v, 4) if isinstance(v, float) else v
            alphas_data[symbol][alpha_name] = alpha_entry

    # 2. 시그널 시점 가격 (forward return 사후 계산용)
    prices = {}
    if bundle:
        for symbol in bundle.universe:
            # 5분봉 최신 close
            if symbol in bundle.ohlcv_5m and not bundle.ohlcv_5m[symbol].empty:
                prices[symbol] = round(float(bundle.ohlcv_5m[symbol]["close"].iloc[-1]), 4)
            elif symbol in bundle.ohlcv_1h and not bundle.ohlcv_1h[symbol].empty:
                prices[symbol] = round(float(bundle.ohlcv_1h[symbol]["close"].iloc[-1]), 4)

    # 3. 시장 컨텍스트
    market = {}
    if bundle:
        # BTC 가격 + 수익률
        btc_sym = next((s for s in bundle.universe if "BTC" in s), None)
        if btc_sym and btc_sym in bundle.ohlcv_1d and not bundle.ohlcv_1d[btc_sym].empty:
            btc_df = bundle.ohlcv_1d[btc_sym]
            btc_close = btc_df["close"].values
            market["btc_price"] = round(float(btc_close[-1]), 2)
            if len(btc_close) >= 2:
                market["btc_ret_1d"] = round(float(btc_close[-1] / btc_close[-2] - 1), 4)
            if len(btc_close) >= 8:
                market["btc_ret_7d"] = round(float(btc_close[-1] / btc_close[-8] - 1), 4)
            if len(btc_close) >= 21:
                import numpy as np
                daily_rets = np.diff(btc_close[-21:]) / btc_close[-21:-1]
                market["btc_vol_20d"] = round(float(np.std(daily_rets) * np.sqrt(365)), 4)

        # VolatilityRegime confidence (시장 전체 변동성 상태)
        for sym_sigs in signals.values():
            vol_sig = sym_sigs.get("VolatilityRegime")
            if vol_sig:
                market["vol_regime"] = round(vol_sig.confidence, 3)
                break

    # 4. 펀딩비율
    funding = {}
    if bundle:
        for symbol in bundle.universe:
            if symbol in bundle.funding_rates and not bundle.funding_rates[symbol].empty:
                fr = float(bundle.funding_rates[symbol]["fundingRate"].iloc[-1])
                funding[symbol] = round(fr, 6)

    # 5. OI 변화율
    oi = {}
    if bundle:
        for symbol in bundle.universe:
            if symbol in bundle.open_interest:
                oi[symbol] = round(bundle.open_interest[symbol].change_pct, 2)

    # 6. 거래량 (5분봉 최근 vs 평균 — 시장 활성도)
    volumes = {}
    if bundle:
        for symbol in bundle.universe:
            if symbol in bundle.ohlcv_5m and not bundle.ohlcv_5m[symbol].empty:
                vol_data = bundle.ohlcv_5m[symbol]["volume"].values
                if len(vol_data) >= 12:
                    recent = float(vol_data[-6:].mean())  # 최근 30분
                    baseline = float(vol_data.mean())     # 전체 평균
                    ratio = round(recent / baseline, 2) if baseline > 0 else 1.0
                    volumes[symbol] = ratio

    # 7. 롱/숏 비율
    ls_ratio = {}
    if bundle:
        for symbol in bundle.universe:
            if symbol in bundle.long_short_ratio:
                ls_ratio[symbol] = round(bundle.long_short_ratio[symbol], 3)

    # 8. 오더북 스프레드 (유동성 지표)
    spreads = {}
    if bundle:
        for symbol in bundle.universe:
            if symbol in bundle.orderbook_snapshots and bundle.orderbook_snapshots[symbol]:
                latest = bundle.orderbook_snapshots[symbol][-1]
                spreads[symbol] = round(latest.spread_bps, 1)

    # 9. 코인별 1일/7일 수익률
    coin_returns = {}
    if bundle:
        for symbol in bundle.universe:
            if symbol in bundle.ohlcv_1d and not bundle.ohlcv_1d[symbol].empty:
                closes = bundle.ohlcv_1d[symbol]["close"].values
                ret = {}
                if len(closes) >= 2:
                    ret["1d"] = round(float(closes[-1] / closes[-2] - 1), 4)
                if len(closes) >= 8:
                    ret["7d"] = round(float(closes[-1] / closes[-8] - 1), 4)
                if ret:
                    coin_returns[symbol] = ret

    # 10. 포지션 상태
    position = None
    if store and store.current:
        pos = store.current
        current_price = prices.get(pos.symbol, 0)
        unrealized_pnl = 0.0
        if pos.entry_price > 0 and current_price > 0:
            if pos.direction == "LONG":
                unrealized_pnl = (current_price - pos.entry_price) / pos.entry_price
            else:
                unrealized_pnl = (pos.entry_price - current_price) / pos.entry_price
        position = {
            "symbol": pos.symbol,
            "direction": pos.direction,
            "entry_price": round(pos.entry_price, 4),
            "current_pnl": round(unrealized_pnl, 4),
            "hold_min": round((now - pos.entry_time).total_seconds() / 60, 1),
            "trailing": pos.trailing_active,
        }

    # 7. 당일 누적 손익
    daily_pnl = None
    if drawdown:
        daily_pnl = {
            "pnl_pct": round(drawdown.daily_pnl_pct, 4),
            "blocked": drawdown.is_blocked(),
        }

    # EMA 전 원본 스코어 (다른 EMA 파라미터로 백테스트 가능)
    raw_scores = {}
    if aggregator and hasattr(aggregator, '_raw_scores'):
        raw_scores = {s: round(sc, 4) for s, sc in aggregator._raw_scores.items()}

    # 의사결정 상태 (왜 진입/보유/청산 했는지 추적)
    decision = {}
    if engine:
        decision["pending"] = {s: [round(x, 4) for x in sl] for s, sl in engine._pending_entry.items()} if engine._pending_entry else {}
        decision["cooldown_symbol"] = engine._last_exit_symbol or ""
        if engine._last_exit_time:
            cd_remaining = max(0, (engine._last_exit_time + timedelta(minutes=COOLDOWN_MINUTES) - now).total_seconds())
            decision["cooldown_remaining_sec"] = round(cd_remaining)
        banned = {}
        for sym, losses in engine._symbol_losses.items():
            if len(losses) >= 3:
                banned[sym] = len(losses)
        if banned:
            decision["banned_symbols"] = banned
    if order:
        decision["action"] = order.action
        decision["target"] = order.symbol
        decision["direction"] = order.direction
        decision["reason"] = order.reason

    # 조립
    entry = {
        "t": now.isoformat(),
        "scores": {s: round(sc, 4) for s, sc in scores.items()},
        "raw_scores": raw_scores,
        "alphas": alphas_data,
        "prices": prices,
        "market": market,
        "funding": funding,
        "oi": oi,
    }
    if decision:
        entry["decision"] = decision
    if position:
        entry["position"] = position
    if daily_pnl:
        entry["daily_pnl"] = daily_pnl
    if volumes:
        entry["volumes"] = volumes
    if ls_ratio:
        entry["ls_ratio"] = ls_ratio
    if spreads:
        entry["spreads"] = spreads
    if coin_returns:
        entry["returns"] = coin_returns

    with open(path, "a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Strategy Ensemble Daemon")
    parser.add_argument("--dry-run", action="store_true", help="Signals only, no orders")
    args = parser.parse_args()
    asyncio.run(main_loop(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
