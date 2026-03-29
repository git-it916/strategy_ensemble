#!/usr/bin/env python3
"""
RL 로그 JSONL → Parquet 변환.

ENTRY/EXIT 페어를 매칭하여 experience tuple 생성:
  (state, action, reward, next_state, done)

Usage:
    python scripts/convert_rl_logs.py                    # 전체 변환
    python scripts/convert_rl_logs.py --symbol btc       # BTC만
    python scripts/convert_rl_logs.py --output data/rl   # 출력 경로 지정
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_jsonl(path: Path) -> list[dict]:
    """JSONL 파일 로드."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_all_logs(base_dir: str, log_type: str, symbol: str | None = None) -> list[dict]:
    """모든 JSONL 파일에서 레코드 로드."""
    base = Path(base_dir) / log_type
    if not base.exists():
        print(f"  디렉토리 없음: {base}")
        return []

    records = []
    dirs = [base / symbol] if symbol else sorted(base.iterdir())
    for sym_dir in dirs:
        if not sym_dir.is_dir():
            continue
        for fpath in sorted(sym_dir.glob("*.jsonl")):
            recs = load_jsonl(fpath)
            records.extend(recs)
            print(f"  {fpath}: {len(recs)} records")

    return records


def pair_trades(trade_records: list[dict]) -> list[dict]:
    """ENTRY/EXIT 페어를 매칭하여 완전한 에피소드 생성."""
    # trade_id 기준으로 그룹핑
    by_id: dict[str, dict] = {}
    # trade_id가 없는 경우 심볼별 순서로 매칭
    pending_entries: dict[str, dict] = {}

    episodes = []

    for rec in trade_records:
        tid = rec.get("trade_id", "")
        event = rec.get("event", "")
        sym = rec.get("symbol", "")

        if event == "ENTRY":
            if tid:
                by_id[tid] = {"entry": rec}
            pending_entries[sym] = rec

        elif event == "EXIT":
            entry_rec = None
            if tid and tid in by_id:
                entry_rec = by_id.pop(tid).get("entry")
            elif sym in pending_entries:
                entry_rec = pending_entries.pop(sym)

            if entry_rec is None:
                continue

            episodes.append({
                "trade_id": tid or f"{sym}_{entry_rec.get('ts', 'unknown')}",
                "symbol": sym,
                "strategy": entry_rec.get("strategy", ""),
                "entry_ts": entry_rec.get("ts", ""),
                "exit_ts": rec.get("ts", ""),
                # State at entry
                "entry_features": entry_rec.get("state", {}).get("features", []),
                # State at exit
                "exit_features": rec.get("state", {}).get("features", []),
                # Action
                "action_type": entry_rec.get("action", {}).get("type", ""),
                "entry_price": entry_rec.get("action", {}).get("entry_price", 0),
                "sl_price": entry_rec.get("action", {}).get("sl_price", 0),
                "risk": entry_rec.get("action", {}).get("risk", 0),
                # Reward
                "pnl_pct": rec.get("reward", {}).get("pnl_pct", 0),
                "pnl_r": rec.get("reward", {}).get("pnl_r", 0),
                "peak_r": rec.get("reward", {}).get("peak_r", 0),
                "max_adverse_r": rec.get("reward", {}).get("max_adverse_r", 0),
                "hold_bars": rec.get("reward", {}).get("hold_bars", 0),
                "exit_reason": rec.get("action", {}).get("exit_reason", ""),
                "tp1_hit": rec.get("reward", {}).get("tp1_hit", False),
                "tp2_hit": rec.get("reward", {}).get("tp2_hit", False),
                "tp3_hit": rec.get("reward", {}).get("tp3_hit", False),
                "direction": rec.get("reward", {}).get("direction", ""),
            })

    return episodes


def build_experience_df(episodes: list[dict]) -> pd.DataFrame:
    """에피소드를 experience DataFrame으로 변환."""
    if not episodes:
        return pd.DataFrame()

    rows = []
    for ep in episodes:
        entry_f = ep["entry_features"]
        exit_f = ep["exit_features"]
        n_features = max(len(entry_f), len(exit_f))

        row = {
            "trade_id": ep["trade_id"],
            "symbol": ep["symbol"],
            "strategy": ep["strategy"],
            "entry_ts": ep["entry_ts"],
            "exit_ts": ep["exit_ts"],
            "action": ep["action_type"],
            "direction": ep["direction"],
            "pnl_pct": ep["pnl_pct"],
            "pnl_r": ep["pnl_r"],
            "peak_r": ep["peak_r"],
            "max_adverse_r": ep["max_adverse_r"],
            "hold_bars": ep["hold_bars"],
            "exit_reason": ep["exit_reason"],
            "tp1_hit": ep["tp1_hit"],
            "tp2_hit": ep["tp2_hit"],
            "tp3_hit": ep["tp3_hit"],
        }

        # Feature columns
        for i, val in enumerate(entry_f):
            row[f"entry_f{i}"] = val
        for i, val in enumerate(exit_f):
            row[f"exit_f{i}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def build_state_sequence_df(state_records: list[dict]) -> pd.DataFrame:
    """상태 로그를 시계열 DataFrame으로 변환."""
    if not state_records:
        return pd.DataFrame()

    rows = []
    for rec in state_records:
        features = rec.get("state", {}).get("features", [])
        pos = rec.get("state", {}).get("position", {})
        action = rec.get("action", {})
        step_rw = rec.get("step_reward", {})

        row = {
            "symbol": rec.get("symbol", ""),
            "ts": rec.get("ts", ""),
            "candle_ts": rec.get("candle_ts", ""),
            "action_type": action.get("type", "HOLD"),
            "signal_generated": action.get("signal_generated", False),
            "hold_reason": action.get("hold_reason", ""),
            "is_in_position": pos.get("is_active", False),
            "unrealized_r": pos.get("unrealized_r", 0),
            "hold_bars": pos.get("hold_bars", 0),
            "step_reward_delta": step_rw.get("unrealized_pnl_delta", 0),
        }

        for i, val in enumerate(features):
            row[f"f{i}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="RL Log Converter")
    parser.add_argument("--input", default="logs/rl", help="RL 로그 베이스 디렉토리")
    parser.add_argument("--output", default="data/rl", help="Parquet 출력 디렉토리")
    parser.add_argument("--symbol", default=None, help="특정 심볼만 (btc, sol, xrp)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 거래 로그 → experience
    print("\n=== Trade Logs ===", flush=True)
    trade_records = load_all_logs(args.input, "trades", args.symbol)
    print(f"  총 {len(trade_records)} trade records", flush=True)

    episodes = pair_trades(trade_records)
    print(f"  매칭된 에피소드: {len(episodes)}건", flush=True)

    if episodes:
        exp_df = build_experience_df(episodes)
        exp_path = output_dir / "experience.parquet"
        exp_df.to_parquet(exp_path, index=False)
        print(f"  저장: {exp_path} ({len(exp_df)} rows, {exp_df.shape[1]} cols)", flush=True)

        # 요약
        print(f"\n  --- 에피소드 요약 ---", flush=True)
        print(f"  심볼: {exp_df['symbol'].value_counts().to_dict()}", flush=True)
        print(f"  평균 PnL(R): {exp_df['pnl_r'].mean():+.3f}", flush=True)
        print(f"  승률: {(exp_df['pnl_r'] > 0).mean()*100:.1f}%", flush=True)
        print(f"  평균 보유: {exp_df['hold_bars'].mean():.1f}봉", flush=True)

    # 상태 로그 → 시계열
    print("\n=== State Logs ===", flush=True)
    state_records = load_all_logs(args.input, "states", args.symbol)
    print(f"  총 {len(state_records)} state records", flush=True)

    if state_records:
        state_df = build_state_sequence_df(state_records)
        state_path = output_dir / "states.parquet"
        state_df.to_parquet(state_path, index=False)
        print(f"  저장: {state_path} ({len(state_df)} rows, {state_df.shape[1]} cols)", flush=True)

    print("\n완료!", flush=True)


if __name__ == "__main__":
    main()
