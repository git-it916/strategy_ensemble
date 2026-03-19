"""
백테스트 — 시그널 로그 기반 시뮬레이션.
새 설정(인트라데이 가중치 + 점진적 청산)으로 과거 데이터 재현.
"""

import json
import sys
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# === 설정 ===
WEIGHTS = {
    'MomentumMultiScale': 0.25, 'IntradayVWAPV2': 0.20,
    'OrderbookImbalance': 0.18, 'IntradayRSIV2': 0.15,
    'FundingCarryEnhanced': 0.08, 'MomentumComposite': 0.06,
    'MeanReversionMultiHorizon': 0.05, 'DerivativesSentiment': 0.03,
}
LONG_TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
SHORT_TH = float(sys.argv[2]) if len(sys.argv) > 2 else -0.12
CONFIRM = 2
MIN_RISE = 0.01
MIN_HOLD = 15
COOLDOWN = 20
MAX_DAY = 6
L_SL = -0.035
L_TP = 0.10
S_SL = -0.025
S_TP = 0.08
TR_ACT = 0.025
TR_DIST = 0.015
FADE_TH = 0.08
FADE_M = 15
WEAK_TH = 0.04
WEAK_M = 10
LEVERAGE = 3


def sim_alpha(name, data):
    s = data.get('s', 0)
    c = data.get('c', 0)

    if name == 'MomentumMultiScale':
        s *= 0.5
    elif name == 'MomentumComposite':
        s *= 0.5
        am = data.get('abs_mom', 0)
        rs = data.get('rank_score', 0)
        ra = data.get('risk_adj', 0)
        c = 0.50 if np.sign(am) == np.sign(rs) == np.sign(ra) else 0.35
    elif name == 'FundingCarryEnhanced':
        c = 0.5
    elif name == 'MeanReversionMultiHorizon':
        valid = [data.get(k) for k in ['z_short', 'z_mid', 'z_long'] if data.get(k) is not None]
        c = 0.55 if len(valid) == 3 and np.sign(valid[0]) == np.sign(valid[1]) == np.sign(valid[2]) else 0.4
    elif name == 'IntradayVWAPV2':
        z = data.get('z', 0)
        if abs(z) < 0.8:
            s = 0.0
        elif z > 0:
            s = float(-np.tanh((z - 0.8) / 1.5))
        else:
            s = float(np.tanh((-z - 0.8) / 1.5))
        c = max(min(abs(z) / 3.0, 0.8), 0.2)
    elif name == 'DerivativesSentiment':
        oi = data.get('oi_change', 0)
        fr = data.get('funding', 0)
        ls = data.get('ls_ratio', 0.5)
        sq = 0.0
        if oi > 5 and fr < -0.00005:
            sq = min(1.0, oi / 20 + abs(fr) * 5000)
        elif oi > 5 and fr > 0.00005:
            sq = -min(1.0, oi / 20 + abs(fr) * 5000)
        la = 0.0
        if ls > 0.55:
            la = -min(1.0, (ls - 0.55) / 0.15) * 0.4
        elif ls < 0.45:
            la = min(1.0, (0.45 - ls) / 0.15) * 0.4
        s = float(np.clip(sq * 0.6 + la * 0.4, -1, 1))
        c = 0.6 if abs(s) > 0.05 else 0.3
    elif name == 'OrderbookImbalance':
        c = max(min(abs(s) * 1.5, 0.7), 0.2)

    return s, c


def calc_scores(row):
    results = {}
    if 'alphas' not in row or not row['alphas']:
        return results
    for sym, alphas in row['alphas'].items():
        vc = max(alphas.get('VolatilityRegime', {}).get('c', 1.0), 0.5)
        raw = tw = 0
        for aname, adata in alphas.items():
            if aname == 'VolatilityRegime':
                continue
            w = WEIGHTS.get(aname, 0)
            if w <= 0:
                continue
            s, c = sim_alpha(aname, adata)
            if c <= 0:
                continue
            raw += s * c * w
            tw += w
        if tw > 0:
            results[sym] = (raw / tw) * vc
    return results


def run_backtest():
    # 로드
    log_dir = Path(__file__).parent.parent / "logs" / "signals"
    rows = []
    for f in sorted(log_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r.get('t') and 'alphas' in r and r['alphas'] and 'prices' in r and r['prices']:
                        if len(r['alphas']) >= 10:
                            rows.append(r)
    rows.sort(key=lambda r: r['t'])
    print(f'데이터: {len(rows)}행, {rows[0]["t"][:10]} ~ {rows[-1]["t"][:10]}')

    # 백테스트
    trades = []
    pos = None
    pend = {}
    last_exit_t = None
    last_exit_s = ""
    daily_t = defaultdict(int)

    for row in rows:
        t = datetime.fromisoformat(row['t'])
        day = row['t'][:10]
        prices = row.get('prices', {})
        scores = calc_scores(row)

        # === 포지션 보유 중 ===
        if pos:
            cp = prices.get(pos['sym'], 0)
            cs = scores.get(pos['sym'], 0)
            if cp <= 0:
                continue
            hm = (t - pos['et']).total_seconds() / 60
            pnl = (cp / pos['ep'] - 1) if pos['d'] == 'LONG' else (1 - cp / pos['ep'])
            eff = cs if pos['d'] == 'LONG' else -cs
            sl = L_SL if pos['d'] == 'LONG' else S_SL
            tp = L_TP if pos['d'] == 'LONG' else S_TP
            ex = None

            # SL
            if pnl <= sl:
                ex = 'SL'

            # 트레일링
            if not ex and pnl >= TR_ACT:
                pos['ta'] = True
                pos['pp'] = max(pos['pp'], pnl)
                if pos['pp'] - pnl >= TR_DIST:
                    ex = 'TRAILING'
            elif not ex:
                pos['pp'] = max(pos['pp'], pnl)

            # TP
            if not ex and not pos['ta'] and pnl >= tp:
                ex = 'TP'

            # 스코어 기반 (최소 보유 후)
            if not ex and hm >= MIN_HOLD:
                if pos['d'] == 'LONG' and cs < SHORT_TH:
                    ex = 'REVERSAL'
                elif pos['d'] == 'SHORT' and cs > LONG_TH:
                    ex = 'REVERSAL'

                if not ex and eff < WEAK_TH:
                    if pos['ws'] is None:
                        pos['ws'] = t
                    elif (t - pos['ws']).total_seconds() / 60 >= WEAK_M:
                        ex = 'WEAK'
                else:
                    pos['ws'] = None

                if not ex and eff < FADE_TH:
                    if pos['fs'] is None:
                        pos['fs'] = t
                    elif (t - pos['fs']).total_seconds() / 60 >= FADE_M:
                        ex = 'FADE'
                else:
                    pos['fs'] = None

            if ex:
                roi = pnl * LEVERAGE
                trades.append({
                    'et': pos['et'].strftime('%m/%d %H:%M'),
                    'xt': t.strftime('%m/%d %H:%M'),
                    'sym': pos['sym'].split('/')[0],
                    'd': pos['d'],
                    'ep': pos['ep'],
                    'xp': cp,
                    'pnl': round(pnl * 100, 2),
                    'roi': round(roi * 100, 2),
                    'hm': round(hm),
                    'r': ex,
                    'es': round(pos['es'], 4),
                })
                last_exit_t = t
                last_exit_s = pos['sym']
                pos = None
            continue

        # === 진입 체크 ===
        if not scores or not prices:
            continue
        if daily_t[day] >= MAX_DAY:
            continue

        # 시장 데이터 (필터용)
        market = row.get('market', {})
        volumes = row.get('volumes', {})
        returns_1h = {}
        btc_1h_ret = market.get('btc_ret_1d', 0)  # 근사값
        # 5분봉 기반 1시간 수익률이 있으면 사용
        for sym in scores:
            if sym in prices:
                # returns 필드에서 1d 수익률 참고 (1h 수익률 없으면 스킵)
                pass

        lc = [(s, v) for s, v in scores.items() if v > LONG_TH and s in prices]
        hc = [(s, v) for s, v in scores.items() if v < SHORT_TH and s in prices]
        ac = [(s, v, 'LONG') for s, v in lc] + [(s, v, 'SHORT') for s, v in hc]

        # === 필터 적용 (실 진입과 동일) ===
        filtered = []
        for sym, sc, direction in ac:
            # 필터 1: 거래량 확인 — 거래량 없는 움직임 제외
            vol_ratio = volumes.get(sym, 1.0)
            if vol_ratio < 0.7:
                continue
            filtered.append((sym, sc, direction))
        ac = filtered

        if not ac:
            pend = {}
            continue

        bs, bv, bd = max(ac, key=lambda x: abs(x[1]))

        # 쿨다운
        if last_exit_t and last_exit_s == bs:
            if (t - last_exit_t).total_seconds() / 60 < COOLDOWN:
                continue

        prev = pend.get(bs, [])
        asc = prev + [bv]
        if len(prev) >= (CONFIRM - 1):
            delta = abs(asc[-1]) - abs(asc[0])
            th = LONG_TH if bd == 'LONG' else abs(SHORT_TH)
            if delta >= MIN_RISE and abs(asc[-1]) >= th:
                pos = {
                    'sym': bs, 'd': bd, 'ep': prices[bs], 'et': t,
                    'es': bv, 'pp': 0, 'ta': False, 'fs': None, 'ws': None,
                }
                pend = {}
                daily_t[day] += 1
                continue

        prev.append(bv)
        pend = {bs: prev[-(CONFIRM - 1):]}

    # === 결과 출력 ===
    print(f'\n{"=" * 75}')
    print(f'총 거래: {len(trades)}건')
    print(f'{"=" * 75}')

    if not trades:
        print('거래 없음')
        return

    total_pnl = sum(t['pnl'] for t in trades)
    total_roi = sum(t['roi'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    print(f'\n{"진입":>14s} {"코인":>5s} {"방향":>5s} {"보유":>5s} {"PnL%":>7s} {"ROI%":>7s} {"청산":>10s} {"score":>7s}')
    print('-' * 72)
    for t in trades:
        w = '✅' if t['pnl'] > 0 else '❌'
        print(
            f'{t["et"]:>14s} {t["sym"]:>5s} {t["d"]:>5s} {t["hm"]:>4d}m '
            f'{t["pnl"]:>+6.2f}% {t["roi"]:>+6.2f}% {t["r"]:>10s} {t["es"]:>+.4f} {w}'
        )

    print(f'\n{"=" * 72}')
    print(f'승률: {len(wins)}/{len(trades)} ({len(wins) / len(trades) * 100:.0f}%)')
    print(f'누적 PnL: {total_pnl:+.2f}% (가격)')
    print(f'누적 ROI: {total_roi:+.2f}% (3x leverage)')
    print(f'평균 보유: {np.mean([t["hm"] for t in trades]):.0f}분')
    if wins:
        print(f'평균 수익: +{np.mean([t["pnl"] for t in wins]):.2f}%')
    if losses:
        print(f'평균 손실: {np.mean([t["pnl"] for t in losses]):.2f}%')

    print(f'\n=== 일별 ===')
    bd = defaultdict(list)
    for t in trades:
        bd[t['et'][:5]].append(t)
    cum = 0
    for d in sorted(bd.keys()):
        dt = bd[d]
        dp = sum(t['roi'] for t in dt)
        cum += dp
        dw = sum(1 for t in dt if t['pnl'] > 0)
        print(f'  {d}: {len(dt)}건 승{dw}/패{len(dt) - dw} ROI={dp:+.2f}% (누적={cum:+.2f}%)')

    print(f'\n=== 청산 사유별 ===')
    br = defaultdict(list)
    for t in trades:
        br[t['r']].append(t)
    for r in sorted(br.keys()):
        rt = br[r]
        ap = np.mean([t['pnl'] for t in rt])
        aw = sum(1 for t in rt if t['pnl'] > 0)
        print(f'  {r:10s}: {len(rt)}건 승{aw} avg_pnl={ap:+.2f}%')


if __name__ == '__main__':
    run_backtest()
