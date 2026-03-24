#!/usr/bin/env python3
"""
Dynamic R:R Strategy — Exhaustive Grid Search Optimizer

Phase 1: Core parameters (EMA, RSI, SL, TP) ~28K combos
Phase 2: Enhancement filters (vol, ADX, trend exit, cooldown) on top 30
Phase 3: Fine-tune around best
"""

from __future__ import annotations
import sys, time, itertools
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

LEVERAGE = 3

# =================================================================
# Data Fetch
# =================================================================
def fetch_data(symbol, tf, months):
    import ccxt, yaml
    keys_path = Path(__file__).parent.parent / "config" / "keys.yaml"
    with open(keys_path) as f:
        keys = yaml.safe_load(f)
    bcfg = keys.get("binance", {})
    ex = ccxt.binance({
        "apiKey": bcfg.get("api_key", ""),
        "secret": bcfg.get("api_secret", ""),
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=months * 30)
    since_ms, until_ms = int(since.timestamp() * 1000), int(now.timestamp() * 1000)

    all_data, cursor = [], since_ms
    print(f"  Fetching {symbol} {tf} ({months}mo)...")
    while cursor < until_ms:
        batch = ex.fetch_ohlcv(symbol, tf, since=cursor, limit=1500)
        if not batch: break
        all_data.extend(batch)
        cursor = batch[-1][0] + 1
        if batch[-1][0] >= until_ms: break
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(all_data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.drop_duplicates(subset=["ts"]).set_index("ts")
    print(f"  Got {len(df)} bars ({since.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')})")
    return df

# =================================================================
# Indicator Precomputation
# =================================================================
def ema_pd(s, p): return s.ewm(span=p, adjust=False).mean()

def precompute(df, ema_lens, swing_lbs):
    c, h, l = df["close"], df["high"], df["low"]
    n = len(df)
    ind = {"close": c.values.copy(), "high": h.values.copy(), "low": l.values.copy(), "n": n}

    # EMAs
    for p in ema_lens:
        ind[f"ema_{p}"] = ema_pd(c, p).values

    # RSI (14)
    delta = c.diff()
    ag = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    al = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    ind["rsi"] = (100 - 100 / (1 + ag / al.replace(0, np.nan))).values

    # ATR (14)
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/14, min_periods=14).mean()
    ind["atr"] = atr_s.values
    ind["atr_sma"] = atr_s.rolling(42, min_periods=10).mean().values

    # ADX/DMI (14)
    ph, pl_ = h.shift(1), l.shift(1)
    pdm = (h - ph).clip(lower=0)
    mdm = (pl_ - l).clip(lower=0)
    pdm = pdm.where(pdm > mdm, 0)
    mdm = mdm.where(mdm > pdm, 0)
    atr_v = atr_s.replace(0, np.nan)
    dip = 100 * ema_pd(pdm, 14) / atr_v
    dim = 100 * ema_pd(mdm, 14) / atr_v
    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    ind["adx"] = ema_pd(dx, 14).values

    # Swing high/low
    for lb in swing_lbs:
        ind[f"swl_{lb}"] = l.rolling(lb, min_periods=lb).min().values
        ind[f"swh_{lb}"] = h.rolling(lb, min_periods=lb).max().values

    # EMA crosses — vectorized
    fast_lens = sorted([x for x in ema_lens if x <= 25])
    slow_lens = sorted([x for x in ema_lens if 30 <= x <= 60])
    for fl in fast_lens:
        for sl in slow_lens:
            f = ind[f"ema_{fl}"]
            s = ind[f"ema_{sl}"]
            above = f > s
            prev_above = np.roll(above, 1); prev_above[0] = above[0]
            ind[f"cu_{fl}_{sl}"] = above & ~prev_above
            ind[f"cd_{fl}_{sl}"] = ~above & prev_above

    return ind

# =================================================================
# Strategy Simulation (optimized inner loop)
# =================================================================
def simulate(ind, p):
    close = ind["close"]; high = ind["high"]; low = ind["low"]; n = ind["n"]
    cu = ind[f"cu_{p['ef']}_{p['es']}"]
    cd = ind[f"cd_{p['ef']}_{p['es']}"]
    ema_t = ind[f"ema_{p['et']}"]
    rsi = ind["rsi"]; atr = ind["atr"]; atr_sma = ind["atr_sma"]; adx = ind["adx"]

    rb = p["rb"]; rbe = p["rbe"]
    slm = p["slm"]; am = p.get("am", 1.5); slb = p.get("slb", 5)
    t1r, t2r, t3r = p["t1"], p["t2"], p["t3"]
    vf = p.get("vf"); admn = p.get("admn"); te = p.get("te", False); cdb = p.get("cd", 0)

    use_swing = slm == "S"
    if use_swing:
        swl = ind[f"swl_{slb}"]; swh = ind[f"swh_{slb}"]

    warmup = p["et"] + 15
    ld = 0; act = False; d = 0
    entry = sl = tp1 = tp2 = tp3 = risk = trail = 0.0
    t1h = t2h = t3h = False; cdr = 0
    pnls = []

    for i in range(warmup, n):
        ci = close[i]; hi = high[i]; li = low[i]
        if cdr > 0: cdr -= 1

        if act:
            ex = False; ep = 0.0
            if te:
                if d == 1 and ci < ema_t[i]: ep = ci; ex = True
                elif d == -1 and ci > ema_t[i]: ep = ci; ex = True
            if not ex:
                if d == 1:
                    if hi >= tp3 and not t3h: t3h = True; trail = tp2
                    if hi >= tp2 and not t2h: t2h = True; trail = tp1
                    if hi >= tp1 and not t1h: t1h = True; trail = entry
                    st = trail if t1h else sl
                    if li <= st: ep = st; ex = True
                else:
                    if li <= tp3 and not t3h: t3h = True; trail = tp2
                    if li <= tp2 and not t2h: t2h = True; trail = tp1
                    if li <= tp1 and not t1h: t1h = True; trail = entry
                    st = trail if t1h else sl
                    if hi >= st: ep = st; ex = True
            if ex:
                pnl = ((ep-entry)/entry*100*LEVERAGE) if d==1 else ((entry-ep)/entry*100*LEVERAGE)
                pnls.append(pnl)
                act = False
                if pnl < 0 and cdb > 0: cdr = cdb

        et_v = ema_t[i]; rsi_v = rsi[i]; atr_v = atr[i]
        if np.isnan(et_v) or np.isnan(rsi_v) or np.isnan(atr_v): continue
        if cdr > 0: continue
        if vf is not None:
            asma = atr_sma[i]
            if asma > 0 and not np.isnan(asma) and atr_v / asma > vf: continue
        if admn is not None:
            adx_v = adx[i]
            if np.isnan(adx_v) or adx_v < admn: continue

        ls = cu[i] and ci > et_v and rsi_v > rb
        ss = cd[i] and ci < et_v and rsi_v < rbe
        if ls and ld == 1: ls = False
        if ss and ld == -1: ss = False
        if ls and ss: ss = False
        if not ls and not ss: continue

        if act:
            sd = 1 if ls else -1
            if sd != d:
                pnl = ((ci-entry)/entry*100*LEVERAGE) if d==1 else ((entry-ci)/entry*100*LEVERAGE)
                pnls.append(pnl); act = False

        if not act:
            entry = ci; av = atr_v
            if ls:
                ld = 1; d = 1
                sl = (swl[i] if use_swing and not np.isnan(swl[i]) else entry - av*am)
                risk = abs(entry - sl)
                if risk < av*0.3: risk = av*0.5; sl = entry - risk
                tp1=entry+risk*t1r; tp2=entry+risk*t2r; tp3=entry+risk*t3r
                trail=sl; act=True; t1h=t2h=t3h=False
            elif ss:
                ld = -1; d = -1
                sl = (swh[i] if use_swing and not np.isnan(swh[i]) else entry + av*am)
                risk = abs(entry - sl)
                if risk < av*0.3: risk = av*0.5; sl = entry + risk
                tp1=entry-risk*t1r; tp2=entry-risk*t2r; tp3=entry-risk*t3r
                trail=sl; act=True; t1h=t2h=t3h=False

    if act:
        pnl = ((close[-1]-entry)/entry*100*LEVERAGE) if d==1 else ((entry-close[-1])/entry*100*LEVERAGE)
        pnls.append(pnl)

    return _metrics(pnls)

def _metrics(pnls):
    if not pnls or len(pnls) < 5:
        return {"n":len(pnls) if pnls else 0,"pf":0,"tot":0,"dd":-100,"wr":0,"bal":1000,"cal":0,"sc":-999}
    a = np.array(pnls); nn = len(a)
    w = a[a>0]; lo = a[a<=0]
    tot = float(a.sum())
    gp = float(w.sum()) if len(w) else 0
    gl = float(abs(lo.sum())) if len(lo) else 0.01
    pf = gp/gl; wr = len(w)/nn*100
    cum = np.cumsum(a); dd = float((cum - np.maximum.accumulate(cum)).min())
    bal = 1000.0
    for p in a: bal *= (1+p/100)
    cal = tot/max(abs(dd),1) if dd<0 else tot*10
    tf = min(nn,80)/80
    sc = cal * pf * tf
    return {"n":nn,"pf":round(pf,3),"tot":round(tot,2),"dd":round(dd,2),"wr":round(wr,1),"bal":round(bal,2),"cal":round(cal,3),"sc":round(sc,3)}

# =================================================================
# Grid Definitions
# =================================================================
def phase1():
    ef_vals = [8,10,12,15,18,20,25]
    es_vals = [30,35,40,50,60]
    et_vals = [80,100,120,150,200]
    rb_vals = [55,60,65]
    rbe_vals = [35,40,45]
    sl_cfgs = [("A",1.5,5),("A",2.0,5),("A",3.0,5),("S",1.5,3),("S",1.5,5),("S",1.5,7)]
    tp_cfgs = [(1,2,3),(1.5,3,5),(1,2,4)]

    combos = []
    for ef,es,et,rb,rbe,(slm,am,slb),(t1,t2,t3) in itertools.product(
        ef_vals,es_vals,et_vals,rb_vals,rbe_vals,sl_cfgs,tp_cfgs):
        if ef >= es: continue
        combos.append({"ef":ef,"es":es,"et":et,"rb":rb,"rbe":rbe,"slm":slm,"am":am,"slb":slb,"t1":t1,"t2":t2,"t3":t3})
    return combos

def phase2(bases):
    combos = []
    for bp in bases:
        for vf,admn,te,cd in itertools.product(
            [None,1.3,1.5,2.0],[None,20,25],[False,True],[0,10,20]):
            p = dict(bp); p["vf"]=vf; p["admn"]=admn; p["te"]=te; p["cd"]=cd
            combos.append(p)
    return combos

# =================================================================
# Display
# =================================================================
def fmt_params(p):
    ema = f"{p['ef']}/{p['es']}/{p['et']}"
    rsi = f"{p['rb']}/{p['rbe']}"
    sl = f"{'ATR' if p['slm']=='A' else 'SW'}{'x'+str(p['am']) if p['slm']=='A' else '_'+str(p['slb'])}"
    tp = f"{p['t1']}/{p['t2']}/{p['t3']}"
    return ema, rsi, sl, tp

def print_row(rank, p, m, extra=""):
    ema, rsi, sl, tp = fmt_params(p)
    print(f"  {rank:3d} | {ema:>13s} | {rsi:>5s} | {sl:>7s} | {tp:>9s} |{extra} {m['n']:3d} | {m['wr']:5.1f}% | {m['pf']:5.2f} | {m['tot']:+8.1f}% | {m['dd']:+7.1f}% | ${m['bal']:>9.2f} | {m['cal']:7.2f} | {m['sc']:7.2f}")

def print_header(extra_hdr=""):
    print(f"  {'#':>3s} | {'EMA':>13s} | {'RSI':>5s} | {'SL':>7s} | {'TP':>9s} |{extra_hdr} {'N':>3s} | {'WR':>5s} | {'PF':>5s} | {'Total':>8s} | {'DD':>7s} | {'$1000->':>9s} | {'Calmar':>7s} | {'Score':>7s}")
    print(f"  {'-'*130}")

# =================================================================
# Main
# =================================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("  DYNAMIC R:R — GRID SEARCH OPTIMIZER")
    print("  BTC/USDT 15m | 12 months | 3x leverage")
    print("=" * 70)

    df = fetch_data("BTC/USDT:USDT", "15m", 12)

    all_ema = [8,10,12,15,18,20,25,30,35,40,50,60,80,100,120,150,200]
    swing_lbs = [3,5,7]

    print("\n  Precomputing indicators...")
    t0 = time.time()
    ind = precompute(df, all_ema, swing_lbs)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ========== PHASE 1 ==========
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Core Parameter Grid")
    print(f"{'='*70}")
    combos = phase1()
    print(f"  Testing {len(combos)} combinations...")

    t0 = time.time()
    results = []
    for i, p in enumerate(combos):
        m = simulate(ind, p)
        results.append((p, m))
        if (i+1) % 5000 == 0:
            print(f"    {i+1}/{len(combos)} ({time.time()-t0:.1f}s)")

    elapsed = time.time()-t0
    print(f"  Done: {elapsed:.1f}s ({len(combos)/elapsed:.0f} sims/sec)")

    valid = [(p,m) for p,m in results if m["n"]>=15 and m["pf"]>=1.0 and m["dd"]>-80]
    valid.sort(key=lambda x: x[1]["sc"], reverse=True)
    print(f"  Valid (n>=15, PF>=1, DD>-80%): {len(valid)}/{len(combos)}")

    print(f"\n  TOP 30 — Phase 1:")
    print_header()
    for i,(p,m) in enumerate(valid[:30],1):
        print_row(i,p,m)

    # ========== PHASE 2 ==========
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Enhancement Filters (top 30 base)")
    print(f"{'='*70}")
    top30 = [p for p,m in valid[:30]]
    p2 = phase2(top30)
    print(f"  Testing {len(p2)} combinations...")

    t0 = time.time()
    r2 = []
    for i, p in enumerate(p2):
        m = simulate(ind, p)
        r2.append((p, m))
        if (i+1) % 1000 == 0:
            print(f"    {i+1}/{len(p2)}...")

    elapsed = time.time()-t0
    print(f"  Done: {elapsed:.1f}s")

    v2 = [(p,m) for p,m in r2 if m["n"]>=15 and m["pf"]>=1.0 and m["dd"]>-60]
    v2.sort(key=lambda x: x[1]["sc"], reverse=True)
    print(f"  Valid (n>=15, PF>=1, DD>-60%): {len(v2)}/{len(p2)}")

    ex_hdr = " VF  |ADX|TE|CD|"
    print(f"\n  TOP 30 — Phase 2:")
    print(f"  {'#':>3s} | {'EMA':>13s} | {'RSI':>5s} | {'SL':>7s} | {'TP':>9s} | {'VF':>4s}|{'ADX':>3s}|{'TE':>2s}|{'CD':>2s}| {'N':>3s} | {'WR':>5s} | {'PF':>5s} | {'Total':>8s} | {'DD':>7s} | {'$1000->':>9s} | {'Calmar':>7s} | {'Score':>7s}")
    print(f"  {'-'*150}")
    for i,(p,m) in enumerate(v2[:30],1):
        ema,rsi,sl,tp = fmt_params(p)
        vf = str(p.get('vf','-')) if p.get('vf') else '  -'
        admn = str(p.get('admn','-')) if p.get('admn') else ' -'
        te = 'Y' if p.get('te') else 'N'
        cd = str(p.get('cd',0))
        print(f"  {i:3d} | {ema:>13s} | {rsi:>5s} | {sl:>7s} | {tp:>9s} | {vf:>4s}|{admn:>3s}|{te:>2s}|{cd:>2s}| {m['n']:3d} | {m['wr']:5.1f}% | {m['pf']:5.2f} | {m['tot']:+8.1f}% | {m['dd']:+7.1f}% | ${m['bal']:>9.2f} | {m['cal']:7.2f} | {m['sc']:7.2f}")

    # ========== PHASE 3: Fine-tune ==========
    if v2:
        best_p = dict(v2[0][0])
        print(f"\n{'='*70}")
        print(f"  PHASE 3: Fine-tune around best")
        print(f"{'='*70}")

        ft = []
        base_ef, base_es = best_p["ef"], best_p["es"]

        # Fine-tune EMA fast
        for ef in range(max(5, base_ef-4), base_ef+5):
            if ef >= base_es: continue
            key = f"cu_{ef}_{base_es}"
            if key not in ind:
                # compute on-the-fly
                c = ind["close"]
                f_ema = pd.Series(c).ewm(span=ef, adjust=False).mean().values
                ind[f"ema_{ef}"] = f_ema
                s_ema = ind[f"ema_{base_es}"]
                above = f_ema > s_ema
                pa = np.roll(above,1); pa[0]=above[0]
                ind[f"cu_{ef}_{base_es}"] = above & ~pa
                ind[f"cd_{ef}_{base_es}"] = ~above & pa
            p = dict(best_p); p["ef"] = ef
            m = simulate(ind, p); ft.append((p,m))

        # Fine-tune EMA slow
        for es in range(max(base_ef+5, base_es-8), base_es+9, 2):
            if es <= base_ef: continue
            key = f"cu_{base_ef}_{es}"
            if key not in ind:
                c = ind["close"]
                s_ema = pd.Series(c).ewm(span=es, adjust=False).mean().values
                ind[f"ema_{es}"] = s_ema
                f_ema = ind[f"ema_{base_ef}"]
                above = f_ema > s_ema
                pa = np.roll(above,1); pa[0]=above[0]
                ind[f"cu_{base_ef}_{es}"] = above & ~pa
                ind[f"cd_{base_ef}_{es}"] = ~above & pa
            p = dict(best_p); p["es"] = es
            m = simulate(ind, p); ft.append((p,m))

        # Fine-tune EMA trend
        for et in range(max(60, best_p["et"]-30), best_p["et"]+31, 10):
            if et <= best_p["es"]: continue
            if f"ema_{et}" not in ind:
                ind[f"ema_{et}"] = pd.Series(ind["close"]).ewm(span=et, adjust=False).mean().values
            p = dict(best_p); p["et"] = et
            m = simulate(ind, p); ft.append((p,m))

        # Fine-tune RSI thresholds
        for rb in range(50, 70, 2):
            p = dict(best_p); p["rb"] = rb; m = simulate(ind, p); ft.append((p,m))
        for rbe in range(30, 50, 2):
            p = dict(best_p); p["rbe"] = rbe; m = simulate(ind, p); ft.append((p,m))

        # Fine-tune ATR mult
        if best_p["slm"] == "A":
            for am in np.arange(0.5, 4.1, 0.25):
                p = dict(best_p); p["am"] = round(am,2); m = simulate(ind, p); ft.append((p,m))
        else:
            for slb in [2,3,4,5,6,7,8,10,12]:
                key = f"swl_{slb}"
                if key not in ind:
                    ind[f"swl_{slb}"] = pd.Series(ind["low"]).rolling(slb, min_periods=slb).min().values
                    ind[f"swh_{slb}"] = pd.Series(ind["high"]).rolling(slb, min_periods=slb).max().values
                p = dict(best_p); p["slb"] = slb; m = simulate(ind, p); ft.append((p,m))

        # Fine-tune TP ratios
        for t1 in np.arange(0.5, 2.1, 0.25):
            for t3 in np.arange(2.0, 6.1, 0.5):
                t2 = (t1+t3)/2
                p = dict(best_p); p["t1"]=round(t1,2); p["t2"]=round(t2,2); p["t3"]=round(t3,2)
                m = simulate(ind, p); ft.append((p,m))

        ft_v = [(p,m) for p,m in ft if m["n"]>=15 and m["pf"]>=1.0 and m["dd"]>-60]
        ft_v.sort(key=lambda x: x[1]["sc"], reverse=True)
        print(f"  Tested {len(ft)} fine-tune variants, {len(ft_v)} valid")

        print(f"\n  TOP 15 — Fine-tuned:")
        print(f"  {'#':>3s} | {'EMA':>13s} | {'RSI':>5s} | {'SL':>7s} | {'TP':>9s} | {'VF':>4s}|{'ADX':>3s}|{'TE':>2s}|{'CD':>2s}| {'N':>3s} | {'WR':>5s} | {'PF':>5s} | {'Total':>8s} | {'DD':>7s} | {'$1000->':>9s} | {'Calmar':>7s} | {'Score':>7s}")
        print(f"  {'-'*150}")
        for i,(p,m) in enumerate(ft_v[:15],1):
            ema,rsi,sl,tp = fmt_params(p)
            vf = str(p.get('vf','-')) if p.get('vf') else '  -'
            admn = str(p.get('admn','-')) if p.get('admn') else ' -'
            te = 'Y' if p.get('te') else 'N'
            cd = str(p.get('cd',0))
            print(f"  {i:3d} | {ema:>13s} | {rsi:>5s} | {sl:>7s} | {tp:>9s} | {vf:>4s}|{admn:>3s}|{te:>2s}|{cd:>2s}| {m['n']:3d} | {m['wr']:5.1f}% | {m['pf']:5.2f} | {m['tot']:+8.1f}% | {m['dd']:+7.1f}% | ${m['bal']:>9.2f} | {m['cal']:7.2f} | {m['sc']:7.2f}")

        overall_best_p, overall_best_m = ft_v[0] if ft_v else v2[0]
    else:
        overall_best_p, overall_best_m = valid[0]

    # ========== FINAL ==========
    baseline = {"ef":20,"es":50,"et":200,"rb":60,"rbe":40,"slm":"S","am":1.5,"slb":5,"t1":1,"t2":2,"t3":3}
    bm = simulate(ind, baseline)

    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  [Baseline] Pine Script Original:")
    print(f"    EMA 20/50/200 | RSI 60/40 | SWING_5 | TP 1/2/3")
    print(f"    n={bm['n']} WR={bm['wr']:.1f}% PF={bm['pf']:.2f} Total={bm['tot']:+.1f}% DD={bm['dd']:+.1f}% ${bm['bal']:,.0f}")

    bp = overall_best_p; bm2 = overall_best_m
    ema,rsi,sl,tp = fmt_params(bp)
    print(f"\n  [BEST FOUND]:")
    print(f"    EMA {ema} | RSI {rsi} | {sl} | TP {tp}")
    extras = []
    if bp.get('vf'): extras.append(f"VolFilter={bp['vf']}")
    if bp.get('admn'): extras.append(f"ADX>={bp['admn']}")
    if bp.get('te'): extras.append("TrendExit=ON")
    if bp.get('cd',0)>0: extras.append(f"Cooldown={bp['cd']}")
    if extras: print(f"    Filters: {', '.join(extras)}")
    print(f"    n={bm2['n']} WR={bm2['wr']:.1f}% PF={bm2['pf']:.2f} Total={bm2['tot']:+.1f}% DD={bm2['dd']:+.1f}% ${bm2['bal']:,.0f}")

    print(f"\n  [Improvement]:")
    print(f"    PF:    {bm['pf']:.2f} -> {bm2['pf']:.2f} ({bm2['pf']-bm['pf']:+.2f})")
    print(f"    Total: {bm['tot']:+.1f}% -> {bm2['tot']:+.1f}% ({bm2['tot']-bm['tot']:+.1f}%)")
    print(f"    DD:    {bm['dd']:+.1f}% -> {bm2['dd']:+.1f}% ({bm2['dd']-bm['dd']:+.1f}%)")
    print(f"    $1K:   ${bm['bal']:,.0f} -> ${bm2['bal']:,.0f}")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
