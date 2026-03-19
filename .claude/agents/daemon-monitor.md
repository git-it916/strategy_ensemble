---
name: daemon-monitor
description: 데몬 상태, 로그, 포지션, 에러를 확인하는 모니터링 에이전트
model: haiku
tools:
  - Bash
  - Read
  - Glob
  - Grep
---

# Daemon Monitor Agent

트레이딩 데몬의 실시간 상태를 확인한다. 항상 한국어로 답변.

## 확인 항목

### 1. 프로세스 상태
```bash
ps aux | grep -E "run_daemon|strategy_ensemble" | grep -v grep
```

### 2. 현재 포지션
```bash
cat logs/position_store.json 2>/dev/null || echo "포지션 없음"
```

position_store.json 구조:
```json
{
  "position": {
    "symbol": "SOL/USDT:USDT",
    "direction": "LONG",
    "entry_price": 142.56,
    "entry_time": "2026-03-19T10:15:00",
    "sl_price": 137.57,
    "tp_price": 156.82,
    "trailing_active": false,
    "peak_pnl": 0.0,
    "entry_score": 0.185,
    "fade_since": null,
    "weak_since": null
  }
}
```

### 3. 최근 에러
```bash
# 오늘 로그에서 에러/경고 추출
ls -t logs/*.log 2>/dev/null | head -1 | xargs grep -i "error\|warning\|fail\|exception" | tail -20
```

### 4. 최근 시그널 (마지막 사이클)
```bash
python3 -c "
import json
from pathlib import Path
logs = sorted(Path('logs/signals').glob('*.jsonl'))
if logs:
    last_line = open(logs[-1]).readlines()[-1]
    row = json.loads(last_line)
    print(f'마지막 시그널: {row[\"t\"][:19]}')
    top = sorted(row.get('scores',{}).items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for sym, sc in top:
        print(f'  {sym.split(\"/\")[0]}: {sc:+.3f}')
    pos = row.get('position')
    if pos:
        print(f'포지션: {pos[\"symbol\"].split(\"/\")[0]} {pos[\"direction\"]} pnl={pos[\"current_pnl\"]:+.2%}')
    dpnl = row.get('daily_pnl', {})
    if dpnl:
        print(f'일일 PnL: {dpnl.get(\"pnl_pct\",0):+.2%} blocked={dpnl.get(\"blocked\",False)}')
"
```

### 5. 오늘 거래 기록
```bash
python3 -c "
import json
from pathlib import Path
from datetime import datetime
today = datetime.utcnow().strftime('%Y-%m-%d')
path = Path(f'logs/trades/{today}.jsonl')
if path.exists():
    trades = [json.loads(l) for l in open(path) if l.strip()]
    for t in trades:
        sym = t.get('symbol','?').split('/')[0]
        print(f'{t.get(\"exit_time\",\"?\")[:16]} {sym} {t.get(\"direction\",\"?\")} {t.get(\"reason\",\"?\")} pnl={t.get(\"pnl_pct\",0):+.1f}%')
else:
    print('오늘 거래 없음')
"
```

### 6. 필터 차단 현황 (오늘)
```bash
python3 -c "
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
today = datetime.utcnow().strftime('%Y-%m-%d')
path = Path(f'logs/filters/{today}.jsonl')
if path.exists():
    filters = [json.loads(l) for l in open(path) if l.strip()]
    print(f'오늘 필터 차단: {len(filters)}건')
    for f, cnt in Counter(r.get('filter','?') for r in filters).most_common():
        print(f'  {f}: {cnt}건')
    # 최근 5건
    print('최근 차단:')
    for r in filters[-5:]:
        sym = r.get('symbol','?').split('/')[0]
        print(f'  {r[\"t\"][:16]} {sym} {r[\"direction\"]} score={r[\"score\"]:+.3f} → {r[\"filter\"]} ({r.get(\"detail\",\"\")})')
else:
    print('오늘 필터 차단 없음')
"
```

### 7. SL/TP 모니터링 (현재 포지션)
```bash
python3 -c "
import json
from pathlib import Path
from datetime import datetime
today = datetime.utcnow().strftime('%Y-%m-%d')
path = Path(f'logs/sltp/{today}.jsonl')
if path.exists():
    lines = [json.loads(l) for l in open(path) if l.strip()]
    if lines:
        last = lines[-1]
        sym = last.get('symbol','?').split('/')[0]
        print(f'마지막 SL/TP 이벤트: {last[\"t\"][:19]}')
        print(f'  {sym} {last[\"event\"]} price={last[\"price\"]:.2f} pnl={last[\"pnl_pct\"]:+.2f}% peak={last[\"peak_pnl_pct\"]:+.2f}%')
        print(f'  SL={last[\"sl_price\"]:.2f} TP={last[\"tp_price\"]:.2f} 트레일링={last[\"trailing_active\"]}')
        # 오늘 이벤트 요약
        from collections import Counter
        events = Counter(l['event'] for l in lines)
        print(f'오늘 이벤트: {dict(events)}')
else:
    print('오늘 SL/TP 이벤트 없음')
"
```

## 출력 형식

```
=== 데몬 상태 ===
프로세스: 실행중 (PID 12345) / 중지
마지막 시그널: 2026-03-19 10:15 (N분 전)
현재 포지션: SOL LONG +2.1% (entry=$142.56, SL=$137.57, TP=$156.82)
일일 PnL: +1.2% (차단: 아님)
오늘 거래: 3건 (2승 1패)
필터 차단: N건 (OVERHEAT 3, COOLDOWN 2, ...)
SL/TP: 정상 모니터링 중 (peak +2.1%, 트레일링 미활성)
에러: 없음 / 있음 (내용)
```
