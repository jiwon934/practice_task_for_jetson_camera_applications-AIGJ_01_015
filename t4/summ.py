"""폴링 기록(t4_events_poll.jsonl)에서 '요약' 이벤트를 중복 없이 시각순으로 출력. 사용: python3 summ.py [시작 HH:MM:SS]"""
import json, sys, datetime
since = sys.argv[1] if len(sys.argv) > 1 else "00:00:00"
seen = {}
for line in open("/home/jiwon934/eg2-lab/applab/runs/t4_events_poll.jsonl"):
    line = line.strip()
    if not line: continue
    try: d = json.loads(line)
    except Exception: continue
    for e in d.get("events", []):
        if "요약" in e["label"]:
            seen[(round(e["ts"], 1), e["who"])] = e["label"]
for (ts, who), lab in sorted(seen.items()):
    t = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    if t >= since: print(t, who, lab)
