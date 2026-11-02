# 회차 표시(10-8_trials.csv) 사이 구간마다: 충돌 위험 알람 수·TTC, 최소 거리, 최대 접근 속도
import csv, sys
log = list(csv.DictReader(open(sys.argv[1])))
tr = [l.strip().split(",") for l in open(sys.argv[2]) if l.strip()]
end = "10:07:00"
for i, (t, kind, n) in enumerate(tr):
    t1 = tr[i + 1][0] if i + 1 < len(tr) else end
    ev = [r for r in log if r["type"] == "event" and t <= r["time"] < t1]
    ttc = [r["label"] for r in ev if "충돌" in r["label"]]
    near = [r["label"] for r in ev if r["label"].startswith("알람: 근접")]
    ps = [r for r in log if r["type"] == "person" and t <= r["time"] < t1 and r["d"] and float(r["h"]) >= 1.1]
    dmin = min((float(r["d"]) for r in ps), default=None)
    cmax = max((float(r["closing"] or 0) for r in ps), default=None)
    kinds = sorted({r["kind"] for r in ps})
    reid = sum(1 for r in ev if "다시 찾음" in r["label"])
    ok = (len(ttc) >= 1) if kind == "충돌" else (len(ttc) == 0)
    print(f"{kind}{n} {t}-{t1} TTC알람 {len(ttc)} {[x.split('(')[0].replace('알람: 충돌 위험 ','') + x[x.find('('):x.find(',')] for x in ttc]} 근접알람 {len(near)} 최소거리 {dmin} 최대접근 {cmax} {kinds} 재식별 {reid} -> {'맞음' if ok else '틀림'}")
