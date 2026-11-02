# 접근 구간 찾기: 같은 트랙에서 6초 안에 d가 1.5m 이상에서 0.8m 이하로 줄어든 구간 = 접근, 그 ±2초 안 TTC 알람 여부
import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1])))
alarms = [(float(r["t"]), r["id"].lstrip("#"), r["label"]) for r in rows if r["type"] == "event" and "충돌" in r["label"]]
T = collections.defaultdict(list)
for r in rows:
    if r["type"] == "person" and r["d"]:
        T[r["id"]].append((float(r["t"]), float(r["d"]), float(r["closing"] or 0), r["kind"], r["time"]))
eps = []
for k, v in T.items():
    i = 0
    while i < len(v):
        t, d, c, kind, tm = v[i]
        if d <= 0.8:
            far = [x for x in v if t - 6 <= x[0] < t and x[1] >= 1.5]
            if far:
                t0 = far[0][0]
                hit = [a for a in alarms if a[1] == k and t0 - 1 <= a[0] <= t + 2]
                eps.append((far[0][4], tm, k, far[0][1], d, max(x[2] for x in v if t0 <= x[0] <= t), hit))
                while i < len(v) and v[i][0] < t + 4: i += 1
                continue
        i += 1
for e in eps:
    print(f"approach #{e[2]} {e[0]}->{e[1]} d {e[3]:.2f}->{e[4]:.2f} max closing {e[5]:.2f}m/s alarm: {[a[2][:30] for a in e[6]] or 'NONE'}")
for a in alarms:
    used = any(a in e[6] for e in eps)
    if not used: print("alarm outside detected approach:", a[1], a[2][:40])
