import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1])))
p = [r for r in rows if r["type"] == "person"]
print("span", p[0]["time"], "-", rows[-1]["time"], "fps", sorted({r["fps"] for r in p})[:3], "...")
for r in rows:
    if r["type"] == "event" and any(s in r["label"] for s in ("충돌", "다시 찾음", "등장", "퇴장")):
        print(" ", r["time"], r["id"], r["label"])
T = collections.defaultdict(list)
for r in p:
    T[r["id"]].append(r)
for k, v in T.items():
    ds = [float(r["d"]) for r in v if r["d"]]
    tt = [float(r["ttc"]) for r in v if r["ttc"]]
    cl = [float(r["closing"]) for r in v if r["closing"]]
    hs = [float(r["h"]) for r in v]
    kinds = dict(collections.Counter(r["kind"] for r in v))
    print(f"#{k} {v[0]['time']}-{v[-1]['time']} n={len(v)} d {min(ds):.2f}-{max(ds):.2f} maxclose {max(cl):.2f} minTTC {min(tt) if tt else '-'} h {min(hs):.2f}-{max(hs):.2f} {kinds}")
