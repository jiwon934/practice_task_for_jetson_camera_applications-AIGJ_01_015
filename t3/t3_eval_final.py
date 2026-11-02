import csv, numpy as np, bisect
G = [r for r in csv.DictReader(open("t3_snaps_1030b/gt_final.csv")) if r["gt"] != ""]
R = list(csv.DictReader(open("t3_occ_1030b_960t_contain.csv")))
ts = np.array([float(r["ts"]) for r in R]); raw = np.array([int(r["raw"]) for r in R])
def at(t, n, f):
    i = int(np.searchsorted(ts, t)); i = min(max(i, 0), len(ts) - 1)
    return f(raw[max(0, i - n + 1):i + 1])
med = lambda v: np.floor(np.median(v) + 0.5); p75 = lambda v: np.floor(np.percentile(v, 75) + 0.5)
g = np.array([int(r["gt"]) for r in G]); t = [float(r["ts"]) for r in G]
P = {"snap raw": np.array([int(r["raw"]) for r in G]), "logger count(10s med)": np.array([int(r["count"]) for r in G])}
for n in (10, 30):
    for nm, f in (("med", med), ("p75", p75)): P[f"{n}s {nm}"] = np.array([at(x, n, f) for x in t])
print("방식 | MAE | bias | 정확% | ±1% | 빈방 오검출%")
for k, p in P.items():
    e = p - g; em = p[g == 0]
    print(f"{k} | {np.abs(e).mean():.2f} | {e.mean():+.2f} | {np.mean(e == 0) * 100:.0f} | {np.mean(np.abs(e) <= 1) * 100:.0f} | {np.mean(em > 0) * 100:.0f} (n{len(em)})")
p = P["logger count(10s med)"]
for lo, hi, nm in ((0, 0, "0명"), (1, 3, "1–3명"), (4, 7, "4–7명"), (8, 99, "8명+")):
    m = (g >= lo) & (g <= hi)
    if m.sum(): e = p[m] - g[m]; print(f"{nm}: n {m.sum()} MAE {np.abs(e).mean():.2f} bias {e.mean():+.2f}")
print("gt 분포", np.bincount(g).tolist())
