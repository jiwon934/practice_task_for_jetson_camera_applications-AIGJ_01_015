import csv, numpy as np
R = list(csv.DictReader(open("t3_occ_1030_allday_960t.csv")))
raw = np.array([int(r["raw"]) for r in R]); T = [r["time"] for r in R]
def roll(x, n, f): return np.array([f(x[max(0, i - n + 1):i + 1]) for i in range(len(x))])
med = lambda v: np.floor(np.median(v) + 0.5); p75 = lambda v: np.floor(np.percentile(v, 75) + 0.5)
idx = [i for i, t in enumerate(T) if "10:10:20" <= t <= "10:11:00"]
print("10:10:20~10:11:00 n", len(idx), "raw", raw[idx].tolist())
for n in (3, 10, 30):
    print("   " + " | ".join(f"{n}s {nm}: mean {y[idx].mean():.2f} exact {np.mean(y[idx] == 3) * 100:.0f}% MAE {np.abs(y[idx] - 3).mean():.2f}" for nm, f in (("median", med), ("p75", p75)) for y in [roll(raw, n, f)]))
print(f"   raw: mean {raw[idx].mean():.2f} exact {np.mean(raw[idx] == 3) * 100:.0f}%")
