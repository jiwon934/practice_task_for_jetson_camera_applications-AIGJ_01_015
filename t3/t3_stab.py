import csv, numpy as np
R = list(csv.DictReader(open("t3_occ_1030_allday_960t.csv")))
ts = np.array([float(r["ts"]) for r in R]); raw = np.array([int(r["raw"]) for r in R]); mk = [r["mark"] for r in R]; cnt = np.array([int(r["count"]) for r in R])
def roll(x, n, f): return np.array([f(x[max(0, i - n + 1):i + 1]) for i in range(len(x))])
med = lambda v: np.floor(np.median(v) + 0.5); p75 = lambda v: np.floor(np.percentile(v, 75) + 0.5)
res = {(n, nm): roll(raw, n, f) for n in (3, 10, 30) for nm, f in (("median", med), ("p75", p75))}
for seg, truth in (("seg1", 2), ("seg1_gt2", 2)):
    idx = [i for i, m in enumerate(mk) if m == seg]
    print(f"{seg}: {R[idx[0]]['time']}~{R[idx[-1]]['time']} n {len(idx)} truth {truth} raw mean {raw[idx].mean():.2f} exact {np.mean(raw[idx] == truth) * 100:.0f}% | logger count(10s median) mean {cnt[idx].mean():.2f}")
    for n in (3, 10, 30):
        print("   " + " | ".join(f"{n}s {nm}: mean {res[(n, nm)][idx].mean():.2f} exact {np.mean(res[(n, nm)][idx] == truth) * 100:.0f}% MAE {np.abs(res[(n, nm)][idx] - truth).mean():.2f}" for nm in ("median", "p75")))
z = raw == 0; runs = []; i = 0
while i < len(z):
    if z[i]:
        j = i; last = i
        while j < len(z) and (z[j] or (j - last) <= 3):
            if z[j]: last = j
            j += 1
        if ts[last] - ts[i] >= 600: runs.append((R[i]["time"], R[last]["time"], round((ts[last] - ts[i]) / 60, 1), int((raw[i:last + 1] > 0).sum())))
        i = last + 1
    else: i += 1
print("raw 0 연속 10분 이상 (3초 이하 깜빡임 허용):", runs)
# looser: 60 s rolling max <= 1 for >= 10 min
m60 = roll(raw, 60, np.max); z = m60 <= 1; runs = []; i = 0
while i < len(z):
    if z[i]:
        j = i
        while j < len(z) and z[j]: j += 1
        if ts[j - 1] - ts[i] >= 600: runs.append((R[i]["time"], R[j - 1]["time"], round((ts[j - 1] - ts[i]) / 60, 1), round(float(raw[i:j].mean()), 2)))
        i = j
    else: i += 1
print("1분 최대 raw ≤ 1 이 10분 이상:", runs)
