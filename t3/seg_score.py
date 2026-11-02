"""과제 3: 구간(mark) 이름에 든 숫자를 실제 인원으로 보고, 그 구간 동안 1초 기록의 raw·창별 보정값 평균과 오차를 계산.
사용: python3 seg_score.py [로그 CSV] (생략하면 runs/t3_occ_*.csv 중 최신)"""
import csv, glob, os, re, sys, collections
f = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(os.path.expanduser("~/eg2-lab/applab/runs/t3_occ_*.csv")), key=os.path.getmtime)
rows = list(csv.DictReader(open(f)))
seg = collections.OrderedDict()
for r in rows:
    m = r.get("mark", "")
    if m: seg.setdefault(m, []).append(r)
wc = [c for c in rows[0] if c.startswith("count_w")] if rows else []
print(os.path.basename(f))
for m, rs in seg.items():
    g = re.search(r"(\d+)", m); gt = int(g.group(1)) if g else None
    def mean(c): return sum(float(r[c]) for r in rs if r[c] not in ("", "None")) / len(rs)
    raw = mean("raw"); vals = {c: mean(c) for c in ["count"] + wc}
    exact = sum(int(r["count"]) == gt for r in rs) / len(rs) if gt is not None else None
    print(f"{m}: {len(rs)}s 정답 {gt} | raw {raw:.2f} | " + " ".join(f"{c} {v:.2f}" for c, v in vals.items())
          + (f" | 오차(count) {vals['count']-gt:+.2f} 정확 {100*exact:.0f}%" if gt is not None else ""))
