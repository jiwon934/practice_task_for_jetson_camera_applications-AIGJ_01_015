"""과제 3 평가. 1) sheet: 스냅샷 목록(snaps.csv)으로 정답 기입용 시트(gt 칸 비어 있음)를 만듦.
2) score: 사람이 gt를 채운 시트로 MAE, 평균 오차(bias = 예측 - 정답), 정확 일치율, ±1 이내 비율, 빈 방 오검출률(gt 0인데 예측 > 0)을 계산.
사용: python eval_occ.py sheet [--snap-dir DIR] [--out gt.csv]   /   python eval_occ.py score gt.csv [--col count|raw]"""
import argparse, csv, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["sheet", "score"])
ap.add_argument("sheet_file", nargs="?")
ap.add_argument("--snap-dir", default=os.path.expanduser("~/eg2-lab/applab/runs/t3_snaps"))
ap.add_argument("--out", default=None)
ap.add_argument("--col", default="count", choices=["count", "raw"], help="평가할 예측 값 (보정 count 또는 원시 raw)")
a = ap.parse_args()

if a.cmd == "sheet":
    out = a.out or os.path.join(a.snap_dir, "gt.csv")
    rows = list(csv.reader(open(os.path.join(a.snap_dir, "snaps.csv"))))
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "ts", "raw", "count", "gt", "note"])
        for ts, fn, raw, cnt in rows:
            w.writerow([fn, ts, raw, cnt, "", ""])
    print(f"{len(rows)}장 → {out}  (gt 칸에 사진 속 실제 인원을 적은 뒤 score 실행)")
    sys.exit()

rows = [r for r in csv.DictReader(open(a.sheet_file)) if r["gt"].strip() != ""]
if not rows:
    sys.exit("gt가 채워진 줄이 없음")
P = [int(r[a.col]) for r in rows]; G = [int(r["gt"]) for r in rows]
E = [p - g for p, g in zip(P, G)]; n = len(rows)
empty = [p for p, g in zip(P, G) if g == 0]
res = {"n": n, "pred": a.col, "MAE": round(sum(abs(e) for e in E) / n, 3), "bias": round(sum(E) / n, 3),
       "exact_%": round(100 * sum(e == 0 for e in E) / n, 1), "within1_%": round(100 * sum(abs(e) <= 1 for e in E) / n, 1),
       "empty_frames": len(empty), "empty_false_%": round(100 * sum(p > 0 for p in empty) / len(empty), 1) if empty else None}
print(res)
worst = sorted(zip(E, rows), key=lambda x: -abs(x[0]))[:5]
print("오차 큰 사진:", [(r["file"], f"pred {r[a.col]} gt {r['gt']}") for e, r in worst if e])
