"""과제 5-3: 촬영한 체커보드 쌍(corners.json)으로 스테레오 보정 → 재투영 오차, 기준선, 정류 후 행 차이, 보정 파일 저장.
사용: python calib_stereo.py --square-mm 25 [--dir runs/t5_calib] [--baseline-mm 140]"""
import argparse, json, os
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/eg2-lab/applab/runs/t5_calib"))
ap.add_argument("--square-mm", type=float, required=True)
ap.add_argument("--baseline-mm", type=float, default=140)
ap.add_argument("--drop-px", type=float, default=0, help="쌍별 재투영 오차가 이 값보다 크면 빼고 다시 보정 (0 = 안 뺌)")
a = ap.parse_args()
d = json.load(open(os.path.join(a.dir, "corners.json")))
cols, rows = d["pattern"]; W, H = 1280, 720
objp = np.array([[i * a.square_mm / 1000, j * a.square_mm / 1000, 0] for j in range(rows) for i in range(cols)], np.float32)
O = [objp] * len(d["pairs"])
def norm(c):
    # 체커보드 검출이 영상마다 코너 순서를 180도 뒤집어 줄 수 있음 → 첫 코너가 왼쪽 위가 되게 맞춤 (왼쪽·오른쪽 짝이 어긋나지 않게)
    c = np.array(c, np.float32)
    if c[0, 0] + c[0, 1] > c[-1, 0] + c[-1, 1]:
        c = c[::-1].copy()
    return c.reshape(-1, 1, 2)
IL = [norm(p["L"]) for p in d["pairs"]]
IR = [norm(p["R"]) for p in d["pairs"]]
K0 = np.array([[950., 0, W / 2], [0, 950., H / 2], [0, 0, 1]])
FL = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
rl, KL, DL, rvl, tvl = cv2.calibrateCamera(O, IL, (W, H), K0.copy(), None, flags=FL)
rr, KR, DR, _, _ = cv2.calibrateCamera(O, IR, (W, H), K0.copy(), None, flags=FL)
res = cv2.stereoCalibrateExtended(O, IL, IR, KL, DL, KR, DR, (W, H), None, None,
                                                                     flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)
rms, KL, DL, KR, DR, Rm, T, E, F = res[:9]; per = res[-1]
dropped = []
if a.drop_px:
    keep = [i for i in range(len(O)) if per[i].max() <= a.drop_px]
    dropped = [d["pairs"][i]["pair"] for i in range(len(O)) if i not in keep]
    O, IL, IR = [O[i] for i in keep], [IL[i] for i in keep], [IR[i] for i in keep]
    rl, KL, DL, rvl, tvl = cv2.calibrateCamera(O, IL, (W, H), K0.copy(), None, flags=FL)
    rr, KR, DR, _, _ = cv2.calibrateCamera(O, IR, (W, H), K0.copy(), None, flags=FL)
    res = cv2.stereoCalibrateExtended(O, IL, IR, KL, DL, KR, DR, (W, H), None, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)
    rms, KL, DL, KR, DR, Rm, T, E, F = res[:9]; per = res[-1]
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(KL, DL, KR, DR, (W, H), Rm, T, alpha=0)
rd = []
for cl, cr in zip(IL, IR):
    pl = cv2.undistortPoints(cl, KL, DL, R=R1, P=P1); pr = cv2.undistortPoints(cr, KR, DR, R=R2, P=P2)
    rd.append(np.abs(pl[:, 0, 1] - pr[:, 0, 1]))
rd = np.concatenate(rd)
dist = [float(np.linalg.norm(t)) for t in tvl]
B = float(np.linalg.norm(T)) * 1000
ang = np.rad2deg(cv2.Rodrigues(Rm)[0].ravel())
np.savez(os.path.join(a.dir, f"stereo_{a.square_mm:g}mm.npz"), KL=KL, DL=DL, KR=KR, DR=DR, R=Rm, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
out = {"square_mm": a.square_mm, "pairs": len(O), "dropped": dropped, "rms_left_px": round(rl, 3), "rms_right_px": round(rr, 3), "rms_stereo_px": round(rms, 3),
       "per_pair_max_px": round(float(per.max()), 3), "f_px": [round(KL[0, 0], 1), round(KR[0, 0], 1)], "rect_f_px": round(float(P1[0, 0]), 1),
       "baseline_mm": round(B, 1), "baseline_err_mm": round(B - a.baseline_mm, 1), "R_deg(x,y,z)": ang.round(2).tolist(),
       "rect_row_diff_px": {"mean": round(float(rd.mean()), 3), "p95": round(float(np.percentile(rd, 95)), 3), "max": round(float(rd.max()), 3)},
       "board_dist_m": {"min": round(min(dist), 2), "median": round(float(np.median(dist)), 2), "max": round(max(dist), 2)}}
json.dump(out, open(os.path.join(a.dir, f"calib_{a.square_mm:g}mm.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
