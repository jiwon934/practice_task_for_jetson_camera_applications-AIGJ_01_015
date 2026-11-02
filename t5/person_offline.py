"""과제 5-6 오프라인: 저장한 원본 쌍(…_L.png/_R.png)에 person_dist.py와 같은 방식(YOLO → 몸통 영역 → 무늬 게이트 → 시차 중앙값)을 적용.
사용: ~/eg2/bin/python person_offline.py runs/t5_person/raw/person3tile [--true 1.35]"""
import argparse, glob, math, os
import cv2, numpy as np
from ultralytics import YOLO
ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--true", type=float, default=0)
ap.add_argument("--calib", default=os.path.expanduser("~/eg2-lab/applab/runs/t5_calib2/stereo_25mm.npz"))
ap.add_argument("--engine", default=os.path.expanduser("~/eg2-lab/applab/t2/yolo11n_fp16.engine"))
ap.add_argument("--zmin", type=float, default=0.8); ap.add_argument("--zmax", type=float, default=6.0)
a = ap.parse_args()
C = np.load(a.calib); W, H = 1280, 720; s = 0.5; w, h = 640, 360; S = np.diag([s, s, 1.0]); P1, P2 = S @ C["P1"], S @ C["P2"]
mL = cv2.initUndistortRectifyMap(C["KL"], C["DL"], C["R1"], P1, (w, h), cv2.CV_16SC2); mR = cv2.initUndistortRectifyMap(C["KR"], C["DR"], C["R2"], P2, (w, h), cv2.CV_16SC2)
mLf = cv2.initUndistortRectifyMap(C["KL"], C["DL"], C["R1"], C["P1"], (W, H), cv2.CV_16SC2)
FB = P1[0, 0] * abs(P2[0, 3] / P2[0, 0]); mn = max(0, int(FB / a.zmax) - 2); nd = int(math.ceil((FB / a.zmin - mn + 2) / 16)) * 16
sg = cv2.StereoSGBM_create(minDisparity=mn, numDisparities=nd, blockSize=5, P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
det = YOLO(a.engine, task="detect"); allz = []
for fl in sorted(glob.glob(a.prefix + "*_L.png")):
    L, R = cv2.imread(fl), cv2.imread(fl.replace("_L.png", "_R.png"), 0)
    gl = cv2.remap(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), *mL, cv2.INTER_LINEAR); gr = cv2.remap(R, *mR, cv2.INTER_LINEAR)
    d = sg.compute(gl, gr).astype(np.float32) / 16; g = gl.astype(np.float32)
    good = (d > mn) & (np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2, 0)) > 4)
    r = det.predict(cv2.remap(L, *mLf, cv2.INTER_LINEAR), classes=[0], conf=0.4, imgsz=640, device=0, verbose=False)[0]
    res = []
    for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        x1, y1, x2, y2 = x1 * s, y1 * s, x2 * s, y2 * s; bw, bh = x2 - x1, y2 - y1
        tx0, ty0, tx1, ty1 = int(x1 + .3 * bw), int(y1 + .2 * bh), int(x2 - .3 * bw), int(y1 + .55 * bh)
        m = good[ty0:ty1, tx0:tx1]; v = m.mean() if m.size else 0
        z = float(np.median(FB / d[ty0:ty1, tx0:tx1][m])) if v >= 0.3 and m.sum() >= 15 else None
        res.append(f"상자 x{int(x1 / s)}-{int(x2 / s)} h{int(bh / s)} conf {cf:.2f} 유효 {v * 100:.0f}% Z {'측정불가' if z is None else f'{z:.3f}'}")
        if z is not None: allz.append((x1, z))
    print(os.path.basename(fl), " | ".join(res) or "사람 없음")
if allz:
    z = np.array([q[1] for q in allz]); print(f"전체 중앙값 {np.median(z):.3f} m, sd {z.std() * 100:.1f} cm, n {len(z)}" + (f", 오차 {(np.median(z) - a.true) / a.true * 100:+.1f}%" if a.true else ""))
