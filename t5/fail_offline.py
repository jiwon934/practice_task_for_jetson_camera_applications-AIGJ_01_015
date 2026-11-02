"""과제 5-5 대책 시험: /sample로 저장한 원본 두 장(…_L.png, …_R.png)에 SGBM 설정을 바꿔 다시 돌리고 ROI 값을 비교.
사용: ~/eg2/bin/python fail_offline.py runs/t5_fail/close_134900 [--roi "x0 y0 x1 y1"(원본 1280x720 기준)] [--set scale=1,zmin=0.35 ...]"""
import argparse, math, os
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("--calib", default=os.path.expanduser("~/eg2-lab/applab/runs/t5_calib2/stereo_25mm.npz"))
ap.add_argument("--roi", default="", help="원본 해상도 기준 x0 y0 x1 y1 (기본: 가운데 8%)")
ap.add_argument("--set", nargs="*", default=["scale=0.5,zmin=0.8,zmax=4.5,block=5", "scale=0.5,zmin=0.35,zmax=4.5,block=5",
                                              "scale=1,zmin=0.35,zmax=4.5,block=5", "scale=0.5,zmin=0.8,zmax=4.5,block=11"])
a = ap.parse_args()
C = np.load(a.calib); W, H = 1280, 720
IL, IR = cv2.imread(a.base + "_L.png", 0), cv2.imread(a.base + "_R.png", 0)
if a.roi: r0 = list(map(float, a.roi.split()))
else: s = 0.08 * W; r0 = [W / 2 - s / 2, H / 2 - s / 2, W / 2 + s / 2, H / 2 + s / 2]
for st in a.set:
    o = dict(kv.split("=") for kv in st.split(",")); sc, zmin, zmax, bs = float(o["scale"]), float(o["zmin"]), float(o["zmax"]), int(o["block"])
    w, h = int(W * sc), int(H * sc); S = np.diag([sc, sc, 1.0]); P1, P2 = S @ C["P1"], S @ C["P2"]
    mL = cv2.initUndistortRectifyMap(C["KL"], C["DL"], C["R1"], P1, (w, h), cv2.CV_16SC2)
    mR = cv2.initUndistortRectifyMap(C["KR"], C["DR"], C["R2"], P2, (w, h), cv2.CV_16SC2)
    FB = P1[0, 0] * abs(P2[0, 3] / P2[0, 0])
    mn = max(0, int(math.floor(FB / zmax)) - 2); nd = int(math.ceil((FB / zmin - mn + 2) / 16.0)) * 16
    sg = cv2.StereoSGBM_create(minDisparity=mn, numDisparities=nd, blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1,
                               uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    gl, gr = cv2.remap(IL, *mL, cv2.INTER_LINEAR), cv2.remap(IR, *mR, cv2.INTER_LINEAR)
    d = sg.compute(gl, gr).astype(np.float32) / 16
    x0, y0, x1, y1 = [int(v * sc) for v in r0]; dr = d[y0:y1, x0:x1]; ok = dr > mn
    z = FB / dr[ok] if ok.sum() else np.array([])
    note = f"왼쪽 {mn + nd}px 열은 시차 계산 불가" if x0 < mn + nd else ""
    print(f"{st:40s} 시차 {mn}~{mn + nd} → Z {FB / (mn + nd):.2f}~{FB / max(mn, 1):.1f} m | 유효 {ok.mean() * 100:5.1f}% | "
          + (f"Z 중앙값 {np.median(z):.3f} m, IQR {np.subtract(*np.percentile(z, [75, 25])) * 100:.1f} cm" if len(z) else "깊이 없음") + f" {note}")
    dv = np.clip((d - mn) / nd * 255, 0, 255).astype(np.uint8); dc = cv2.applyColorMap(dv, cv2.COLORMAP_JET); dc[d <= mn] = 0
    cv2.rectangle(dc, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.imwrite(a.base + "_off_" + st.replace(",", "_").replace("=", "") + ".jpg", dc)
