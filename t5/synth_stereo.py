"""과제 5-1: 가상 스테레오 (정답 거리가 정해진 합성 장면).
1) 알려진 내부 파라미터(f, 주점)와 기준선 B, 오른쪽 카메라의 약간의 회전으로 두 카메라를 만든다.
2) 광선-평면 교차로 영상을 그린다: 체커보드(9x6 내부 코너, 25 mm) 25쌍 + 정면 평면 3개(1.5 / 2.5 / 4.0 m) + 배경(7 m).
3) 체커보드 코너 검출 → stereoCalibrate → stereoRectify → remap → SGBM → reprojectImageTo3D 로 깊이 → 평면별 중앙값과 정답 비교.
사용: python synth_stereo.py [--out DIR]"""
import argparse, json, os, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/eg2-lab/applab/runs/t5_synth"))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--zmax", type=float, default=1.1, help="체커보드 최대 거리(m)")
ap.add_argument("--joint", action="store_true", help="stereoCalibrate에서 내부 파라미터도 함께 다듬기")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)

W, H, F = 1280, 720, 1000.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
B = 0.100                                        # 기준선 100 mm (오른쪽 카메라가 +x)
R_lr = cv2.Rodrigues(np.deg2rad([0.3, -0.8, 0.2]))[0]   # 오른쪽 카메라의 작은 회전 (실제 장착 오차 흉내)
CAMS = {"L": (np.eye(3), np.zeros(3)), "R": (R_lr, np.array([B, 0, 0]))}   # 카메라→월드 회전, 카메라 중심 (왼쪽 카메라 좌표 = 월드)
NOISE = 2.0                                      # 화소 잡음 (0~255)

# 결정적 텍스처: 큰 무작위 패턴을 미리 만들어 평면 좌표(m)로 조회
TEX = cv2.GaussianBlur((rng.random((4096, 4096)) * 255).astype(np.float32), (0, 0), 1.2)
TEX = cv2.normalize(TEX, None, 20, 235, cv2.NORM_MINMAX)


def rays(cam):
    Rc, C = CAMS[cam]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u)], -1) @ Rc.T
    return C, d


def render(cam, planes):
    """planes: [(O, u, v, (a0,a1,b0,b1), texfn)] — 가장 가까운 교차 평면의 텍스처를 그림"""
    C, d = rays(cam)
    img = np.full((H, W), 128.0); best = np.full((H, W), np.inf)
    for O, u, v, ext, tex in planes:
        n = np.cross(u, v)
        den = d @ n
        s = ((O - C) @ n) / np.where(np.abs(den) < 1e-9, 1e-9, den)
        P = C + s[..., None] * d
        aa, bb = (P - O) @ u, (P - O) @ v
        ok = (s > 0) & (s < best) & (aa >= ext[0]) & (aa <= ext[1]) & (bb >= ext[2]) & (bb <= ext[3])
        img[ok] = tex(aa[ok], bb[ok]); best[ok] = s[ok]
    img += rng.normal(0, NOISE, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def random_tex(off):
    def f(aa, bb):
        x = ((aa * 900 + off[0]) % 4096).astype(int); y = ((bb * 900 + off[1]) % 4096).astype(int)
        return TEX[y, x]
    return f


SQ, NX, NY = 0.025, 9, 6                         # 칸 25 mm, 내부 코너 9x6 → 칸 10x7


def board_tex(aa, bb):
    i, j = np.floor(aa / SQ).astype(int), np.floor(bb / SQ).astype(int)
    inside = (i >= 0) & (i < NX + 1) & (j >= 0) & (j < NY + 1)
    return np.where(inside & ((i + j) % 2 == 0), 20.0, 235.0)


# ---- 1) 체커보드 25쌍 ----
objp = np.array([[(i + 1) * SQ, (j + 1) * SQ, 0] for j in range(NY) for i in range(NX)], np.float32)
bw, bh = (NX + 1) * SQ, (NY + 1) * SQ
obj_pts, img_l, img_r, used = [], [], [], 0
t0 = time.time()
for k in range(40):
    if used == 25: break
    Rb = cv2.Rodrigues(np.deg2rad(rng.uniform([-25, -25, -15], [25, 25, 15])))[0]
    Z = rng.uniform(0.55, a.zmax)
    center = np.array([rng.uniform(-0.2, 0.25) * Z, rng.uniform(-0.12, 0.12) * Z, Z])
    u, v = Rb[:, 0], Rb[:, 1]
    O = center - u * bw / 2 - v * bh / 2
    margin = [(O - u * 0.02 - v * 0.02, u, v, (0, bw + 0.04, 0, bh + 0.04), lambda aa, bb: np.full(aa.shape, 235.0))]
    planes = [(O, u, v, (0, bw, 0, bh), board_tex)] + margin + [(np.array([-5, -5, 3.0]), np.array([1., 0, 0]), np.array([0, 1., 0]), (0, 10, 0, 10), random_tex((0, 0)))]
    L, R = render("L", planes), render("R", planes)
    okL, cL = cv2.findChessboardCorners(L, (NX, NY))
    okR, cR = cv2.findChessboardCorners(R, (NX, NY))
    if not (okL and okR): continue
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    cL = cv2.cornerSubPix(L, cL, (7, 7), (-1, -1), crit); cR = cv2.cornerSubPix(R, cR, (7, 7), (-1, -1), crit)
    obj_pts.append(objp); img_l.append(cL); img_r.append(cR); used += 1
    if used == 1:
        cv2.imwrite(os.path.join(a.out, "board_L.png"), L); cv2.imwrite(os.path.join(a.out, "board_R.png"), R)
print(f"체커보드 {used}쌍 검출 ({time.time()-t0:.0f}s)", flush=True)

rmsL, KL, DL, _, _ = cv2.calibrateCamera(obj_pts, img_l, (W, H), None, None)
rmsR, KR, DR, _, _ = cv2.calibrateCamera(obj_pts, img_r, (W, H), None, None)
rms, KL, DL, KR, DR, Rs, Ts, E, Fm = cv2.stereoCalibrate(obj_pts, img_l, img_r, KL, DL, KR, DR, (W, H),
                                                          flags=cv2.CALIB_USE_INTRINSIC_GUESS if a.joint else cv2.CALIB_FIX_INTRINSIC)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(KL, DL, KR, DR, (W, H), Rs, Ts, alpha=0)
m1 = cv2.initUndistortRectifyMap(KL, DL, R1, P1, (W, H), cv2.CV_32FC1)
m2 = cv2.initUndistortRectifyMap(KR, DR, R2, P2, (W, H), cv2.CV_32FC1)
B_est = float(np.linalg.norm(Ts))
# 정류 후 같은 코너의 행 차이
rowdiff = []
for cL, cR in zip(img_l, img_r):
    pl = cv2.undistortPoints(cL, KL, DL, R=R1, P=P1); pr = cv2.undistortPoints(cR, KR, DR, R=R2, P=P2)
    rowdiff.append(np.abs(pl[:, 0, 1] - pr[:, 0, 1]))
rowdiff = np.concatenate(rowdiff)

# ---- 2) 정답 장면: 정면 평면 3개 + 배경 ----
truth = [1.5, 2.5, 4.0]
planes, boxes = [], []
for i, Z in enumerate(truth):
    x0 = (-0.55 + 0.37 * i) * Z                  # 화면을 세로 띠 3개로 나눠 각 거리 평면 배치
    planes.append((np.array([x0, -0.25 * Z, Z]), np.array([1., 0, 0]), np.array([0, 1., 0]), (0, 0.3 * Z, 0, 0.5 * Z), random_tex((500 * i, 900 * i))))
planes.append((np.array([-20, -20, 7.0]), np.array([1., 0, 0]), np.array([0, 1., 0]), (0, 40, 0, 40), random_tex((3000, 1000))))
L, R = render("L", planes), render("R", planes)
cv2.imwrite(os.path.join(a.out, "scene_L.png"), L); cv2.imwrite(os.path.join(a.out, "scene_R.png"), R)
Lr, Rr = cv2.remap(L, *m1, cv2.INTER_LINEAR), cv2.remap(R, *m2, cv2.INTER_LINEAR)
nd, bs = 128, 5
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
t1 = time.time(); disp = sgbm.compute(Lr, Rr).astype(np.float32) / 16.0; t_sgbm = time.time() - t1
xyz = cv2.reprojectImageTo3D(disp, Q)
vis = cv2.normalize(np.clip(disp, 0, None), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(os.path.join(a.out, "disparity.png"), cv2.applyColorMap(vis, cv2.COLORMAP_JET))

# 평면별 영역: 왼쪽(원본) 카메라에서 평면 중앙부를 투영한 뒤 정류 좌표로 옮김
res = []
for i, Z in enumerate(truth):
    O, u, v, ext, _ = planes[i]
    c3 = O + u * ext[1] * 0.5 + v * ext[3] * 0.5
    pts = np.array([c3 + u * dx + v * dy for dx in np.linspace(-0.3, 0.3, 7) * ext[1] for dy in np.linspace(-0.3, 0.3, 7) * ext[3]])
    pix, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, None)
    rp = cv2.undistortPoints(pix.astype(np.float32), KL, DL, R=R1, P=P1)[:, 0]
    zs = [xyz[int(y), int(x), 2] for x, y in rp if 0 <= int(x) < W and 0 <= int(y) < H and disp[int(y), int(x)] > 0]
    zm = float(np.median(zs)) if zs else float("nan")
    d_exp = P1[0, 0] * B_est / Z
    res.append({"truth_m": Z, "est_m": round(zm, 4), "err_pct": round(100 * (zm - Z) / Z, 2), "valid_pts": len(zs),
                "disparity_px": round(d_exp, 1), "z_res_per_px_m": round(Z * Z / (P1[0, 0] * B_est), 4)})

out = {"zmax": a.zmax, "joint": a.joint, "P1_f": round(float(P1[0, 0]), 1), "image": [W, H], "f_true": F, "B_true_m": B, "pairs": used, "rms_stereo_px": round(rms, 4),
       "rms_left_px": round(rmsL, 4), "rms_right_px": round(rmsR, 4), "f_est": [round(KL[0, 0], 1), round(KR[0, 0], 1)],
       "B_est_m": round(B_est, 5), "B_err_mm": round(1000 * (B_est - B), 2),
       "rect_row_diff_px": {"mean": round(float(rowdiff.mean()), 3), "max": round(float(rowdiff.max()), 3)},
       "sgbm": {"numDisparities": nd, "blockSize": bs, "ms": round(1000 * t_sgbm)}, "results": res}
json.dump(out, open(os.path.join(a.out, "result.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
