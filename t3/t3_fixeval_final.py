"""40장 정답으로, 타일 경계 중복 대책을 다시 검출해 비교: (a) 지금 방식 (NMS IoU 0.5), (b) + 포함 억제(작은 상자의 60% 이상이 더 큰 상자 안이면 작은 것 제거), (c) + 타일 안쪽 경계에 잘린 상자 제거"""
import csv, json, os
import cv2, numpy as np, torch, torchvision, sys
CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
from ultralytics import YOLO
A = os.path.expanduser("~/eg2-lab/applab")
det = YOLO(os.path.join(A, "t3/yolo11n_fp16_960.engine"), task="detect")
Z = json.load(open(os.path.join(A, "t3/zones/room.json")))
G = list(csv.DictReader(open(os.path.join(A, "runs/t3_snaps_1030b/gt_final.csv"))))
def inside(poly, p, w, h): return cv2.pointPolygonTest((np.array(poly) * (w, h)).astype(np.int32), (float(p[0]), float(p[1])), False) >= 0
def ios(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
out = {"a": [], "b": [], "c": []}; g = []
for r in G:
    f = cv2.imread(os.path.join(A, "runs/t3_snaps_1030b", r["file"])); h, w = f.shape[:2]
    tw, th = w / 2, h / 2; ox, oy = tw * 0.15, th * 0.15
    T = [(int(max(0, c * tw - ox)), int(max(0, rr * th - oy)), int(min(w, (c + 1) * tw + ox)), int(min(h, (rr + 1) * th + oy))) for rr in range(2) for c in range(2)]
    res = det.predict([f[y0:y1, x0:x1] for x0, y0, x1, y1 in T], classes=[0], conf=CONF, imgsz=960, device=0, verbose=False)
    B, S, cut = [], [], []
    for (x0, y0, x1, y1), q in zip(T, res):
        for (a1, b1, a2, b2), s in zip(q.boxes.xyxy.tolist(), q.boxes.conf.tolist()):
            bb = [a1 + x0, b1 + y0, a2 + x0, b2 + y0]; B.append(bb); S.append(s)
            e = 3  # 타일 안쪽 경계(이미지 가장자리가 아닌 쪽)에 닿은 상자
            cut.append((x0 > 0 and bb[0] - x0 < e) or (x1 < w and x1 - bb[2] < e) or (y0 > 0 and bb[1] - y0 < e) or (y1 < h and y1 - bb[3] < e))
    keep = torchvision.ops.nms(torch.tensor(B), torch.tensor(S), 0.5).tolist() if B else []
    cnt = lambda L: sum(1 for i in L if any(inside(p, ((B[i][0] + B[i][2]) / 2, B[i][3]), w, h) for p in Z["include"]) and not any(inside(p, ((B[i][0] + B[i][2]) / 2, B[i][3]), w, h) for p in Z["exclude"]))
    kb = sorted(keep, key=lambda i: -(B[i][2] - B[i][0]) * (B[i][3] - B[i][1])); kept = []
    for i in kb:
        if all(ios(B[i], B[j]) <= 0.6 for j in kept): kept.append(i)
    kc = [i for i in keep if not cut[i]]; kc2 = []
    for i in sorted(kc, key=lambda i: -(B[i][2] - B[i][0]) * (B[i][3] - B[i][1])):
        if all(ios(B[i], B[j]) <= 0.6 for j in kc2): kc2.append(i)
    out["a"].append(cnt(keep)); out["b"].append(cnt(kept)); out["c"].append(cnt(kc2)); g.append(int(r["gt"]))
g = np.array(g)
for k, nm in (("a", "지금 (NMS)"), ("b", "+ 포함 억제"), ("c", "+ 타일 경계 잘린 상자 제거")):
    p = np.array(out[k]); e = p - g
    s = " ".join(f"{lab} {np.abs(e[(g >= lo) & (g <= hi)]).mean():.2f}/{e[(g >= lo) & (g <= hi)].mean():+.2f}" for lo, hi, lab in ((0, 0, "0"), (1, 3, "1-3"), (4, 7, "4-7")))
    print(f"{nm}: MAE {np.abs(e).mean():.2f} bias {e.mean():+.2f} 정확 {np.mean(e == 0) * 100:.0f}% ±1 {np.mean(np.abs(e) <= 1) * 100:.0f}% 빈방오검출 {np.mean(p[g == 0] > 0) * 100:.0f}% | 인원별 MAE/bias {s}")
