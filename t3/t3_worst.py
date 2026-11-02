"""오차 큰 스냅샷을 occupancy.py와 같은 방식(960, 2x2 타일, NMS)으로 다시 검출해 상자 특성을 숫자로만 출력 (사진은 젯슨에)"""
import csv, json, os, sys
import cv2, numpy as np, torch, torchvision
from ultralytics import YOLO
A = os.path.expanduser("~/eg2-lab/applab")
det = YOLO(os.path.join(A, "t3/yolo11n_fp16_960.engine"), task="detect")
Z = json.load(open(os.path.join(A, "t3/zones/room.json")))
G = list(csv.DictReader(open(os.path.join(A, "runs/t3_snaps_1030/gt.csv"))))
worst = sorted(G, key=lambda r: -abs(int(r["count"]) - int(r["gt"])))[:6]
def inside(poly, p, w, h): return cv2.pointPolygonTest((np.array(poly) * (w, h)).astype(np.int32), (float(p[0]), float(p[1])), False) >= 0
for r in worst:
    f = cv2.imread(os.path.join(A, "runs/t3_snaps_1030", r["file"])); h, w = f.shape[:2]
    tw, th = w / 2, h / 2; ox, oy = tw * 0.15, th * 0.15
    T = [(int(max(0, c * tw - ox)), int(max(0, rr * th - oy)), int(min(w, (c + 1) * tw + ox)), int(min(h, (rr + 1) * th + oy))) for rr in range(2) for c in range(2)]
    res = det.predict([f[y0:y1, x0:x1] for x0, y0, x1, y1 in T], classes=[0], conf=0.25, imgsz=960, device=0, verbose=False)
    B, S = [], []
    for (x0, y0, _, _), q in zip(T, res):
        for (a1, b1, a2, b2), s in zip(q.boxes.xyxy.tolist(), q.boxes.conf.tolist()): B.append([a1 + x0, b1 + y0, a2 + x0, b2 + y0]); S.append(s)
    keep = torchvision.ops.nms(torch.tensor(B), torch.tensor(S), 0.5).tolist() if B else []
    B = [B[i] for i in keep]; S = [S[i] for i in keep]
    cnt = [(b, s) for b, s in zip(B, S) if any(inside(p, ((b[0] + b[2]) / 2, b[3]), w, h) for p in Z["include"]) and not any(inside(p, ((b[0] + b[2]) / 2, b[3]), w, h) for p in Z["exclude"])]
    ov = []
    for i in range(len(cnt)):
        for j in range(i + 1, len(cnt)):
            a, b = cnt[i][0], cnt[j][0]; ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy; small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if small and inter / small > 0.6: ov.append((i, j, round(inter / small, 2)))
    print(f"{r['file']} gt {r['gt']} logger {r['count']} raw {r['raw']} | 재검출 셈 {len(cnt)} (conf≥0.25 기준)")
    for k, (b, s) in enumerate(sorted(cnt, key=lambda x: x[1])):
        print(f"   conf {s:.2f} 크기 {int(b[2] - b[0])}x{int(b[3] - b[1])} 위치 x{(b[0] + b[2]) / 2 / w:.2f} 발y{b[3] / h:.2f} 세로/가로 {(b[3] - b[1]) / max(1, b[2] - b[0]):.1f}")
    if ov: print("   상자 겹침(작은 상자의 60% 이상이 다른 상자 안 = 중복/부분 검출 의심):", ov)
