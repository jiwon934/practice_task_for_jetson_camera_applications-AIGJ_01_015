"""1~3명 장면에서 적게 센 사진: conf 0.15까지 검출해 구역 필터에서 빠진 이유를 숫자로"""
import csv, json, os
import cv2, numpy as np, torch, torchvision
from ultralytics import YOLO
A = os.path.expanduser("~/eg2-lab/applab")
det = YOLO(os.path.join(A, "t3/yolo11n_fp16_960.engine"), task="detect")
Z = json.load(open(os.path.join(A, "t3/zones/room.json")))
G = [r for r in csv.DictReader(open(os.path.join(A, "runs/t3_snaps_1030/gt.csv"))) if 1 <= int(r["gt"]) <= 3 and int(r["count"]) < int(r["gt"])]
def inside(poly, p, w, h): return cv2.pointPolygonTest((np.array(poly) * (w, h)).astype(np.int32), (float(p[0]), float(p[1])), False) >= 0
for r in G:
    f = cv2.imread(os.path.join(A, "runs/t3_snaps_1030", r["file"])); h, w = f.shape[:2]
    tw, th = w / 2, h / 2; ox, oy = tw * 0.15, th * 0.15
    T = [(int(max(0, c * tw - ox)), int(max(0, rr * th - oy)), int(min(w, (c + 1) * tw + ox)), int(min(h, (rr + 1) * th + oy))) for rr in range(2) for c in range(2)]
    res = det.predict([f[y0:y1, x0:x1] for x0, y0, x1, y1 in T], classes=[0], conf=0.15, imgsz=960, device=0, verbose=False)
    B, S = [], []
    for (x0, y0, _, _), q in zip(T, res):
        for (a1, b1, a2, b2), s in zip(q.boxes.xyxy.tolist(), q.boxes.conf.tolist()): B.append([a1 + x0, b1 + y0, a2 + x0, b2 + y0]); S.append(s)
    keep = torchvision.ops.nms(torch.tensor(B), torch.tensor(S), 0.5).tolist() if B else []
    print(f"{r['file']} gt {r['gt']} logger {r['count']} raw {r['raw']}")
    for i in sorted(keep, key=lambda i: -S[i]):
        b = B[i]; foot = ((b[0] + b[2]) / 2, b[3]); cen = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        inc = any(inside(p, foot, w, h) for p in Z["include"]); exc = any(inside(p, foot, w, h) for p in Z["exclude"])
        why = "셈" if inc and not exc else ("발이 제외 영역" if exc else "발이 포함 영역 밖")
        cwhy = "가운데 기준이면 셈" if any(inside(p, cen, w, h) for p in Z["include"]) and not any(inside(p, cen, w, h) for p in Z["exclude"]) else ""
        print(f"   conf {S[i]:.2f} 크기 {int(b[2] - b[0])}x{int(b[3] - b[1])} 발 x{foot[0] / w:.2f} y{foot[1] / h:.2f} → {why} {cwhy}")
