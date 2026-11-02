"""과제 5: 아이패드 화면용 체커보드 PNG. 화면 해상도(px)와 ppi를 주면 칸 한 변이 정확히 --square-mm가 되도록 그림.
내부 코너 cols x rows (칸은 그보다 1개씩 많음), 둘레에 흰 여백. 화면을 100% 크기로(확대 없이) 띄워야 칸 크기가 맞음.
사용: python make_board.py --w 2360 --h 1640 --ppi 264 [--square-mm 15] [--cols 9 --rows 6]"""
import argparse
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--w", type=int, required=True); ap.add_argument("--h", type=int, required=True); ap.add_argument("--ppi", type=float, required=True)
ap.add_argument("--square-mm", type=float, default=None, help="비우면 화면에 들어가는 가장 큰 정수 mm")
ap.add_argument("--cols", type=int, default=9); ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--out", default="ipad_checkerboard.png")
a = ap.parse_args()
px_per_mm = a.ppi / 25.4
nx, ny = a.cols + 1, a.rows + 1
if a.square_mm is None:
    a.square_mm = float(int(min(a.w / (nx + 1), a.h / (ny + 1)) / px_per_mm))
sq = a.square_mm * px_per_mm
img = np.full((a.h, a.w), 255, np.uint8)
x0, y0 = (a.w - nx * sq) / 2, (a.h - ny * sq) / 2
for j in range(ny):
    for i in range(nx):
        if (i + j) % 2 == 0:
            cv2.rectangle(img, (round(x0 + i * sq), round(y0 + j * sq)), (round(x0 + (i + 1) * sq) - 1, round(y0 + (j + 1) * sq) - 1), 0, -1)
cv2.putText(img, f"{a.cols}x{a.rows} inner corners, square {a.square_mm:g} mm @ {a.ppi:g} ppi", (20, a.h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 128, 2)
cv2.imwrite(a.out, img)
print(f"{a.out}: {a.w}x{a.h}, 칸 {a.square_mm:g} mm = {sq:.1f} px, 보드 {nx * a.square_mm:g} x {ny * a.square_mm:g} mm")
