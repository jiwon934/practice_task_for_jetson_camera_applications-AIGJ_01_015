"""T0-4 지연 누적 비교: (a) cap.read() 그대로 반복 vs (b) 최신 1장 리더, 둘 다 매 프레임 처리 0.2초
지연(lag) = (시작 후 지난 실제 시간) - (시작 후 지난 프레임 시각)
  프레임 시각: CAP_PROP_POS_MSEC (웹캠 V4L2 = 버퍼 타임스탬프, RTSP = 스트림 시각)
  POS_MSEC가 0이면 프레임 수 / 카메라 fps 로 대신 셈 (a만)
사용: python latency.py --url ... --mode a --seconds 120 [--size 1280x720 --fps 30]
결과: runs/latency_<mode>_<이름>.csv (1초마다), 요약은 runs/latency.jsonl, 0초·끝 스냅샷 jpg
"""
import argparse, csv, json, os, re, time
import cv2
from camlib import open_cap, safe_name, LatestReader

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--mode", choices=["a", "b"], required=True)
ap.add_argument("--seconds", type=float, default=120)
ap.add_argument("--work", type=float, default=0.2, help="프레임마다 처리 시간(초)")
ap.add_argument("--size", default=None)
ap.add_argument("--fps", type=int, default=None)
ap.add_argument("--tag", default=None)
a = ap.parse_args()

w, h = map(int, a.size.split("x")) if a.size else (None, None)
name = safe_name(a.url)
tag = a.tag or re.sub(r"[^A-Za-z0-9]+", "_", name.split("/")[-1] if "rtsp" in name else "webcam")[:30]
os.makedirs("runs", exist_ok=True)
base = f"runs/latency_{a.mode}_{tag}"

if a.mode == "a":
    cap = open_cap(a.url, w, h, a.fps)
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    def get():
        ok, f = cap.read()
        return (f, cap.get(cv2.CAP_PROP_POS_MSEC)) if ok else (None, None)
else:
    rd = LatestReader(a.url, w, h, a.fps); rd.start()
    while rd.latest()[0] is None:
        time.sleep(0.05)
    def get():
        f, ts, pos, seq = rd.latest()
        return f, pos

rows, n, t0, p0, last_print = [], 0, None, None, 0
snap0 = snap1 = None
use_count = False
start = time.time()
while time.time() - start < a.seconds:
    f, pos = get()
    if f is None:
        time.sleep(0.05); continue
    now = time.time()
    n += 1
    if t0 is None:
        t0, p0 = now, pos
        use_count = not pos  # POS_MSEC를 못 주면 프레임 수로
        snap0 = f.copy()
    if use_count and a.mode == "a":
        frame_elapsed = (n - 1) / cam_fps
    else:
        frame_elapsed = ((pos or 0) - (p0 or 0)) / 1000.0
    lag = (now - t0) - frame_elapsed
    if now - last_print >= 1:
        rows.append([round(now - start, 1), n, round(lag * 1000)])
        last_print = now
    snap1 = f
    time.sleep(a.work)  # 처리 흉내

if a.mode == "b":
    rd.stop()
else:
    cap.release()
cv2.imwrite(base + "_t0.jpg", snap0); cv2.imwrite(base + f"_t{int(a.seconds)}.jpg", snap1)
with open(base + ".csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["t_s", "frames_used", "lag_ms"]); wr.writerows(rows)
lags = [r[2] for r in rows]
summ = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "url": name, "mode": a.mode, "work_s": a.work,
        "seconds": a.seconds, "frames_used": n, "used_fps": round(n / (time.time() - start), 2),
        "lag_ms_first10s": round(sum(lags[:10]) / max(1, len(lags[:10]))),
        "lag_ms_last10s": round(sum(lags[-10:]) / max(1, len(lags[-10:]))),
        "lag_ms_max": max(lags) if lags else None, "clock": "frame_count" if use_count else "pos_msec"}
print(json.dumps(summ, ensure_ascii=False))
open("runs/latency.jsonl", "a").write(json.dumps(summ, ensure_ascii=False) + "\n")
