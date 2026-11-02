"""T0-1 입력 점검: 첫 프레임 시간, 실측 FPS, 해상도, 끊김, CPU 사용률
사용: python probe.py --url /dev/v4l/by-path/...video-index0 --size 1280x720 --fps 30 --seconds 30 --snapshot runs/x.jpg
      python probe.py --url rtsp:stream2 --seconds 30
결과는 화면과 runs/probe.jsonl 에 한 줄씩 (비밀번호 없음)
"""
import argparse, json, os, time
import cv2
from camlib import open_cap, safe_name, cpu_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--size", default=None, help="웹캠 요청 해상도 예: 1280x720")
ap.add_argument("--fps", type=int, default=None, help="웹캠 요청 FPS")
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--snapshot", default=None)
ap.add_argument("--out", default="runs/probe.jsonl")
a = ap.parse_args()

w, h = map(int, a.size.split("x")) if a.size else (None, None)
name = safe_name(a.url)
cpu = cpu_sampler()
t0 = time.time()
cap = open_cap(a.url, w, h, a.fps)
t_open = time.time() - t0
if not cap.isOpened():
    raise SystemExit(f"열기 실패: {name}")
ok, f = cap.read()
t_first = time.time() - t0
if not ok:
    raise SystemExit(f"첫 프레임 실패: {name}")
if a.snapshot:
    os.makedirs(os.path.dirname(a.snapshot) or ".", exist_ok=True)
    cv2.imwrite(a.snapshot, f)
res = f"{f.shape[1]}x{f.shape[0]}"
rep_fps = cap.get(cv2.CAP_PROP_FPS)
cpu()
n, fails, gaps, max_gap = 0, 0, 0, 0.0
t_start = last = time.time()
while time.time() - t_start < a.seconds:
    ok, f = cap.read()
    now = time.time()
    if not ok:
        fails += 1; time.sleep(0.02); continue
    gap = now - last; last = now
    max_gap = max(max_gap, gap)
    if n > 5 and rep_fps > 0 and gap > 3.0 / rep_fps:
        gaps += 1
    n += 1
dur = time.time() - t_start
cpu_proc, cpu_sys = cpu()
cap.release()
r = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "url": name, "req": a.size, "req_fps": a.fps,
     "res": res, "reported_fps": round(rep_fps, 2), "open_s": round(t_open, 3), "first_frame_s": round(t_first, 3),
     "frames": n, "seconds": round(dur, 1), "fps": round(n / dur, 2), "read_fail": fails,
     "gaps_over_3x": gaps, "max_gap_ms": round(max_gap * 1000), "cpu_proc_pct": round(cpu_proc, 1),
     "cpu_sys_pct": round(cpu_sys, 1)}
print(json.dumps(r, ensure_ascii=False))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
open(a.out, "a").write(json.dumps(r, ensure_ascii=False) + "\n")
