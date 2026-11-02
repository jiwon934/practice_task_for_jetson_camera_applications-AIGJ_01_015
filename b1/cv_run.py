"""B1: OpenCV FFmpeg(CPU)로 같은 입력을 복원. 파일은 --rt 없으면 최대 속도, --rt면 파일 FPS에 맞춰 읽음.
사용: ~/eg2/bin/python cv_run.py NAME 입력(파일 또는 rtsp:stream1) [초] [--rt]"""
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/eg2-lab/applab/t0"))
import cv2
from camlib import open_cap

name, src = sys.argv[1], sys.argv[2]
dur = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 60.0
rt = "--rt" in sys.argv


def sys_cpu():
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    return sum(v), v[3] + v[4]


cap = open_cap(src)
fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
n, t_first = 0, None
t0, c0, s0 = time.time(), os.times(), sys_cpu()
while time.time() - t0 < dur:
    ok, f = cap.read()
    if not ok:
        break
    now = time.time()
    if t_first is None:
        t_first = now
    n += 1
    if rt:
        wait = t_first + n / fps_src - time.time()
        if wait > 0:
            time.sleep(wait)
t1, c1, s1 = time.time(), os.times(), sys_cpu()
cap.release()
wall = t1 - t0
proc = (c1.user + c1.system - c0.user - c0.system) / wall * 100
tot, idle = s1[0] - s0[0], s1[1] - s0[1]
print(json.dumps({"case": name, "frames": n, "sec": round(wall, 1), "fps": round(n / (t1 - t_first), 1) if t_first else 0,
                  "proc_cpu_1core": round(proc, 1), "proc_cpu_6core": round(proc / 6, 1),
                  "sys_cpu": round(100 * (tot - idle) / tot, 1), "error": None}, ensure_ascii=False), flush=True)
