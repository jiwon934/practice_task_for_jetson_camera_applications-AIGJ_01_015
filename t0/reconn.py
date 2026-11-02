"""T0-5 재연결 시험: LatestReader로 계속 읽으며 1초마다 FPS·재연결 수를 기록하고, 프레임이 끊긴/다시 온 시각을 남김."""
import argparse, os, time
from camlib import LatestReader

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--size", default="1280x720")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--stall", type=float, default=1.0, help="이 시간(초) 넘게 새 프레임이 없으면 끊김으로 봄")
a = ap.parse_args()
w, h = map(int, a.size.split("x"))
log = lambda m: print(time.strftime("%H:%M:%S.") + f"{time.time()%1*1000:03.0f}", m, flush=True)
rd = LatestReader(a.url, w, h, a.fps, log=log); rd.start()
log(f"시작 pid={os.getpid()} 입력={a.url}")
last_seq, last_t, stalled, stall_at = 0, time.time(), False, None
while True:
    time.sleep(1)
    _, ts, _, seq = rd.latest()
    now = time.time()
    fps = (seq - last_seq) / (now - last_t); last_seq, last_t = seq, now
    if ts and not stalled and now - ts > a.stall:
        stalled, stall_at = True, ts
        log(f"프레임 멈춤 (마지막 프레임 {time.strftime('%H:%M:%S', time.localtime(ts))}.{ts%1*1000:03.0f})")
    elif stalled and now - ts < a.stall:
        stalled = False
        log(f"프레임 다시 들어옴 (첫 프레임 {time.strftime('%H:%M:%S', time.localtime(ts))}, 끊긴 시간 {ts-stall_at:.1f}초, 재연결 {rd.reconnects}회)")
    log(f"fps={fps:.1f} seq={seq} 재연결={rd.reconnects}회")
