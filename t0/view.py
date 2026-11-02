"""T0-2·T0-3·T0-6 뷰어: 최신 1장 리더 + MJPEG로 PC 브라우저에 보여 줌 (포트 8020 또는 8030만)
  /           영상 페이지 (받은 FPS·보낸 FPS·재연결 횟수 표시)
  /stream     MJPEG
  /clock      glass-to-glass용 큰 ms 시계 (맥에서 열고 카메라를 그 화면에 향하게)
  /stats      JSON (recv_fps, sent_fps, reconnects, sharpness)
--focus     선명도(라플라시안 분산) 표시 — 초점·각도 맞출 때
사용: python view.py --url /dev/v4l/by-path/... --size 1280x720 --fps 30 --port 8020 [--focus]
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from camlib import LatestReader, safe_name

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--size", default=None)
ap.add_argument("--fps", type=int, default=None)
ap.add_argument("--port", type=int, default=8020, choices=[8020, 8030])
ap.add_argument("--focus", action="store_true")
ap.add_argument("--quality", type=int, default=80)
ap.add_argument("--snap", default=None, help="이 경로에 원본 프레임을 --snap-every 초마다 덮어써 저장 (구도 확인용)")
ap.add_argument("--snap-every", type=float, default=10)
a = ap.parse_args()
w, h = map(int, a.size.split("x")) if a.size else (None, None)

log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
rd = LatestReader(a.url, w, h, a.fps, log=log); rd.start()
st = {"recv_fps": 0.0, "sent_fps": 0.0, "sharp": None, "sent": 0}
jpeg = {"seq": -1, "data": None}
cond = threading.Condition()


def encoder():
    """새 프레임이 오면 글자를 얹고 JPEG로 한 번만 인코딩 (클라이언트 수와 무관)"""
    last_seq, t_win, recv_win, t_snap = -1, time.time(), 0, 0.0
    while True:
        f, ts, pos, seq = rd.latest()
        if f is None or seq == last_seq:
            time.sleep(0.005); continue
        recv_win += seq - last_seq if last_seq >= 0 else 1
        last_seq = seq
        now = time.time()
        if now - t_win >= 1:
            st["recv_fps"] = recv_win / (now - t_win); recv_win, t_win = 0, now
        if a.snap and now - t_snap >= a.snap_every:
            cv2.imwrite(a.snap, f); t_snap = now
        img = f.copy()
        txt =f"{safe_name(a.url)[-28:]}  recv {st['recv_fps']:.1f}  sent {st['sent_fps']:.1f}  reconn {rd.reconnects}"
        if a.focus:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            hh, ww = g.shape
            st["sharp"] = cv2.Laplacian(g[hh // 4:3 * hh // 4, ww // 4:3 * ww // 4], cv2.CV_64F).var()
            txt += f"  sharp {st['sharp']:.0f}"
            cv2.rectangle(img, (ww // 4, hh // 4), (3 * ww // 4, 3 * hh // 4), (0, 255, 255), 1)
        cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        with cond:
            jpeg["seq"], jpeg["data"] = seq, buf.tobytes(); cond.notify_all()


def sent_meter():
    last = 0
    while True:
        time.sleep(1); st["sent_fps"] = st["sent"] - last; last = st["sent"]


CLOCK = b"""<!doctype html><meta charset=utf-8><title>clock</title>
<body style="margin:0;background:#000;color:#fff;font:bold 22vw monospace;display:flex;align-items:center;justify-content:center;height:100vh">
<div id=c></div><script>
function t(){const d=new Date();document.getElementById('c').textContent=
String(d.getSeconds()).padStart(2,'0')+'.'+String(d.getMilliseconds()).padStart(3,'0');requestAnimationFrame(t)}t()
</script>"""
PAGE = b"""<!doctype html><meta charset=utf-8><title>t0 view</title><body style="margin:0;background:#111;color:#ddd;font-family:sans-serif">
<img src=/stream style="max-width:100%"><p id=s></p><script>
setInterval(async()=>{document.getElementById('s').textContent=JSON.stringify(await (await fetch('/stats')).json())},1000)
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
            seq = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: jpeg["seq"] != seq, timeout=5)
                        seq, data = jpeg["seq"], jpeg["data"]
                    if data is None:
                        continue
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                    st["sent"] += 1
            except (BrokenPipeError, ConnectionResetError):
                return
        body, ctype = (CLOCK, "text/html") if self.path == "/clock" else \
            (json.dumps({"recv_fps": round(st["recv_fps"], 1), "sent_fps": st["sent_fps"], "reconnects": rd.reconnects,
                         "sharpness": None if st["sharp"] is None else round(st["sharp"], 1)}).encode(), "application/json") \
            if self.path == "/stats" else (PAGE, "text/html")
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)


threading.Thread(target=encoder, daemon=True).start()
threading.Thread(target=sent_meter, daemon=True).start()
log(f"뷰어 http://<Jetson IP>:{a.port}/  (입력 {safe_name(a.url)})")
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
