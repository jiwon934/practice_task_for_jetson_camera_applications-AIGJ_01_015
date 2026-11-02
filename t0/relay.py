"""view.py(:8020) MJPEG를 받아 작게 줄여 다시 내보냄. RTSP를 새로 열지 않음 (원격 브라우저 대역폭 줄이기용)."""
import argparse, json, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="http://127.0.0.1:8020")
ap.add_argument("--port", type=int, default=8030, choices=[8020, 8030])
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--quality", type=int, default=70)
a = ap.parse_args()
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
cond, jpeg, st = threading.Condition(), {"seq": 0, "data": None}, {"in": 0, "sent": 0, "in_fps": 0, "sent_fps": 0, "bytes": 0}

def reader():
    while True:
        try:
            r = urllib.request.urlopen(a.src + "/stream", timeout=10); buf = b""
            while True:
                buf += r.read(65536)
                s = buf.find(b"\xff\xd8"); e = buf.find(b"\xff\xd9", s + 2)
                if s < 0 or e < 0: continue
                img = cv2.imdecode(np.frombuffer(buf[s:e + 2], np.uint8), cv2.IMREAD_COLOR); buf = buf[e + 2:]
                if img is None: continue
                h, w = img.shape[:2]; img = cv2.resize(img, (a.width, a.width * h // w), interpolation=cv2.INTER_AREA)
                ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                with cond:
                    jpeg["data"], jpeg["seq"] = enc.tobytes(), jpeg["seq"] + 1; cond.notify_all()
                st["in"] += 1; st["bytes"] = len(enc)
        except Exception as ex:
            log(f"원본 끊김 {ex!r}, 2초 뒤 다시"); time.sleep(2)

def meter():
    li, ls = 0, 0
    while True:
        time.sleep(1); st["in_fps"], st["sent_fps"] = st["in"] - li, st["sent"] - ls; li, ls = st["in"], st["sent"]

CLOCK = urllib.request.urlopen(a.src + "/clock").read()
PAGE = b"""<!doctype html><meta charset=utf-8><title>t0 relay</title><body style="margin:0;background:#111;color:#ddd;font-family:sans-serif">
<img src=/stream style="max-width:100%"><p id=s></p><script>
setInterval(async()=>{document.getElementById(s).textContent=JSON.stringify(await (await fetch(/stats)).json())},1000)
</script>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers(); seq = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: jpeg["seq"] != seq, timeout=5); seq, data = jpeg["seq"], jpeg["data"]
                    if data is None: continue
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"); st["sent"] += 1
            except (BrokenPipeError, ConnectionResetError):
                return
        body, ctype = (CLOCK, "text/html") if self.path == "/clock" else \
            (json.dumps({"in_fps": st["in_fps"], "sent_fps": st["sent_fps"], "jpeg_kb": round(st["bytes"] / 1024)}).encode(), "application/json") \
            if self.path == "/stats" else (PAGE, "text/html")
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)

threading.Thread(target=reader, daemon=True).start(); threading.Thread(target=meter, daemon=True).start()
log(f"릴레이 :{a.port} ← {a.src} ({a.width}px, q{a.quality})")
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
