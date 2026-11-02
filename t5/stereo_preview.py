"""과제 5: 웹캠 2대 나란히 미리보기. 가로 기준선(같은 높이 비교용), 카메라별 선명도·FPS·체커보드 검출 표시.
페이지는 /frame.jpg를 계속 새로 받아 바꿔 끼워서, 연결이 멈춰도 3초 안에 스스로 다시 붙음.
사용: ~/eg2/bin/python stereo_preview.py [--left 2.4] [--right 2.2] [--port 8031]"""
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.expanduser("~/eg2-lab/applab/t0"))
import cv2, numpy as np
from camlib import LatestReader

ap = argparse.ArgumentParser()
ap.add_argument("--left", default="2.4"); ap.add_argument("--right", default="2.2")
ap.add_argument("--size", default="1280x720"); ap.add_argument("--port", type=int, default=8031)
ap.add_argument("--cols", type=int, default=9); ap.add_argument("--rows", type=int, default=6)
a = ap.parse_args()
w, h = map(int, a.size.split("x"))
dev = lambda p: f"/dev/v4l/by-path/platform-3610000.usb-usb-0:{p}:1.0-video-index0"
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
cams = {"LEFT " + a.left: LatestReader(dev(a.left), w, h, 30, log=log), "RIGHT " + a.right: LatestReader(dev(a.right), w, h, 30, log=log)}
for r in cams.values(): r.start()
st, jpeg = {}, {"data": None}


def loop():
    last = {k: (0, time.time()) for k in cams}
    while True:
        time.sleep(0.1); tiles = []
        for k, r in cams.items():
            f, ts, _, seq = r.latest()
            if f is None:
                tiles.append(np.zeros((360, 640, 3), np.uint8)); continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); hh, ww = g.shape
            small = cv2.resize(g, (640, 360))
            ok, cs = cv2.findChessboardCorners(small, (a.cols, a.rows), flags=cv2.CALIB_CB_FAST_CHECK)
            n0, t0 = last[k]; now = time.time()
            if now - t0 >= 1:
                st[k] = {"fps": round((seq - n0) / (now - t0), 1), "size": f"{ww}x{hh}",
                         "sharp": round(cv2.Laplacian(g[hh // 4:3 * hh // 4, ww // 4:3 * ww // 4], cv2.CV_64F).var(), 1)}
                last[k] = (seq, now)
            t = cv2.resize(f, (640, 360))
            for y in range(0, 360, 30):
                cv2.line(t, (0, y), (640, y), (0, 255, 255) if y == 180 else (0, 140, 255), 1)
            cv2.line(t, (320, 0), (320, 360), (0, 140, 255), 1)
            if ok:
                cv2.drawChessboardCorners(t, (a.cols, a.rows), cs, ok)
            s = st.get(k, {})
            cv2.putText(t, f"{k}  {s.get('fps', 0)} fps  sharp {s.get('sharp', 0)}  {'BOARD' if ok else ''}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(t)
        _, buf = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg["data"] = buf.tobytes()


PAGE = b"""<!doctype html><meta charset=utf-8><title>t5 stereo preview</title><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>
<img id=v style='width:100%'><p id=s></p><script>
const v=document.getElementById('v');let last=Date.now(),busy=false;
function next(){if(busy)return;busy=true;const im=new Image();im.onload=()=>{v.src=im.src;last=Date.now();busy=false;setTimeout(next,100)};
 im.onerror=()=>{busy=false;setTimeout(next,1000)};im.src='/frame.jpg?'+Date.now()}
setInterval(()=>{if(Date.now()-last>3000){busy=false;next()}},1000);next();
setInterval(async()=>{try{document.getElementById('s').textContent=JSON.stringify(await (await fetch('/stats')).json())}catch(e){}},1000)
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def do_GET(self):
        if self.path.startswith("/frame.jpg") and jpeg["data"]:
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store"); self.end_headers()
            return self.wfile.write(jpeg["data"])
        body, ct = (json.dumps(st).encode(), "application/json") if self.path == "/stats" else (PAGE, "text/html")
        self.send_response(200); self.send_header("Content-Type", ct); self.end_headers(); self.wfile.write(body)


threading.Thread(target=loop, daemon=True).start()
log(f"미리보기 :{a.port}  왼쪽 {a.left}, 오른쪽 {a.right}")
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
