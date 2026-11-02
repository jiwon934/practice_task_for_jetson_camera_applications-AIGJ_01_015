"""B2 받는 쪽: RTSP(또는 파일·웹캠)를 최신 1장 리더로 받아 YOLO11n(GPU, FP16)으로 사람 검출, 결과 화면을 브라우저(MJPEG)로 보여 줌.
1초마다 받은 FPS·처리 FPS·추론 ms·프레임 나이(받은 뒤 처리 끝날 때까지)를 로그로 남김.
사용: ~/eg2/bin/python rtsp_infer.py [--src rtsp://127.0.0.1:8554/webcam] [--port 8030] [--sec 0]"""
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.expanduser("~/eg2-lab/applab/t0"))
import cv2, torch
from ultralytics import YOLO
from camlib import LatestReader, safe_name

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="rtsp://127.0.0.1:8554/webcam")
ap.add_argument("--port", type=int, default=8030, choices=[0, 8020, 8030], help="0이면 화면 서버 없이 측정만")
ap.add_argument("--model", default=os.path.expanduser("~/eg2-lab/yolo11n.pt"))
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--sec", type=float, default=0, help="이 시간 뒤 끝냄 (0 = 계속)")
a = ap.parse_args()
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

det = YOLO(a.model)
dev = "cuda" if torch.cuda.is_available() else "cpu"
rd = LatestReader(a.src, log=log); rd.start()
cond, jpeg = threading.Condition(), {"seq": 0, "data": None}
st = {"recv_fps": 0.0, "proc_fps": 0.0, "infer_ms": 0.0, "age_ms": 0.0, "persons": 0}


def infer_loop():
    last_seq, n, t0, recv0 = -1, 0, time.time(), 0
    inf_sum = age_sum = 0.0
    while True:
        f, ts, _, seq = rd.latest()
        if f is None or seq == last_seq:
            time.sleep(0.002); continue
        last_seq = seq
        t1 = time.time()
        r = det.predict(f, classes=[0], conf=a.conf, imgsz=a.imgsz, half=dev == "cuda", device=dev, verbose=False)[0]
        t2 = time.time()
        vis = r.plot(line_width=2, labels=True)
        age = (time.time() - ts) * 1000
        cv2.putText(vis, f"{st['proc_fps']:.1f} fps  infer {st['infer_ms']:.0f} ms  age {st['age_ms']:.0f} ms  persons {len(r.boxes)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with cond:
            jpeg["data"], jpeg["seq"] = buf.tobytes(), jpeg["seq"] + 1; cond.notify_all()
        n += 1; inf_sum += (t2 - t1) * 1000; age_sum += age; st["persons"] = len(r.boxes)
        now = time.time()
        if now - t0 >= 1:
            st.update(recv_fps=(rd.read_count - recv0) / (now - t0), proc_fps=n / (now - t0),
                      infer_ms=inf_sum / n, age_ms=age_sum / n)
            log(json.dumps({k: round(v, 1) for k, v in st.items()} | {"reconnects": rd.reconnects}))
            n, t0, recv0, inf_sum, age_sum = 0, now, rd.read_count, 0.0, 0.0


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
            seq = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: jpeg["seq"] != seq, timeout=5); seq, data = jpeg["seq"], jpeg["data"]
                    if data is not None:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        if self.path == "/stats":
            body, ctype = json.dumps(st | {"reconnects": rd.reconnects}).encode(), "application/json"
        else:
            body = (b"<!doctype html><meta charset=utf-8><title>b2 infer</title><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>"
                    b"<img src=/stream style='max-width:100%'><p id=s></p><script>setInterval(async()=>{document.getElementById('s').textContent="
                    b"JSON.stringify(await (await fetch('/stats')).json())},1000)</script>"); ctype = "text/html"
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)


log(f"입력 {safe_name(a.src)}, 모델 {os.path.basename(a.model)} ({dev}), 화면 :{a.port or '없음'}")
threading.Thread(target=infer_loop, daemon=True).start()
if a.port:
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever, daemon=True).start()
end = time.time() + a.sec if a.sec else None
while end is None or time.time() < end:
    time.sleep(0.5)
rd.stop()
os._exit(0)   # 데몬 스레드(CUDA·HTTP)가 남은 채 인터프리터가 정리되면 죽으므로 바로 끝냄
