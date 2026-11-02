"""과제 2-2: 조명 × 거리별 사람 신뢰도. IP 카메라 + YOLO11n TensorRT FP16. 화면(:8030)에 가장 확실한 사람의 conf·박스 높이(px)·발 위치를 크게 표시.
/label?name=bright_2m 을 열면 그때부터 10초 동안 매 프레임 값을 모아 runs/t2_conf.csv에 한 줄로 요약(평균·최소·최대 conf, 검출률, 평균 박스 높이) + 스냅샷 저장.
사용: ~/eg2/bin/python conf_probe.py [--src rtsp:stream1] [--sec 10]"""
import argparse, csv, json, os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.expanduser("~/eg2-lab/applab/t0"))
os.environ["YOLO_AUTOINSTALL"] = "False"
import cv2, numpy as np
from ultralytics import YOLO
from camlib import LatestReader, safe_name

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="rtsp:stream1")
ap.add_argument("--model", default=os.path.expanduser("~/eg2-lab/applab/t2/yolo11n_fp16.engine"))
ap.add_argument("--conf", type=float, default=0.05, help="낮게 둬서 약한 검출도 기록")
ap.add_argument("--sec", type=float, default=10)
ap.add_argument("--port", type=int, default=8030, choices=[8020, 8030])
a = ap.parse_args()
RUNS = os.path.expanduser("~/eg2-lab/applab/runs")
CSV = os.path.join(RUNS, "t2_conf.csv")
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
if not os.path.exists(CSV):
    csv.writer(open(CSV, "w", newline="")).writerow(
        ["time", "label", "frames", "det_rate", "conf_mean", "conf_min", "conf_max", "box_h_px_mean", "foot_x", "foot_y", "snapshot"])

det = YOLO(a.model, task="detect")
rd = LatestReader(a.src, log=log); rd.start()
cond, jpeg = threading.Condition(), {"seq": 0, "data": None}
st = {"fps": 0.0, "conf": 0.0, "box_h": 0, "foot": None, "rec": None, "last": None}
rec = {"label": None, "end": 0, "rows": [], "snap": None}


def finish():
    rows = rec["rows"]; c = [r[0] for r in rows]; hit = [r for r in rows if r[0] >= 0.25]
    snap = os.path.join(RUNS, f"t2_conf_{rec['label']}.jpg")
    if rec["snap"] is not None:
        cv2.imwrite(snap, rec["snap"])
    s = {"time": time.strftime("%H:%M:%S"), "label": rec["label"], "frames": len(rows),
         "det_rate": round(len(hit) / max(len(rows), 1), 3), "conf_mean": round(float(np.mean(c)), 3) if c else 0,
         "conf_min": round(float(np.min(c)), 3) if c else 0, "conf_max": round(float(np.max(c)), 3) if c else 0,
         "box_h_px_mean": round(float(np.mean([r[1] for r in hit]))) if hit else 0,
         "foot_x": round(float(np.mean([r[2] for r in hit])), 3) if hit else None,
         "foot_y": round(float(np.mean([r[3] for r in hit])), 3) if hit else None, "snapshot": os.path.basename(snap)}
    csv.writer(open(CSV, "a", newline="")).writerow(list(s.values()))
    log(f"기록 {json.dumps(s, ensure_ascii=False)}"); st["last"] = s; rec["label"] = None


def loop():
    last, n, t0 = -1, 0, time.time()
    while True:
        f, ts, _, seq = rd.latest()
        if f is None or seq == last:
            time.sleep(0.003); continue
        last = seq; h, w = f.shape[:2]
        r = det.predict(f, classes=[0], conf=a.conf, imgsz=640, device=0, verbose=False)[0]
        vis = f.copy(); best = (0.0, 0, None, None)
        for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            col = (0, 255, 0) if cf >= 0.25 else (0, 165, 255)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
            cv2.putText(vis, f"{cf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
            if cf > best[0]:
                best = (cf, int(y2 - y1), (x1 + x2) / 2 / w, y2 / h)
        st.update(conf=round(best[0], 3), box_h=best[1], foot=None if best[2] is None else [round(best[2], 3), round(best[3], 3)])
        if rec["label"]:
            rec["rows"].append((best[0], best[1], best[2] or 0, best[3] or 0))
            if rec["snap"] is None and best[0] > 0:
                rec["snap"] = vis
            left = rec["end"] - time.time()
            st["rec"] = f"{rec['label']} {max(left, 0):.0f}s"
            if left <= 0:
                st["rec"] = None; finish()
        txt = f"max conf {best[0]:.2f}  box {best[1]} px" + (f"  foot ({best[2]:.2f},{best[3]:.2f})" if best[2] is not None else "  (no person)")
        cv2.rectangle(vis, (0, 0), (w, 230), (0, 0, 0), -1)
        cv2.putText(vis, txt, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 7)
        cv2.putText(vis, ("REC " + st["rec"]) if st["rec"] else (f"last: {st['last']['label']} mean {st['last']['conf_mean']}" if st["last"] else "idle"),
                    (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 255) if st["rec"] else (200, 200, 200), 6)
        vis = cv2.resize(vis, (1280, int(1280 * h / w)))
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with cond:
            jpeg["data"], jpeg["seq"] = buf.tobytes(), jpeg["seq"] + 1; cond.notify_all()
        n += 1; now = time.time()
        if now - t0 >= 1:
            st["fps"] = round(n / (now - t0), 1); n, t0 = 0, now


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def reply(self, body, ct="application/json"):
        self.send_response(200); self.send_header("Content-Type", ct); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == "/stream":
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers(); seq = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: jpeg["seq"] != seq, timeout=5); seq, data = jpeg["seq"], jpeg["data"]
                    if data is not None:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        if u.path == "/label":
            name = "".join(ch for ch in q.get("name", ["x"])[0] if ch.isalnum() or ch in "_-")[:40]
            if rec["label"]:
                return self.reply(json.dumps({"busy": rec["label"]}).encode())
            rec.update(label=name, end=time.time() + a.sec, rows=[], snap=None); log(f"기록 시작 {name}")
            return self.reply(json.dumps({"recording": name, "sec": a.sec}).encode())
        if u.path == "/stats":
            return self.reply(json.dumps(st).encode())
        self.reply(b"<!doctype html><meta charset=utf-8><title>t2 conf</title><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>"
                   b"<img src=/stream style='max-width:100%'><p id=s></p><script>setInterval(async()=>{document.getElementById('s').textContent="
                   b"JSON.stringify(await (await fetch('/stats')).json())},1000)</script>", "text/html")


log(f"입력 {safe_name(a.src)}, 기록 {a.sec:.0f}초, 결과 {CSV}")
threading.Thread(target=loop, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
