"""과제 2-3: IP 카메라 + YOLO11n TensorRT FP16 + ByteTrack으로 가상선 IN/OUT 세기 → MQTT 발행, CSV 기록, 화면 :8030.
선은 정규화 좌표 (x1,y1)-(x2,y2). 박스 아래 가운데 점(발 위치)이 선의 어느 쪽인지(외적 부호)로 판단하고,
선에서 --margin 이상 떨어져야 '그쪽'으로 인정(선 위에서 흔들려도 두 번 세지 않음). 같은 ID는 같은 방향을 연달아 두 번 세지 않음 (IN 다음엔 OUT만, OUT 다음엔 IN만) → 들락날락은 매번 셈, 선 근처 흔들림은 안 셈.
사용: ~/eg2/bin/python line_count.py [--src rtsp:stream1] [--line 0.30,0.80,0.66,0.80] [--in-dir down]"""
import argparse, csv, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.expanduser("~/eg2-lab/applab/t0"))
os.environ["YOLO_AUTOINSTALL"] = "False"
import cv2, numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from camlib import LatestReader, safe_name

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="rtsp:stream1")
ap.add_argument("--model", default=os.path.expanduser("~/eg2-lab/applab/t2/yolo11n_fp16.engine"))
ap.add_argument("--line", default="0.30,0.80,0.66,0.80", help="x1,y1,x2,y2 (0~1)")
ap.add_argument("--in-dir", default="down", choices=["down", "up"], help="발이 선을 위→아래(카메라 쪽)로 넘으면 IN이면 down")
ap.add_argument("--margin", type=float, default=0.02, help="선에서 이만큼(화면 높이 비율) 떨어져야 그쪽으로 인정")
ap.add_argument("--conf", type=float, default=0.3)
ap.add_argument("--port", type=int, default=8030, choices=[0, 8020, 8030])
ap.add_argument("--topic", default="jiwon934/t2/line")
ap.add_argument("--csv", default=os.path.expanduser("~/eg2-lab/applab/runs/t2_line.csv"))
a = ap.parse_args()
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
L = np.array([float(v) for v in a.line.split(",")]).reshape(2, 2)

mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"jiwon934-t2-{os.getpid()}")
try:
    mq.connect("127.0.0.1", 1883, 30); mq.loop_start(); log("MQTT 연결 127.0.0.1:1883")
except Exception as e:
    log(f"MQTT 연결 실패 {e!r} (계속 진행, CSV만 기록)"); mq = None
new_csv = not os.path.exists(a.csv)
cf = open(a.csv, "a", newline=""); cw = csv.writer(cf)
if new_csv:
    cw.writerow(["ts", "time", "id", "dir", "in", "out"])

det = YOLO(a.model, task="detect")
rd = LatestReader(a.src, log=log); rd.start()
side, counted = {}, {}          # id -> 마지막으로 확실했던 쪽(+1/-1), id -> 마지막으로 센 방향
cnt = {"in": 0, "out": 0}
cond, jpeg = threading.Condition(), {"seq": 0, "data": None}
st = {"fps": 0.0, "infer_ms": 0.0, "tracks": 0}


def signed(p, w, h):
    """선(A→B)에 대한 점 p의 부호 있는 거리 (화면 높이로 정규화). 선분 바깥(좌우 끝 넘어)이면 None"""
    A, B = L[0] * (w, h), L[1] * (w, h)
    d = B - A; t = np.dot(p - A, d) / np.dot(d, d)
    if t < -0.05 or t > 1.05:
        return None
    q = p - A
    return float((d[0] * q[1] - d[1] * q[0]) / np.linalg.norm(d) / h)


def loop():
    last, n, t0, isum = -1, 0, time.time(), 0.0
    down_sign = None
    while True:
        f, ts, _, seq = rd.latest()
        if f is None or seq == last:
            time.sleep(0.003); continue
        last = seq
        h, w = f.shape[:2]
        if down_sign is None:   # 화면 아래쪽(카메라 쪽) 점의 부호 = 'down' 쪽
            # 선의 중점에서 화면 아래로 내린 점의 부호 (짧고 기운 선이면 화면 맨 아래 점이 선분 범위 밖이라 None이 될 수 있어 외적을 직접 계산)
            A, Bq = L[0] * (w, h), L[1] * (w, h); d = Bq - A; q = (A + Bq) / 2 + np.array([0, 0.1 * h]) - A
            down_sign = 1 if d[0] * q[1] - d[1] * q[0] > 0 else -1
        t1 = time.time()
        r = det.track(f, persist=True, tracker="bytetrack.yaml", classes=[0], conf=a.conf, imgsz=640, device=0, verbose=False)[0]
        isum += (time.time() - t1) * 1000
        vis = f.copy()
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
        for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.tolist(), ids):
            foot = np.array([(x1 + x2) / 2, y2])
            s = signed(foot, w, h)
            if s is not None and abs(s) >= a.margin:
                cur = 1 if s > 0 else -1
                prev = side.get(tid)
                if prev is not None and cur != prev:
                    d = "IN" if (cur == down_sign) == (a.in_dir == "down") else "OUT"
                    if counted.get(tid) != d:
                        counted[tid] = d; cnt[d.lower()] += 1
                        msg = {"ts": round(time.time(), 3), "id": tid, "dir": d, "in": cnt["in"], "out": cnt["out"]}
                        log(f"통과 {json.dumps(msg)}")
                        cw.writerow([msg["ts"], time.strftime("%H:%M:%S"), tid, d, cnt["in"], cnt["out"]]); cf.flush()
                        if mq: mq.publish(a.topic, json.dumps(msg), qos=1)
                side[tid] = cur
            col = (0, 200, 255) if tid in counted else (0, 255, 0)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
            cv2.circle(vis, tuple(foot.astype(int)), 8, col, -1)
            cv2.putText(vis, f"id{tid}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
        A, B = (L[0] * (w, h)).astype(int), (L[1] * (w, h)).astype(int)
        cv2.line(vis, tuple(A), tuple(B), (0, 0, 255), 6)
        cv2.putText(vis, f"IN {cnt['in']}  OUT {cnt['out']}   {st['fps']:.1f} fps  {st['infer_ms']:.0f} ms",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
        vis = cv2.resize(vis, (1280, int(1280 * h / w)))
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with cond:
            jpeg["data"], jpeg["seq"] = buf.tobytes(), jpeg["seq"] + 1; cond.notify_all()
        n += 1; now = time.time()
        if now - t0 >= 1:
            st.update(fps=n / (now - t0), infer_ms=isum / n, tracks=len(ids)); n, t0, isum = 0, now, 0.0


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers(); seq = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: jpeg["seq"] != seq, timeout=5); seq, data = jpeg["seq"], jpeg["data"]
                    if data is not None:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        if self.path == "/reset":   # 카운터 0으로 (트랙 기억도 비움), CSV에 표시
            cnt.update({"in": 0, "out": 0}); counted.clear()
            cw.writerow([round(time.time(), 3), time.strftime("%H:%M:%S"), "-", "RESET", 0, 0]); cf.flush(); log("카운터 초기화")
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers(); self.wfile.write(b"reset\n"); return
        if self.path == "/stats":
            body, ct = json.dumps(st | cnt | {"reconnects": rd.reconnects}).encode(), "application/json"
        else:
            body, ct = (b"<!doctype html><meta charset=utf-8><title>t2 line</title><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>"
                        b"<img src=/stream style='max-width:100%'><p id=s></p><script>setInterval(async()=>{document.getElementById('s').textContent="
                        b"JSON.stringify(await (await fetch('/stats')).json())},1000)</script>"), "text/html"
        self.send_response(200); self.send_header("Content-Type", ct); self.end_headers(); self.wfile.write(body)


log(f"입력 {safe_name(a.src)}, 선 {a.line}, IN = 발이 선을 {'위→아래(카메라 쪽으로)' if a.in_dir == 'down' else '아래→위'} 넘을 때, MQTT {a.topic}")
threading.Thread(target=loop, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
