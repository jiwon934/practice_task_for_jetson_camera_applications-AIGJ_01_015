"""과제 3: 재실 인원 카운터. IP 카메라 + YOLO11n TensorRT FP16 (사람만), 타일 나눠 검출 → NMS로 합침,
구역 JSON(포함·제외 다각형, 이름 붙은 구역)으로 기준점(발 또는 가운데)이 들어간 사람만 셈, 최근 N초 통계로 흔들림 줄임.
화면 :8030 (원시·보정 인원, 구역 표시), /draw 에서 스냅샷 위에 다각형을 클릭해 그려 저장.
MQTT 127.0.0.1:1883 토픽 jiwon934/t3/occupancy (retain) — 값이 바뀔 때와 10초마다. 기록 --log CSV, --snap-every 초마다 원본 스냅샷(젯슨에만).
사용: ~/eg2/bin/python occupancy.py [--imgsz 640|960] [--tiles 1x1|2x2] [--overlap 0.15] [--window 5] [--stat median|p75|max] [--anchor foot|center]"""
import argparse, collections, csv, json, os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.expanduser("~/eg2-lab/applab/t0"))
os.environ["YOLO_AUTOINSTALL"] = "False"
import cv2, numpy as np, torch, torchvision
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from camlib import LatestReader, safe_name

T3 = os.path.expanduser("~/eg2-lab/applab/t3")
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="rtsp:stream1")
ap.add_argument("--zones", default=os.path.join(T3, "zones/room.json"))
ap.add_argument("--imgsz", type=int, default=640, choices=[640, 960])
ap.add_argument("--tiles", default="1x1", help="열x행, 예: 2x2")
ap.add_argument("--overlap", type=float, default=0.15)
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--iou", type=float, default=0.5, help="타일 사이 겹친 박스 합칠 때 NMS IoU")
ap.add_argument("--anchor", default="foot", choices=["foot", "center"])
ap.add_argument("--window", type=float, default=5, help="보정 인원 통계 구간(초)")
ap.add_argument("--stat", default="median", choices=["median", "p75", "max"])
ap.add_argument("--windows", default="3,10,30", help="같은 검출로 동시에 계산해 기록할 보정 창들(초) — T2-4 비교용")
ap.add_argument("--log", default=None, help="CSV 경로 (ts, raw, count, 구역별)")
ap.add_argument("--snap-every", type=float, default=0, help="초마다 원본 스냅샷 (0 = 안 함)")
ap.add_argument("--snap-dir", default=os.path.expanduser("~/eg2-lab/applab/runs/t3_snaps"))
ap.add_argument("--topic", default="jiwon934/t3/occupancy")
ap.add_argument("--port", type=int, default=8030, choices=[0, 8020, 8030], help="0이면 화면 서버·그리기 없음 (장시간 실행용)")
ap.add_argument("--log-every", type=float, default=0, help="CSV를 이 초마다 한 줄 (0 = 매 프레임)")
a = ap.parse_args()
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
COLS, ROWS = map(int, a.tiles.lower().split("x"))

ENGINE = {640: os.path.expanduser("~/eg2-lab/applab/t2/yolo11n_fp16.engine"), 960: os.path.join(T3, "yolo11n_fp16_960.engine")}[a.imgsz]
det = YOLO(ENGINE, task="detect")


def load_zones():
    if os.path.exists(a.zones):
        return json.load(open(a.zones))
    return {"room": "classroom", "capacity": 30, "include": [[[0, 0], [1, 0], [1, 1], [0, 1]]], "exclude": [], "zones": {}}


Z = load_zones()
zlock = threading.Lock()


def tiles(w, h):
    """겹침을 둔 타일 (x0, y0, x1, y1)"""
    tw, th = w / COLS, h / ROWS; ox, oy = tw * a.overlap, th * a.overlap
    return [(int(max(0, c * tw - ox)), int(max(0, r * th - oy)), int(min(w, (c + 1) * tw + ox)), int(min(h, (r + 1) * th + oy)))
            for r in range(ROWS) for c in range(COLS)]


def detect(f):
    h, w = f.shape[:2]; T = tiles(w, h)
    crops = [f[y0:y1, x0:x1] for x0, y0, x1, y1 in T]
    res = det.predict(crops, classes=[0], conf=a.conf, imgsz=a.imgsz, device=0, verbose=False)
    boxes, scores = [], []
    for (x0, y0, _, _), r in zip(T, res):
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            boxes.append([x1 + x0, y1 + y0, x2 + x0, y2 + y0]); scores.append(s)
    if not boxes:
        return []
    b, s = torch.tensor(boxes), torch.tensor(scores)
    keep = torchvision.ops.nms(b, s, a.iou) if len(T) > 1 else torch.arange(len(b))
    return [(boxes[i], scores[i]) for i in keep.tolist()]


def inside(poly, p, w, h):
    return cv2.pointPolygonTest((np.array(poly) * (w, h)).astype(np.int32), (float(p[0]), float(p[1])), False) >= 0


def count(dets, w, h):
    with zlock:
        z = dict(Z)
    per = {n: 0 for n in z.get("zones", {})}; total = 0; flags = []
    for (x1, y1, x2, y2), s in dets:
        p = ((x1 + x2) / 2, y2 if a.anchor == "foot" else (y1 + y2) / 2)
        ok = any(inside(P, p, w, h) for P in z["include"]) and not any(inside(P, p, w, h) for P in z.get("exclude", []))
        flags.append((p, ok))
        if ok:
            total += 1
            for n, P in z.get("zones", {}).items():
                if inside(P, p, w, h):
                    per[n] += 1
    return total, per, flags


hist = collections.deque()
WINS = [float(x) for x in a.windows.split(",")] if a.windows else []
MARK = {"name": ""}
st = {"raw": 0, "count": 0, "zones": {}, "fps": 0.0, "ms": 0.0, "wins": {}}
cond, jpeg, latest = threading.Condition(), {"seq": 0, "data": None}, {"frame": None}
mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"jiwon934-t3-{os.getpid()}")
try:
    mq.connect("127.0.0.1", 1883, 30); mq.loop_start(); log("MQTT 연결 127.0.0.1:1883")
except Exception as e:
    log(f"MQTT 연결 실패 {e!r}"); mq = None
if a.log:
    new = not os.path.exists(a.log)
    lf = open(a.log, "a", newline=""); lw = csv.writer(lf)
    if new:
        lw.writerow(["ts", "time", "raw", "count", "fps", "infer_ms", "reconnects"] + [f"count_w{w:g}" for w in WINS] + ["mark"] + [f"zone_{n}" for n in Z.get("zones", {})])
if a.snap_every:
    os.makedirs(a.snap_dir, exist_ok=True)
    sf = open(os.path.join(a.snap_dir, "snaps.csv"), "a", newline=""); sw = csv.writer(sf)


def stat_over(now, win):
    v = np.array([x for t, x in hist if now - t <= win])
    return int(round({"median": np.median(v), "p75": np.percentile(v, 75), "max": v.max()}[a.stat]))


def smooth(now, raw):
    hist.append((now, raw))
    keep = max([a.window] + WINS)
    while hist and now - hist[0][0] > keep:
        hist.popleft()
    st["wins"] = {f"w{w:g}": stat_over(now, w) for w in WINS}
    return stat_over(now, a.window)


def publish(now):
    if mq:
        mq.publish(a.topic, json.dumps({"ts": round(now, 3), "room": Z.get("room"), "count": st["count"], "raw": st["raw"],
                                        "zones": st["zones"], "capacity": Z.get("capacity")}), qos=1, retain=True)


def loop():
    last, n, t0, msum, last_pub, last_cnt, last_snap, last_log = -1, 0, time.time(), 0.0, 0, None, 0, 0
    while True:
        f, ts, _, seq = rd.latest()
        if f is None or seq == last:
            time.sleep(0.003); continue
        last = seq; h, w = f.shape[:2]; latest["frame"] = f
        t = time.perf_counter(); dets = detect(f); msum += (time.perf_counter() - t) * 1000
        raw, per, flags = count(dets, w, h); now = time.time()
        st.update(raw=raw, count=smooth(now, raw), zones=per)
        if st["count"] != last_cnt or now - last_pub >= 10:
            publish(now); last_pub, last_cnt = now, st["count"]
        if a.log and now - last_log >= a.log_every:
            lw.writerow([round(now, 3), time.strftime("%H:%M:%S"), raw, st["count"], st["fps"], st["ms"], rd.reconnects] + [st["wins"].get(f"w{w:g}") for w in WINS] + [MARK["name"]] + [per[k] for k in per]); lf.flush()
            last_log = now
        if not a.port:                     # 화면 없음: 그리기·인코딩 생략
            n += 1
            if now - t0 >= 1:
                st.update(fps=round(n / (now - t0), 1), ms=round(msum / n, 1)); n, t0, msum = 0, now, 0.0
            continue
        if a.snap_every and now - last_snap >= a.snap_every:
            fn = f"snap_{int(now)}.jpg"; cv2.imwrite(os.path.join(a.snap_dir, fn), f)
            sw.writerow([round(now, 3), fn, raw, st["count"]]); sf.flush(); last_snap = now
        vis = f.copy()
        with zlock:
            z = dict(Z)
        for P in z["include"]:
            cv2.polylines(vis, [(np.array(P) * (w, h)).astype(np.int32)], True, (0, 255, 0), 4)
        for P in z.get("exclude", []):
            cv2.polylines(vis, [(np.array(P) * (w, h)).astype(np.int32)], True, (0, 0, 255), 4)
        for nm, P in z.get("zones", {}).items():
            q = (np.array(P) * (w, h)).astype(np.int32); cv2.polylines(vis, [q], True, (255, 200, 0), 3)
            cv2.putText(vis, f"{nm}:{per.get(nm, 0)}", tuple(q[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 200, 0), 4)
        if COLS * ROWS > 1:
            for x0, y0, x1, y1 in tiles(w, h):
                cv2.rectangle(vis, (x0, y0), (x1, y1), (120, 120, 120), 2)
        for ((x1, y1, x2, y2), s), (p, ok) in zip(dets, flags):
            col = (0, 255, 0) if ok else (0, 0, 255)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
            cv2.circle(vis, (int(p[0]), int(p[1])), 10, col, -1)
            cv2.putText(vis, f"{s:.2f}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        cv2.rectangle(vis, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.putText(vis, f"count {st['count']}  (raw {raw}, {a.stat} {a.window:.0f}s)  {st['fps']:.1f} fps  {st['ms']:.0f} ms  {a.imgsz} {a.tiles}",
                    (24, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 5)
        vis = cv2.resize(vis, (1280, int(1280 * h / w)))
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with cond:
            jpeg["data"], jpeg["seq"] = buf.tobytes(), jpeg["seq"] + 1; cond.notify_all()
        n += 1; now = time.time()
        if now - t0 >= 1:
            st.update(fps=round(n / (now - t0), 1), ms=round(msum / n, 1)); n, t0, msum = 0, now, 0.0


DRAW = b"""<!doctype html><meta charset=utf-8><title>t3 zones</title><body style="background:#111;color:#ddd;font-family:sans-serif;margin:8px">
<p>Click points on the snapshot, then press a button to close the polygon as include / exclude / named zone. Save writes zones/room.json.</p>
<canvas id=c style="max-width:100%;cursor:crosshair"></canvas><br>
<button onclick="add('include')">close as INCLUDE (green)</button> <button onclick="add('exclude')">close as EXCLUDE (red)</button>
<input id=zn placeholder="zone name"> <button onclick="add('zone')">close as named ZONE</button>
<button onclick="pts=[];draw()">clear points</button> <button onclick="Z={room:Z.room,capacity:Z.capacity,include:[],exclude:[],zones:{}};draw()">clear all</button>
room <input id=rm size=10> capacity <input id=cp size=4> <button onclick="save()">SAVE</button> <span id=msg></span>
<script>
let Z, pts=[], img=new Image(), c=document.getElementById('c'), x=c.getContext('2d');
fetch('/zones').then(r=>r.json()).then(z=>{Z=z; rm.value=Z.room||''; cp.value=Z.capacity||''; img.src='/snapshot.jpg?'+Date.now();});
img.onload=()=>{c.width=img.width;c.height=img.height;draw()};
c.onclick=e=>{const b=c.getBoundingClientRect();pts.push([(e.clientX-b.left)/b.width,(e.clientY-b.top)/b.height]);draw()};
function poly(P,col){x.strokeStyle=col;x.lineWidth=3;x.beginPath();P.forEach((p,i)=>{const X=p[0]*c.width,Y=p[1]*c.height;i?x.lineTo(X,Y):x.moveTo(X,Y)});x.closePath();x.stroke()}
function draw(){x.drawImage(img,0,0);Z.include.forEach(P=>poly(P,'#0f0'));(Z.exclude||[]).forEach(P=>poly(P,'#f00'));
 Object.entries(Z.zones||{}).forEach(([n,P])=>{poly(P,'#fc0');x.fillStyle='#fc0';x.font='24px sans-serif';x.fillText(n,P[0][0]*c.width,P[0][1]*c.height)});
 x.fillStyle='#0ff';pts.forEach(p=>{x.beginPath();x.arc(p[0]*c.width,p[1]*c.height,5,0,7);x.fill()})}
function add(k){if(pts.length<3)return alert('3 points or more');const P=pts.map(p=>[+p[0].toFixed(4),+p[1].toFixed(4)]);
 if(k=='zone'){if(!zn.value)return alert('zone name');Z.zones=Z.zones||{};Z.zones[zn.value]=P}else{Z[k]=Z[k]||[];Z[k].push(P)} pts=[];draw()}
function save(){Z.room=rm.value;Z.capacity=+cp.value||null;fetch('/zones',{method:'POST',body:JSON.stringify(Z)}).then(r=>r.text()).then(t=>msg.textContent=t)}
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def reply(self, body, ct="application/json"):
        body = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(200); self.send_header("Content-Type", ct); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
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
        if u.path == "/frame.jpg":           # 화면 주석이 그려진 최신 1장 (페이지가 반복해서 가져감)
            data = jpeg["data"]
            if data is None:
                return self.reply({"error": "no frame yet"})
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store"); self.end_headers()
            return self.wfile.write(data)
        if u.path == "/snapshot.jpg":
            f = latest["frame"]
            if f is None:
                return self.reply({"error": "no frame yet"})
            ok, buf = cv2.imencode(".jpg", cv2.resize(f, (1280, int(1280 * f.shape[0] / f.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 85])
            return self.reply(buf.tobytes(), "image/jpeg")
        if u.path == "/zones":
            return self.reply(Z)
        if u.path == "/mark":
            MARK["name"] = urllib.parse.parse_qs(u.query).get("name", [""])[0][:40]; log(f"구간 {MARK['name']!r}")
            return self.reply({"mark": MARK["name"]})
        if u.path == "/draw":
            return self.reply(DRAW, "text/html")
        if u.path == "/stats":
            return self.reply(st | {"room": Z.get("room"), "capacity": Z.get("capacity"), "reconnects": rd.reconnects})
        # MJPEG 스트림 대신 JS가 /frame.jpg를 계속 새로 받아 바꿔 끼움 → 연결이 멈춰도 3초 안에 스스로 다시 붙음
        self.reply(b"""<!doctype html><meta charset=utf-8><title>t3 occupancy</title><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>
<img id=v style='max-width:100%'><p><a style='color:#8cf' href=/draw>draw zones</a> <span id=s></span></p><script>
const v=document.getElementById('v');let last=Date.now(),busy=false;
function next(){if(busy)return;busy=true;const im=new Image();
 im.onload=()=>{v.src=im.src;last=Date.now();busy=false;setTimeout(next,80)};
 im.onerror=()=>{busy=false;setTimeout(next,1000)};im.src='/frame.jpg?'+Date.now()}
setInterval(()=>{if(Date.now()-last>3000){busy=false;next()}},1000);next();
setInterval(async()=>{try{document.getElementById('s').textContent=JSON.stringify(await (await fetch('/stats')).json())}catch(e){}},1000)
</script>""", "text/html")

    def do_POST(self):
        if self.path != "/zones":
            return self.reply({"error": "not found"})
        z = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        z.setdefault("exclude", []); z.setdefault("zones", {})
        if not z.get("include"):
            z["include"] = [[[0, 0], [1, 0], [1, 1], [0, 1]]]
        os.makedirs(os.path.dirname(a.zones), exist_ok=True)
        json.dump(z, open(a.zones, "w"), ensure_ascii=False, indent=1)
        with zlock:
            Z.clear(); Z.update(z)
        log(f"구역 저장 {a.zones}: 포함 {len(z['include'])}, 제외 {len(z['exclude'])}, 구역 {list(z['zones'])}")
        self.reply("saved " + a.zones, "text/plain")


rd = LatestReader(a.src, log=log); rd.start()
log(f"입력 {safe_name(a.src)}, 엔진 {os.path.basename(ENGINE)}, 타일 {a.tiles} 겹침 {a.overlap}, 기준점 {a.anchor}, 보정 {a.stat} {a.window}s, MQTT {a.topic}")
threading.Thread(target=loop, daemon=True).start()
if a.port:
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
else:
    while True:
        time.sleep(60)
