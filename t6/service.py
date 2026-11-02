"""과제 6: A1 재실 인원 + A4 근접 경고를 한 서비스로.
- A1: IP 카메라(rtsp:stream1) → yolo11n fp16 960, 2x2 타일 + NMS + 포함 억제 → 구역 JSON(발 기준) → 최근 --window초 median
- A4: USB 스테레오(2.4 왼쪽, 2.2 오른쪽) → 정류 → yolo11n fp16 640(왼쪽) + SGBM → 몸통 영역 시차(무늬 게이트, 유효 비율) → 사람별 1초 중앙값
       → 가장 가까운 사람 기준 상태: near(< --near m) / mid / far(> --far m) / none(사람 없음) / unknown(측정 불가)
- 카메라마다 읽기 스레드가 따로 있고 끊기면 스스로 다시 연결(camlib.LatestReader). 웹캠이 없거나 뽑혀도 서비스는 계속 돌고 status=offline
- MQTT 127.0.0.1:1883 (발행만): {prefix}/occupancy (retain, --occ-every초마다 + 값이 바뀔 때), {prefix}/proximity (retain, 상태가 바뀔 때 + 60초마다),
  {prefix}/status/<cam> (retain, 바뀔 때), {prefix}/status/service (retain, LWT offline)
- REST :--port  GET /v1/occupancy /v1/proximity /v1/health (ts, age_s, confidence, 카메라 상태)
- 이미지는 디스크에 한 장도 저장하지 않음. --ops이면 메모리 화면(/debug/*.jpg)도 끔
사용: ~/eg2/bin/python service.py [--ops] [--port 8030]"""
import argparse, collections, json, math, os, sys, threading, time
import urllib.parse as up
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
A = os.path.expanduser("~/eg2-lab/applab")
sys.path.insert(0, os.path.join(A, "t0"))
os.environ["YOLO_AUTOINSTALL"] = "False"
import cv2, numpy as np, torch, torchvision
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from camlib import LatestReader, safe_name

ap = argparse.ArgumentParser()
ap.add_argument("--ops", action="store_true", help="운영 모드: 이미지 저장 없음 + 디버그 화면 끔")
ap.add_argument("--port", type=int, default=8030)
ap.add_argument("--no-stereo", action="store_true", help="A4 끔: 웹캠을 열지 않음 (팀원과 웹캠 사용을 정하기 전)")
ap.add_argument("--prefix", default="jiwon934/t6")
ap.add_argument("--mqtt", default="127.0.0.1:1883")
# A1
ap.add_argument("--src", default="rtsp:stream1")
ap.add_argument("--occ-engine", default=os.path.join(A, "t3/yolo11n_fp16_960.engine"))
ap.add_argument("--zones", default=os.path.join(A, "t3/zones/room.json"))
ap.add_argument("--occ-fps", type=float, default=5); ap.add_argument("--occ-conf", type=float, default=0.35)
ap.add_argument("--contain", type=float, default=0.6); ap.add_argument("--window", type=float, default=10)
ap.add_argument("--occ-every", type=float, default=10)
# A4
ap.add_argument("--left", default="2.4"); ap.add_argument("--right", default="2.2")
ap.add_argument("--calib", default=os.path.join(A, "runs/t5_calib2/stereo_25mm.npz"))
ap.add_argument("--prox-engine", default=os.path.join(A, "t2/yolo11n_fp16.engine"))
ap.add_argument("--prox-fps", type=float, default=8); ap.add_argument("--prox-conf", type=float, default=0.4)
ap.add_argument("--near", type=float, default=1.0); ap.add_argument("--far", type=float, default=3.0)
ap.add_argument("--hyst", type=float, default=0.1, help="상태 경계에서 이만큼(m) 넘어야 상태를 바꿈")
ap.add_argument("--hold", type=float, default=2.0, help="새 상태가 이 시간(초) 동안 이어져야 바꿈 (경계 근처에서 1초에 여러 번 바뀌는 것 방지). 웹캠 끊김은 바로 반영")
ap.add_argument("--zmin", type=float, default=0.8); ap.add_argument("--zmax", type=float, default=6.0)
ap.add_argument("--tex", type=float, default=4); ap.add_argument("--min-valid", type=float, default=0.3)
a = ap.parse_args()
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
cv2.setNumThreads(3)
GPU = threading.Lock()          # 두 엔진이 한 CUDA 컨텍스트를 같이 쓰므로 추론은 한 번에 하나씩
T0 = time.time()

# ---------- MQTT ----------
host, port = a.mqtt.split(":")
mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"jiwon934-t6-{os.getpid()}")
mq.will_set(f"{a.prefix}/status/service", json.dumps({"state": "offline"}), qos=1, retain=True)
mq.reconnect_delay_set(1, 30)
try: mq.connect_async(host, int(port), 30); mq.loop_start()
except Exception as e: log(f"MQTT 연결 실패 {e!r} (REST만)")
def pub(topic, obj): mq.publish(f"{a.prefix}/{topic}", json.dumps(obj, ensure_ascii=False), qos=1, retain=True)
pub("status/service", {"state": "online", "ts": round(time.time(), 3), "ops": a.ops})

# ---------- 카메라 ----------
dev = lambda p: f"/dev/v4l/by-path/platform-3610000.usb-usb-0:{p}:1.0-video-index0"
CAMS = {"ipcam": LatestReader(a.src, log=log)}
if not a.no_stereo: CAMS.update(left=LatestReader(dev(a.left), 1280, 720, 30, log=log), right=LatestReader(dev(a.right), 1280, 720, 30, log=log))
for r in CAMS.values(): r.start()
CAM_STATE = {}
def cam_status():
    out = {}
    for k, r in CAMS.items():
        _, ts, _, seq = r.latest(); age = time.time() - ts if ts else None
        out[k] = {"online": age is not None and age < 3.0, "age_s": None if age is None else round(age, 2), "reconnects": r.reconnects, "src": safe_name(r.url)}
    return out
def status_loop():
    while True:
        for k, s in cam_status().items():
            if CAM_STATE.get(k) != s["online"]:
                CAM_STATE[k] = s["online"]; pub(f"status/{k}", {"state": "online" if s["online"] else "offline", "ts": round(time.time(), 3)})
                log(f"카메라 {k}: {'online' if s['online'] else 'offline'}")
        time.sleep(1)

# ---------- A1 재실 인원 ----------
det_occ = YOLO(a.occ_engine, task="detect")
Z = json.load(open(a.zones))
OCC = {"ts": None, "count": None, "raw": None, "zones": {}, "confidence": None, "fps": 0.0, "infer_ms": None}
def inside(poly, p, w, h): return cv2.pointPolygonTest((np.array(poly) * (w, h)).astype(np.int32), (float(p[0]), float(p[1])), False) >= 0
def detect_occ(f):
    h, w = f.shape[:2]; tw, th = w / 2, h / 2; ox, oy = tw * 0.15, th * 0.15
    T = [(int(max(0, c * tw - ox)), int(max(0, r * th - oy)), int(min(w, (c + 1) * tw + ox)), int(min(h, (r + 1) * th + oy))) for r in range(2) for c in range(2)]
    with GPU: res = det_occ.predict([f[y0:y1, x0:x1] for x0, y0, x1, y1 in T], classes=[0], conf=a.occ_conf, imgsz=960, device=0, verbose=False)
    B, S = [], []
    for (x0, y0, _, _), q in zip(T, res):
        for (x1, y1, x2, y2), s in zip(q.boxes.xyxy.tolist(), q.boxes.conf.tolist()): B.append([x1 + x0, y1 + y0, x2 + x0, y2 + y0]); S.append(s)
    if not B: return [], []
    keep = torchvision.ops.nms(torch.tensor(B), torch.tensor(S), 0.5).tolist()
    area = lambda q: (q[2] - q[0]) * (q[3] - q[1]); kept = []
    for i in sorted(keep, key=lambda i: -area(B[i])):   # 포함 억제: 타일 경계에서 잘린 부분 검출 제거
        p = B[i]
        if all(max(0, min(p[2], q[2]) - max(p[0], q[0])) * max(0, min(p[3], q[3]) - max(p[1], q[1])) <= a.contain * max(1e-6, area(p)) for q in (B[j] for j in kept)): kept.append(i)
    return [B[i] for i in kept], [S[i] for i in kept]
def occ_loop():
    hist = collections.deque(); last_seq, last_pub, last_cnt, n, t_fps = -1, 0, None, 0, time.time()
    while True:
        t0 = time.time()
        f, ts, _, seq = CAMS["ipcam"].latest()
        if f is None or seq == last_seq or time.time() - ts > 3:
            time.sleep(0.2); continue
        last_seq = seq; h, w = f.shape[:2]
        ti = time.time(); boxes, scores = detect_occ(f); OCC["infer_ms"] = round((time.time() - ti) * 1000, 1)
        feet = [((b[0] + b[2]) / 2, b[3]) for b in boxes]
        ok = [i for i, p in enumerate(feet) if any(inside(q, p, w, h) for q in Z["include"]) and not any(inside(q, p, w, h) for q in Z["exclude"])]
        raw = len(ok); now = time.time(); hist.append((now, raw))
        while hist and now - hist[0][0] > a.window: hist.popleft()
        cnt = int(np.floor(np.median([x[1] for x in hist]) + 0.5))
        zc = {k: sum(1 for i in ok if inside(poly, feet[i], w, h)) for k, poly in Z.get("zones", {}).items()}
        n += 1
        if now - t_fps >= 5: OCC["fps"] = round(n / (now - t_fps), 2); n, t_fps = 0, now
        OCC.update(ts=round(ts, 3), count=cnt, raw=raw, zones=zc, confidence=round(float(np.mean([scores[i] for i in ok])), 3) if ok else None)
        if cnt != last_cnt or now - last_pub >= a.occ_every:
            pub("occupancy", {"ts": OCC["ts"], "room": Z.get("room"), "count": cnt, "raw": raw, "zones": zc, "capacity": Z.get("capacity"), "window_s": a.window})
            last_pub, last_cnt = now, cnt
        time.sleep(max(0, 1 / a.occ_fps - (time.time() - t0)))

# ---------- A4 근접 경고 ----------
PROX = {"ts": None, "state": "disabled" if a.no_stereo else "unknown", "nearest_m": None, "people": [], "confidence": None, "fps": 0.0, "reason": "--no-stereo" if a.no_stereo else "시작 중"}
if not a.no_stereo: det_prox = YOLO(a.prox_engine, task="detect")
C = np.load(a.calib); W, H, s = 1280, 720, 0.5; w2, h2 = 640, 360; Sm = np.diag([s, s, 1.0]); P1, P2 = Sm @ C["P1"], Sm @ C["P2"]
mL = cv2.initUndistortRectifyMap(C["KL"], C["DL"], C["R1"], P1, (w2, h2), cv2.CV_16SC2); mR = cv2.initUndistortRectifyMap(C["KR"], C["DR"], C["R2"], P2, (w2, h2), cv2.CV_16SC2)
mLf = cv2.initUndistortRectifyMap(C["KL"], C["DL"], C["R1"], C["P1"], (W, H), cv2.CV_16SC2)
FB = float(P1[0, 0]) * abs(float(P2[0, 3] / P2[0, 0])); MIN_D = max(0, int(FB / a.zmax) - 2); NUM_D = int(math.ceil((FB / a.zmin - MIN_D + 2) / 16)) * 16
SG = cv2.StereoSGBM_create(minDisparity=MIN_D, numDisparities=NUM_D, blockSize=5, P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i; return i / u if u > 0 else 0
def classify(z, prev):
    """히스테리시스: 경계에서 --hyst m 넘게 들어와야 상태가 바뀜"""
    hy = a.hyst if prev in ("near", "mid", "far") else 0
    if prev == "near" and z < a.near + hy: return "near"
    if prev == "far" and z > a.far - hy: return "far"
    if prev == "mid" and a.near - hy <= z <= a.far + hy: return "mid"
    return "near" if z < a.near else "far" if z > a.far else "mid"
def prox_loop():
    tracks, nid, last_state, last_pub, n, t_fps = {}, 1, None, 0, 0, time.time()
    pend, pend_t = None, 0.0
    while True:
        t0 = time.time()
        fl, tl, _, _ = CAMS["left"].latest(); fr, tr, _, _ = CAMS["right"].latest()
        if fl is None or fr is None or time.time() - min(tl, tr) > 3:
            PROX.update(ts=round(time.time(), 3), state="unknown", nearest_m=None, people=[], confidence=None, reason="웹캠 없음/끊김")
            if last_state != "unknown": pub("proximity", {k: PROX[k] for k in ("ts", "state", "nearest_m", "reason")}); last_state = "unknown"
            time.sleep(1); continue
        gl = cv2.remap(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), *mL, cv2.INTER_LINEAR); gr = cv2.remap(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), *mR, cv2.INTER_LINEAR)
        d = SG.compute(gl, gr).astype(np.float32) / 16; g = gl.astype(np.float32)
        good = (d > MIN_D) & (np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2, 0)) > a.tex)
        with GPU: r = det_prox.predict(cv2.remap(fl, *mLf, cv2.INTER_LINEAR), classes=[0], conf=a.prox_conf, imgsz=640, device=0, verbose=False)[0]
        now = time.time(); people = []
        for k in [k for k, t in tracks.items() if now - t["last"] > 1.0]: del tracks[k]
        for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            b = [x1 * s, y1 * s, x2 * s, y2 * s]
            k = max(tracks, key=lambda k: iou(b, tracks[k]["box"]), default=None)
            if k is None or iou(b, tracks[k]["box"]) < 0.3: k, nid = nid, nid + 1; tracks[k] = {"hist": collections.deque()}
            tracks[k].update(box=b, last=now)
            bw, bh = b[2] - b[0], b[3] - b[1]
            tx0, ty0, tx1, ty1 = int(b[0] + .3 * bw), int(b[1] + .2 * bh), int(b[2] - .3 * bw), int(b[1] + .55 * bh)
            m = good[max(0, ty0):ty1, max(0, tx0):tx1]; v = float(m.mean()) if m.size else 0.0
            z = float(np.median(FB / d[max(0, ty0):ty1, max(0, tx0):tx1][m])) if m.size and v >= a.min_valid and m.sum() >= 15 else None
            hs = tracks[k]["hist"]; hs.append((now, z))
            while hs and now - hs[0][0] > 1.0: hs.popleft()
            zs = [x[1] for x in hs if x[1] is not None]
            people.append({"id": k, "z_m": round(float(np.median(zs)), 3) if zs else None, "valid": round(v, 2), "conf": round(cf, 2)})
        meas = [p for p in people if p["z_m"] is not None]
        if not people: cand, nearest, reason = "none", None, "사람 없음"
        elif not meas: cand, nearest, reason = "unknown", None, "측정 불가(무늬·유효 비율 부족)"
        else: nearest = min(p["z_m"] for p in meas); cand, reason = classify(nearest, last_state), ""
        if last_state in (None, "unknown") and PROX["reason"] == "웹캠 없음/끊김": last_state = None   # 웹캠 복구 직후엔 바로 반영
        if cand == last_state or last_state is None: state, pend = cand, None
        elif pend != cand: state, pend, pend_t = last_state, cand, now
        elif now - pend_t >= a.hold: state, pend = cand, None
        else: state = last_state
        n += 1
        if now - t_fps >= 5: PROX["fps"] = round(n / (now - t_fps), 2); n, t_fps = 0, now
        PROX.update(ts=round(min(tl, tr), 3), state=state, nearest_m=nearest, people=people, confidence=round(max(p["valid"] for p in meas), 2) if meas else None, reason=reason, skew_ms=round(abs(tl - tr) * 1000, 1))
        if state != last_state or now - last_pub >= 60:
            pub("proximity", {"ts": PROX["ts"], "state": state, "nearest_m": nearest, "people": len(people), "near_m": a.near, "far_m": a.far, "reason": reason})
            if state != last_state: log(f"근접 상태 {last_state} → {state} ({nearest} m)")
            last_state, last_pub = state, now
        time.sleep(max(0, 1 / a.prox_fps - (time.time() - t0)))

# ---------- REST ----------
def age(ts): return None if ts is None else round(time.time() - ts, 2)
class Hd(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def j(self, o, code=200):
        b = json.dumps(o, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = up.urlparse(self.path).path
        if u == "/v1/occupancy": return self.j({**OCC, "age_s": age(OCC["ts"]), "room": Z.get("room"), "capacity": Z.get("capacity"), "camera": cam_status()["ipcam"]})
        if u == "/v1/proximity":
            cs = cam_status(); return self.j({**PROX, "age_s": age(PROX["ts"]), "near_m": a.near, "far_m": a.far, "cameras": {k: cs[k] for k in ("left", "right") if k in cs}})
        if u == "/v1/health":
            cs = cam_status(); ok = all(c["online"] for c in cs.values())
            return self.j({"status": "ok" if ok else "degraded", "uptime_s": round(time.time() - T0), "ops": a.ops, "mqtt": mq.is_connected(), "cameras": cs,
                           "occupancy": {"age_s": age(OCC["ts"]), "fps": OCC["fps"], "infer_ms": OCC["infer_ms"]}, "proximity": {"age_s": age(PROX["ts"]), "fps": PROX["fps"], "state": PROX["state"]}})
        self.j({"error": "not found", "endpoints": ["/v1/occupancy", "/v1/proximity", "/v1/health"]}, 404)

for fn in (status_loop, occ_loop) + (() if a.no_stereo else (prox_loop,)): threading.Thread(target=fn, daemon=True).start()
log(f"t6 서비스 :{a.port} ops={a.ops} MQTT {a.mqtt} {a.prefix}/*, A1 {safe_name(a.src)}, A4 {a.left}/{a.right}, near {a.near} far {a.far}")
ThreadingHTTPServer(("0.0.0.0", a.port), Hd).serve_forever()
