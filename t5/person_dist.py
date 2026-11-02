"""과제 5-6: 사람별 거리. 정류된 왼쪽 영상에서 YOLO(yolo11n, TensorRT fp16)로 사람을 찾고, 상자 안 몸통 영역의 SGBM 시차로 거리를 잼.
- 몸통 영역: 상자 가로 가운데 40%, 세로 20~55% (머리·팔·배경을 덜 섞으려고)
- 무늬 게이트(7x7 밝기 표준편차 ≤ --tex 픽셀 제외)를 거친 유효 비율이 --min-valid보다 낮으면 "측정 불가"
- IoU로 간단히 추적해서 사람마다 ID, 최근 --avg-s초 동안의 Z 중앙값을 표시
- 기록: runs/t5_person_dist.csv (매 --log-every초, 사람마다 한 줄). /sample?name=사람3칸&sec=5 → runs/t5_person_tests.csv에 요약 + 화면을 runs/t5_person/에 저장 (Jetson 밖으로 내보내지 않음)
- 과제 3 로거와 GPU를 같이 쓰므로 검출은 --det-fps로 제한하고 nice로 실행
사용: nice -n 10 ~/eg2/bin/python person_dist.py [--port 8031]"""
import argparse, collections, csv, json, math, os, subprocess, threading, time
import urllib.parse as up
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np
from ultralytics import YOLO

R0 = os.path.expanduser("~/eg2-lab/applab/runs")
ap = argparse.ArgumentParser()
ap.add_argument("--left", default="2.4"); ap.add_argument("--right", default="2.2")
ap.add_argument("--calib", default=os.path.join(R0, "t5_calib2/stereo_25mm.npz"))
ap.add_argument("--engine", default=os.path.expanduser("~/eg2-lab/applab/t2/yolo11n_fp16.engine"))
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--det-fps", type=float, default=8, help="검출 최대 횟수/초 (과제 3 로거와 GPU 나눠 쓰기)")
ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--zmin", type=float, default=0.8); ap.add_argument("--zmax", type=float, default=6.0)
ap.add_argument("--block", type=int, default=5)
ap.add_argument("--tex", type=float, default=4, help="무늬 게이트: 7x7 밝기 표준편차가 이 값 이하인 픽셀은 시차를 안 씀")
ap.add_argument("--min-valid", type=float, default=0.3, help="몸통 영역 유효 비율이 이보다 낮으면 측정 불가")
ap.add_argument("--avg-s", type=float, default=1.0, help="사람마다 이 시간 동안의 Z 중앙값")
ap.add_argument("--log-every", type=float, default=0.5)
ap.add_argument("--focus", type=int, default=0); ap.add_argument("--exposure", type=int, default=60); ap.add_argument("--gain", type=int, default=30)
ap.add_argument("--port", type=int, default=8031)
a = ap.parse_args()
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
cv2.setNumThreads(3)
OUT_DIR = os.path.join(R0, "t5_person"); os.makedirs(OUT_DIR, exist_ok=True)

dev = lambda p: f"/dev/v4l/by-path/platform-3610000.usb-usb-0:{p}:1.0-video-index0"
def v4l(p, *cs):
    for c in cs:
        r = subprocess.run(["v4l2-ctl", "-d", dev(p), "-c", c], capture_output=True, text=True)
        if r.returncode: log(f"[{p}] {c} 실패: {r.stderr.strip()[:80]}")
for p in (a.left, a.right):
    v4l(p, "focus_automatic_continuous=0", f"focus_absolute={a.focus}", "auto_exposure=1", f"exposure_time_absolute={a.exposure}",
        f"gain={a.gain}", "white_balance_automatic=0", "white_balance_temperature=4500", "power_line_frequency=2", "exposure_dynamic_framerate=0")

W, H = 1280, 720
C = np.load(a.calib)
w, h = int(round(W * a.scale)), int(round(H * a.scale))
S = np.diag([a.scale, a.scale, 1.0])
P1, P2 = S @ C["P1"], S @ C["P2"]
mL = cv2.initUndistortRectifyMap(C["KL"], C["DL"], C["R1"], P1, (w, h), cv2.CV_16SC2)
mR = cv2.initUndistortRectifyMap(C["KR"], C["DR"], C["R2"], P2, (w, h), cv2.CV_16SC2)
mLf = cv2.initUndistortRectifyMap(C["KL"], C["DL"], C["R1"], C["P1"], (W, H), cv2.CV_16SC2)   # 검출용 원본 크기 정류
F = float(P1[0, 0]); B = abs(float(P2[0, 3] / P2[0, 0])); FB = F * B
MIN_D = max(0, int(math.floor(FB / a.zmax)) - 2)
NUM_D = int(math.ceil((FB / a.zmin - MIN_D + 2) / 16.0)) * 16
bs = a.block
SG = cv2.StereoSGBM_create(minDisparity=MIN_D, numDisparities=NUM_D, blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs,
                           disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
log(f"정류 {w}x{h}, f {F:.1f} px, B {B * 1000:.1f} mm, 시차 {MIN_D}~{MIN_D + NUM_D} px (→ {FB / (MIN_D + NUM_D):.2f}~{FB / max(MIN_D, 1):.1f} m)")
det = YOLO(a.engine, task="detect")


class Cam(threading.Thread):
    def __init__(self, p):
        super().__init__(daemon=True)
        self.c = cv2.VideoCapture(dev(p), cv2.CAP_V4L2)
        self.c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); self.c.set(3, W); self.c.set(4, H); self.c.set(5, 30)
        self.c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.f, self.t = None, 0.0
    def run(self):
        while True:
            ok, f = self.c.read()
            if ok: self.f, self.t = f, time.time()


L, R = Cam(a.left), Cam(a.right); L.start(); R.start()
LOCK = threading.Lock()
TRACKS = {}          # id → {"box", "last", "hist": deque[(t, Z or None, valid)]}
NEXT = {"id": 1}
FR = collections.deque(maxlen=2000)   # (t, id, z_raw, valid, z_med) /sample용
VIEW = {"jpg": None, "snap": None, "stat": {}}


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    i = ix * iy; u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i
    return i / u if u > 0 else 0


def match(boxes, now):
    """탐욕적 IoU 매칭. 1초 넘게 안 보인 트랙은 지움"""
    for k in [k for k, t in TRACKS.items() if now - t["last"] > 1.0]: del TRACKS[k]
    pairs = sorted(((iou(b, t["box"]), i, k) for i, b in enumerate(boxes) for k, t in TRACKS.items()), reverse=True)
    used_b, used_t, ids = set(), set(), [None] * len(boxes)
    for v, i, k in pairs:
        if v < 0.3: break
        if i in used_b or k in used_t: continue
        ids[i] = k; used_b.add(i); used_t.add(k)
    for i in range(len(boxes)):
        if ids[i] is None:
            ids[i] = NEXT["id"]; NEXT["id"] += 1
            TRACKS[ids[i]] = {"hist": collections.deque()}
        TRACKS[ids[i]].update(box=boxes[i], last=now)
    return ids


def torso(b):
    x1, y1, x2, y2 = b; bw, bh = x2 - x1, y2 - y1
    return x1 + 0.3 * bw, y1 + 0.2 * bh, x2 - 0.3 * bw, y1 + 0.55 * bh


LOGF = os.path.join(R0, "t5_person_dist.csv")
LCOLS = ["ts", "time", "id", "z_raw_m", "z_med_m", "valid_frac", "status", "box_x1", "box_y1", "box_x2", "box_y2", "conf"]
if not os.path.exists(LOGF): csv.writer(open(LOGF, "w", newline="")).writerow(LCOLS)
LAST_LOG = {}


def loop():
    last_det, dets = 0.0, []
    while True:
        if L.f is None or R.f is None: time.sleep(0.05); continue
        t0 = time.time()
        fl, fr = L.f, R.f
        gl0, gr0 = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gl = cv2.remap(gl0, mL[0], mL[1], cv2.INTER_LINEAR); gr = cv2.remap(gr0, mR[0], mR[1], cv2.INTER_LINEAR)
        disp = SG.compute(gl, gr).astype(np.float32) / 16.0
        g = gl.astype(np.float32); sd = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2, 0))
        good = (disp > MIN_D) & (sd > a.tex)
        now = time.time()
        if now - last_det >= 1.0 / a.det_fps:
            rect = cv2.remap(fl, mLf[0], mLf[1], cv2.INTER_LINEAR)
            r = det.predict(rect, classes=[0], conf=a.conf, imgsz=640, device=0, verbose=False)[0]
            dets = [(bx, c) for bx, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
            last_det = now
        boxes = [[v * a.scale for v in bx] for bx, _ in dets]
        out = []
        with LOCK:
            ids = match(boxes, now)
            for (bx, cf), b, k in zip(dets, boxes, ids):
                x0, y0, x1, y1 = [int(round(v)) for v in torso(b)]
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                m = good[y0:y1, x0:x1]; v = float(m.mean()) if m.size else 0.0
                z = float(np.median(FB / disp[y0:y1, x0:x1][m])) if m.size and v >= a.min_valid and m.sum() >= 15 else None
                hs = TRACKS[k]["hist"]; hs.append((now, z, v))
                while hs and now - hs[0][0] > a.avg_s: hs.popleft()
                zs = [x[1] for x in hs if x[1] is not None]
                zm = float(np.median(zs)) if len(zs) >= max(1, len(hs) // 2) else None
                FR.append((now, k, z, v, zm))
                out.append((k, b, (x0, y0, x1, y1), z, zm, v, cf))
        # 기록
        for k, b, _, z, zm, v, cf in out:
            if now - LAST_LOG.get(k, 0) >= a.log_every:
                LAST_LOG[k] = now
                with open(LOGF, "a", newline="") as fh:
                    csv.writer(fh).writerow([round(now, 3), time.strftime("%H:%M:%S"), k, "" if z is None else round(z, 4), "" if zm is None else round(zm, 4),
                                             round(v, 3), "ok" if zm is not None else "측정불가"] + [round(x / a.scale, 1) for x in b] + [round(cf, 3)])
        # 화면: 정류 왼쪽 | 시차(게이트 후)
        vl = cv2.cvtColor(gl, cv2.COLOR_GRAY2BGR)
        dv = np.clip((disp - MIN_D) / NUM_D * 255, 0, 255).astype(np.uint8); dc = cv2.applyColorMap(dv, cv2.COLORMAP_JET); dc[~good] = 0
        for k, b, tb, z, zm, v, cf in out:
            col = (0, 255, 0) if zm is not None else (0, 0, 255)
            for im in (vl, dc):
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
                cv2.rectangle(im, tb[:2], tb[2:], (255, 255, 0), 1)
            txt = f"#{k} {zm:.2f} m" if zm is not None else f"#{k} 측정 불가".replace("측정 불가", "N/A")
            cv2.putText(vl, txt, (int(b[0]), max(16, int(b[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(vl, f"valid {v * 100:.0f}%", (int(b[0]), int(b[3]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        img = np.hstack([vl, dc])
        VIEW["snap"] = img
        VIEW["jpg"] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        VIEW["stat"] = {"ms": round((time.time() - t0) * 1000), "people": [{"id": k, "z_med_m": None if zm is None else round(zm, 3),
                        "z_raw_m": None if z is None else round(z, 3), "valid": round(v, 2)} for k, _, _, z, zm, v, _ in out]}


TCOLS = ["time", "name", "sec", "id", "frames", "measured_frames", "z_median_m", "z_sd_m", "valid_mean", "note"]


def sample(name, sec, note):
    t0 = time.time(); time.sleep(sec)
    with LOCK: it = [x for x in FR if x[0] >= t0]
    rows = []
    for k in sorted({x[1] for x in it}):
        xs = [x for x in it if x[1] == k]; zs = np.array([x[2] for x in xs if x[2] is not None])
        rows.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "sec": sec, "id": k, "frames": len(xs), "measured_frames": len(zs),
                     "z_median_m": round(float(np.median(zs)), 4) if len(zs) else "", "z_sd_m": round(float(zs.std()), 4) if len(zs) else "",
                     "valid_mean": round(float(np.mean([x[3] for x in xs])), 3), "note": note})
    path = os.path.join(R0, "t5_person_tests.csv"); new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, TCOLS)
        if new: wr.writeheader()
        wr.writerows(rows)
    if VIEW["snap"] is not None: cv2.imwrite(os.path.join(OUT_DIR, f"{name}_{time.strftime('%H%M%S')}.jpg"), VIEW["snap"])
    log(f"시험 {name}: " + ", ".join(f"#{r['id']} {r['z_median_m']} m ({r['measured_frames']}/{r['frames']})" for r in rows))
    return rows


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>t5 person distance</title>
<body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'><img id=v style='width:100%'><h2 id=s style='margin:6px'></h2><script>
const v=document.getElementById('v');let last=Date.now(),busy=false;
function next(){if(busy)return;busy=true;const im=new Image();im.onload=()=>{v.src=im.src;last=Date.now();busy=false;setTimeout(next,150)};
 im.onerror=()=>{busy=false;setTimeout(next,1000)};im.src='/frame.jpg?'+Date.now()}
setInterval(()=>{if(Date.now()-last>3000){busy=false;next()}},1000);next();
setInterval(async()=>{try{const s=await (await fetch('/stats')).json();document.getElementById('s').textContent=
 s.people.length?s.people.map(p=>`#${p.id} ${p.z_med_m===null?'측정 불가':p.z_med_m.toFixed(2)+' m'} (유효 ${Math.round(p.valid*100)}%)`).join(' · ')+` · ${s.ms} ms`:'사람 없음'}catch(e){}},500)
</script>""".encode()


class Hd(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def send(self, body, ct="application/json"):
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = up.urlparse(self.path); q = {k: v[0] for k, v in up.parse_qs(u.query).items()}
        j = lambda o: self.send(json.dumps(o, ensure_ascii=False).encode())
        if u.path == "/frame.jpg" and VIEW["jpg"]: return self.send(VIEW["jpg"], "image/jpeg")
        if u.path == "/stats": return j(VIEW["stat"])
        if u.path == "/sample": return j(sample(q.get("name", "test"), float(q.get("sec", 5)), q.get("note", "")))
        self.send(PAGE, "text/html")


threading.Thread(target=loop, daemon=True).start()
log(f"사람 거리 :{a.port}  왼쪽 {a.left}, 오른쪽 {a.right}, 기록 → {LOGF}")
ThreadingHTTPServer(("0.0.0.0", a.port), Hd).serve_forever()
