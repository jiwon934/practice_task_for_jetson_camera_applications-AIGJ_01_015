"""과제 5-4: 스테레오 거리 측정 시험.
- 보정 결과(stereo_25mm.npz)로 두 영상을 정류하고 SGBM으로 시차를 구해, 화면 가운데 ROI(클릭으로 옮김)의 중앙값 깊이를 ~2초 평균으로 보여 줌.
- 깊이 Z = f·B/d (렌즈 평면에서 수직 거리). 시차 범위(min/num-disp)는 --zmin~--zmax에 맞게 자동으로 정함.
- 실제 거리를 넣고 "기록"을 누르면(또는 /rec?true=1.35) 실제 거리와 측정 통계를 --out CSV에 한 줄 추가, 그때 화면은 --snap-dir에 저장.
- 노출·초점·화이트밸런스는 capture_pairs.py처럼 고정 (v4l2-ctl).
사용: ~/eg2/bin/python measure_dist.py [--scale 0.5] [--zmin 0.8 --zmax 4.5] [--port 8031]"""
import argparse, collections, csv, json, math, os, subprocess, threading, time
import urllib.parse as up
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

R0 = os.path.expanduser("~/eg2-lab/applab/runs")
ap = argparse.ArgumentParser()
ap.add_argument("--left", default="2.4"); ap.add_argument("--right", default="2.2")
ap.add_argument("--calib", default=os.path.join(R0, "t5_calib2/stereo_25mm.npz"))
ap.add_argument("--scale", type=float, default=0.5, help="SGBM에 넣는 영상 배율 (1280x720 기준). 0.5 = 640x360")
ap.add_argument("--zmin", type=float, default=0.8); ap.add_argument("--zmax", type=float, default=4.5)
ap.add_argument("--block", type=int, default=5, help="SGBM blockSize (홀수)")
ap.add_argument("--roi", type=float, default=0.08, help="ROI 한 변 (영상 너비 대비 비율)")
ap.add_argument("--avg-s", type=float, default=2.0, help="ROI 깊이를 이 시간(초) 동안 평균")
ap.add_argument("--focus", type=int, default=0); ap.add_argument("--exposure", type=int, default=60); ap.add_argument("--gain", type=int, default=30)
ap.add_argument("--out", default=os.path.join(R0, "t5_dist.csv"))
ap.add_argument("--snap-dir", default=os.path.join(R0, "t5_dist_snaps"))
ap.add_argument("--port", type=int, default=8031)
a = ap.parse_args()
os.makedirs(a.snap_dir, exist_ok=True)
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
cv2.setNumThreads(4)

dev = lambda p: f"/dev/v4l/by-path/platform-3610000.usb-usb-0:{p}:1.0-video-index0"
def v4l(p, *cs):
    for c in cs:
        r = subprocess.run(["v4l2-ctl", "-d", dev(p), "-c", c], capture_output=True, text=True)
        if r.returncode: log(f"[{p}] {c} 실패: {r.stderr.strip()[:80]}")
for p in (a.left, a.right):
    v4l(p, "focus_automatic_continuous=0", f"focus_absolute={a.focus}", "auto_exposure=1", f"exposure_time_absolute={a.exposure}",
        f"gain={a.gain}", "white_balance_automatic=0", "white_balance_temperature=4500", "power_line_frequency=2", "exposure_dynamic_framerate=0")

# 정류 맵: P1/P2를 배율만큼 줄여서 바로 작은 크기로 정류
W, H = 1280, 720
C = np.load(a.calib)
w, h = int(round(W * a.scale)), int(round(H * a.scale))
S = np.diag([a.scale, a.scale, 1.0])
P1, P2 = S @ C["P1"], S @ C["P2"]
mL = cv2.initUndistortRectifyMap(C["KL"], C["DL"], C["R1"], P1, (w, h), cv2.CV_16SC2)
mR = cv2.initUndistortRectifyMap(C["KR"], C["DR"], C["R2"], P2, (w, h), cv2.CV_16SC2)
F = float(P1[0, 0]); B = abs(float(P2[0, 3] / P2[0, 0]))   # f(px, 배율 적용), 기준선(m)
FB = F * B
MIN_D = max(0, int(math.floor(FB / a.zmax)) - 2)
NUM_D = int(math.ceil((FB / a.zmin - MIN_D + 2) / 16.0)) * 16
bs = a.block
SG = cv2.StereoSGBM_create(minDisparity=MIN_D, numDisparities=NUM_D, blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs,
                           disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
log(f"정류 {w}x{h}, f {F:.1f} px, B {B * 1000:.1f} mm, 시차 {MIN_D}~{MIN_D + NUM_D} px (→ {FB / (MIN_D + NUM_D):.2f}~{FB / max(MIN_D, 1):.1f} m), block {bs}")


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
ROI = {"cx": 0.5, "cy": 0.5, "s": a.roi}     # 정규화 좌표
HIST = collections.deque()                     # (t, Z 중앙값, d 중앙값, 유효 비율, ROI 안 Z IQR)
FR = collections.deque(maxlen=600)             # 프레임별 같은 값 (ROI 바꿔도 안 지움, /sample용)
VIEW = {"jpg": None, "snap": None, "stat": {}, "ms": 0}
LOCK = threading.Lock()


def roi_px():
    s = ROI["s"] * w
    x0, y0 = int(ROI["cx"] * w - s / 2), int(ROI["cy"] * h - s / 2)
    return max(0, x0), max(0, y0), min(w, int(x0 + s)), min(h, int(y0 + s))


def summary():
    with LOCK:
        hs = [x for x in HIST if x[1] is not None]
        n_all = len(HIST)
    if not hs: return {"n": n_all, "valid_frames": 0}
    z = np.array([x[1] for x in hs]); d = np.array([x[2] for x in hs])
    dm = float(d.mean())
    return {"n": n_all, "valid_frames": len(hs), "z_mean_m": round(float(z.mean()), 4), "z_std_m": round(float(z.std()), 4),
            "z_from_dmean_m": round(FB / dm, 4), "d_mean_px": round(dm, 3), "d_full_px": round(dm / a.scale, 2), "d_std_px": round(float(d.std()), 3),
            "valid_frac": round(float(np.mean([x[3] for x in hs])), 3), "roi_iqr_m": round(float(np.mean([x[4] for x in hs])), 4)}


def loop():
    while True:
        if L.f is None or R.f is None: time.sleep(0.05); continue
        t0 = time.time()
        fl, fr = L.f, R.f
        gl = cv2.remap(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), mL[0], mL[1], cv2.INTER_LINEAR)
        gr = cv2.remap(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), mR[0], mR[1], cv2.INTER_LINEAR)
        disp = SG.compute(gl, gr).astype(np.float32) / 16.0
        x0, y0, x1, y1 = roi_px()
        dr = disp[y0:y1, x0:x1]; ok = dr > MIN_D
        frac = float(ok.mean()) if dr.size else 0.0
        now = time.time()
        if ok.sum() >= 20:
            zr = FB / dr[ok]
            item = (now, float(np.median(zr)), float(np.median(dr[ok])), frac, float(np.subtract(*np.percentile(zr, [75, 25]))))
        else:
            item = (now, None, None, frac, None)
        with LOCK:
            HIST.append(item); FR.append(item)
            while HIST and now - HIST[0][0] > a.avg_s: HIST.popleft()
        st = summary()
        # 화면: 정류된 왼쪽 | 시차 컬러맵, ROI 상자와 값
        vl = cv2.cvtColor(gl, cv2.COLOR_GRAY2BGR)
        dv = np.clip((disp - MIN_D) / NUM_D * 255, 0, 255).astype(np.uint8)
        dc = cv2.applyColorMap(dv, cv2.COLORMAP_JET); dc[disp <= MIN_D] = 0
        for im in (vl, dc):
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        txt = f"Z {st['z_mean_m']:.3f} m  sd {st['z_std_m'] * 100:.1f} cm  d {st['d_full_px']:.1f}px  valid {st['valid_frac'] * 100:.0f}%" if "z_mean_m" in st else f"no depth in ROI (valid {frac * 100:.0f}%)"
        cv2.putText(vl, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * w / 640, (0, 255, 0), 2)
        img = np.hstack([vl, dc])
        VIEW["snap"] = img; VIEW["raw"] = (fl, fr)
        VIEW["jpg"] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        VIEW["ms"] = round((time.time() - t0) * 1000)
        st.update(ms=VIEW["ms"], skew_ms=round(abs(L.t - R.t) * 1000, 1), roi=[x0, y0, x1, y1])
        VIEW["stat"] = st


FULL = {}


def rowcheck():
    """보정 유지 확인: 원본 해상도로 정류한 두 영상에서 ORB 특징을 짝지어 세로(행) 차이를 잼. 카메라가 움직였으면 커짐."""
    if not FULL:
        FULL["L"] = cv2.initUndistortRectifyMap(C["KL"], C["DL"], C["R1"], C["P1"], (W, H), cv2.CV_16SC2)
        FULL["R"] = cv2.initUndistortRectifyMap(C["KR"], C["DR"], C["R2"], C["P2"], (W, H), cv2.CV_16SC2)
    gl = cv2.remap(cv2.cvtColor(L.f, cv2.COLOR_BGR2GRAY), *FULL["L"], cv2.INTER_LINEAR)
    gr = cv2.remap(cv2.cvtColor(R.f, cv2.COLOR_BGR2GRAY), *FULL["R"], cv2.INTER_LINEAR)
    orb = cv2.ORB_create(3000)
    kl, dl = orb.detectAndCompute(gl, None); kr, dr = orb.detectAndCompute(gr, None)
    if dl is None or dr is None: return {"n": 0}
    ms = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr)
    pl = np.array([kl[m.queryIdx].pt for m in ms]); pr = np.array([kr[m.trainIdx].pt for m in ms])
    if not len(ms): return {"n": 0}
    dx, dy = pl[:, 0] - pr[:, 0], pl[:, 1] - pr[:, 1]
    ok = (dx > 0) & (dx < FB / a.scale / 0.5) & (np.abs(dy) < 15)
    dy = dy[ok]
    if len(dy) < 10: return {"n": int(len(dy))}
    return {"n": int(len(dy)), "dy_median_px": round(float(np.median(dy)), 3), "abs_dy_median_px": round(float(np.median(np.abs(dy))), 3),
            "abs_dy_p95_px": round(float(np.percentile(np.abs(dy), 95)), 3)}


COLS = ["time", "true_m", "z_mean_m", "err_m", "err_pct", "z_std_m", "z_from_dmean_m", "d_full_px", "d_std_px", "valid_frac", "roi_iqr_m",
        "valid_frames", "pred_dz_0.25px_m", "pred_dz_0.5px_m", "roi_px", "scale", "min_disp", "num_disp", "block", "exposure", "note", "snap"]


def record(true_m, note):
    st = summary()
    if "z_mean_m" not in st: return {"ok": False, "msg": "ROI에 유효한 깊이가 없음"}
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    snap = os.path.join(a.snap_dir, f"dist_{true_m:.2f}m_{time.strftime('%H%M%S')}.jpg")
    if VIEW["snap"] is not None: cv2.imwrite(snap, VIEW["snap"])
    Ff = F / a.scale   # 원본 해상도 기준 f
    row = {"time": ts, "true_m": true_m, "z_mean_m": st["z_mean_m"], "err_m": round(st["z_mean_m"] - true_m, 4),
           "err_pct": round((st["z_mean_m"] - true_m) / true_m * 100, 2), "z_std_m": st["z_std_m"], "z_from_dmean_m": st["z_from_dmean_m"],
           "d_full_px": st["d_full_px"], "d_std_px": st["d_std_px"], "valid_frac": st["valid_frac"], "roi_iqr_m": st["roi_iqr_m"],
           "valid_frames": st["valid_frames"], "pred_dz_0.25px_m": round(true_m ** 2 * 0.25 / (Ff * B), 4),
           "pred_dz_0.5px_m": round(true_m ** 2 * 0.5 / (Ff * B), 4), "roi_px": "%d %d %d %d" % roi_px(), "scale": a.scale,
           "min_disp": MIN_D, "num_disp": NUM_D, "block": bs, "exposure": a.exposure, "note": note, "snap": os.path.basename(snap)}
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        wr = csv.DictWriter(fh, COLS)
        if new: wr.writeheader()
        wr.writerow(row)
    log(f"기록: 실제 {true_m:.2f} m → 측정 {row['z_mean_m']:.3f} m (오차 {row['err_m'] * 100:+.1f} cm, {row['err_pct']:+.1f}%)")
    return {"ok": True, "row": row}


FAIL_DIR = os.path.join(R0, "t5_fail")
FCOLS = ["time", "name", "sec", "frames", "depth_frames", "valid_frac_mean", "valid_frac_min", "z_median_m", "z_sd_m", "z_min_m", "z_max_m",
         "d_full_px", "roi_iqr_m", "roi_px", "exposure", "note"]


def sample(name, sec, note):
    """실패 조건 시험: sec초 동안 프레임별 ROI 값을 모아 runs/t5_fail.csv에 한 줄, 화면·원본 두 장은 runs/t5_fail/에 저장"""
    os.makedirs(FAIL_DIR, exist_ok=True)
    t0 = time.time(); time.sleep(sec)
    with LOCK: it = [x for x in FR if x[0] >= t0]
    zs = np.array([x[1] for x in it if x[1] is not None]); ds = np.array([x[2] for x in it if x[1] is not None])
    fr = np.array([x[3] for x in it]) if it else np.zeros(1)
    base = os.path.join(FAIL_DIR, f"{name}_{time.strftime('%H%M%S')}")
    if VIEW.get("snap") is not None: cv2.imwrite(base + "_view.jpg", VIEW["snap"])
    if VIEW.get("raw"): cv2.imwrite(base + "_L.png", VIEW["raw"][0]); cv2.imwrite(base + "_R.png", VIEW["raw"][1])
    iq = [x[4] for x in it if x[4] is not None]
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "sec": sec, "frames": len(it), "depth_frames": len(zs),
           "valid_frac_mean": round(float(fr.mean()), 3), "valid_frac_min": round(float(fr.min()), 3),
           "z_median_m": round(float(np.median(zs)), 4) if len(zs) else "", "z_sd_m": round(float(zs.std()), 4) if len(zs) else "",
           "z_min_m": round(float(zs.min()), 4) if len(zs) else "", "z_max_m": round(float(zs.max()), 4) if len(zs) else "",
           "d_full_px": round(float(np.median(ds)) / a.scale, 2) if len(ds) else "", "roi_iqr_m": round(float(np.mean(iq)), 4) if iq else "",
           "roi_px": "%d %d %d %d" % roi_px(), "exposure": a.exposure, "note": note}
    path = os.path.join(R0, "t5_fail.csv"); new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, FCOLS)
        if new: wr.writeheader()
        wr.writerow(row)
    row["files"] = os.path.basename(base) + "_{view.jpg,L.png,R.png}"
    log(f"실패 시험 {name}: 유효 {row['valid_frac_mean']}, Z {row['z_median_m']} sd {row['z_sd_m']}")
    return row


def rows():
    if not os.path.exists(a.out): return []
    return list(csv.DictReader(open(a.out)))


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>t5 distance</title>
<body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>
<img id=v style='width:100%;cursor:crosshair'>
<h2 id=s style='margin:6px'></h2>
<p style='margin:6px'>실제 거리(m): <input id=t size=6 value=0.90 style='font-size:1.3em;width:4em'>
 TILES <input id=n size=20 placeholder=메모> <button onclick=rec() style='font-size:1.3em'>기록</button> <span id=m></span></p>
<p style='margin:6px'>ROI: 왼쪽 영상을 클릭해 옮기기 · <button onclick="roi(0,0,0.8)">작게</button> <button onclick="roi(0,0,1.25)">크게</button>
 · 노출 <button onclick="ex(-20)">−</button> <button onclick="ex(20)">+</button> <span id=e></span></p>
<table id=r border=1 cellpadding=3 style='margin:6px;border-collapse:collapse;font-size:.9em'></table><script>
const v=document.getElementById('v');let last=Date.now(),busy=false;
function next(){if(busy)return;busy=true;const im=new Image();im.onload=()=>{v.src=im.src;last=Date.now();busy=false;setTimeout(next,150)};
 im.onerror=()=>{busy=false;setTimeout(next,1000)};im.src='/frame.jpg?'+Date.now()}
setInterval(()=>{if(Date.now()-last>3000){busy=false;next()}},1000);next();
v.onclick=e=>{const r=v.getBoundingClientRect();const x=(e.clientX-r.left)/r.width*2,y=(e.clientY-r.top)/r.height;if(x<=1)roi(x,y,1)};
async function roi(x,y,k){await fetch(`/roi?x=${x}&y=${y}&k=${k}`)}
let E=EXP0;async function ex(d){E=Math.max(10,E+d);document.getElementById('e').textContent=await (await fetch('/exp?v='+E)).text()}
async function rec(){const t=document.getElementById('t').value,n=document.getElementById('n').value;
 const j=await (await fetch('/rec?true='+t+'&note='+encodeURIComponent(n))).json();
 document.getElementById('m').textContent=j.ok?`실제 ${j.row.true_m} → ${j.row.z_mean_m} m (${(j.row.err_m*100).toFixed(1)} cm, ${j.row.err_pct}%)`:j.msg;if(!j.ok)alert('기록 안 됨: '+j.msg);tab()}
async function tab(){const rs=await (await fetch('/rows')).json();const c=['time','true_m','z_mean_m','err_m','err_pct','z_std_m','d_full_px','valid_frac','pred_dz_0.25px_m','note'];
 document.getElementById('r').innerHTML='<tr>'+c.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+rs.map(r=>'<tr>'+c.map(x=>'<td>'+(r[x]??'')+'</td>').join('')+'</tr>').join('')}
setInterval(async()=>{try{const s=await (await fetch('/stats')).json();document.getElementById('s').textContent=s.z_mean_m!==undefined?
 `Z ${s.z_mean_m.toFixed(3)} m (시차로 ${s.z_from_dmean_m.toFixed(3)}) · 흔들림 sd ${(s.z_std_m*100).toFixed(1)} cm · 시차 ${s.d_full_px} px · 유효 ${(s.valid_frac*100).toFixed(0)}% · ${s.valid_frames}/${s.n}프레임 · ${s.ms} ms`:
 `ROI에 깊이 없음 (${s.n}프레임) — 무늬 있는 평평한 물체를 상자 안에`}catch(e){}},500);tab()
</script>"""
PAGE = PAGE.replace("EXP0", str(a.exposure)).replace("TILES", " ".join(f"<button onclick=\"t.value='{x}';document.querySelectorAll('.tb').forEach(b=>b.style.background='');this.style.background='#4c4'\" class=tb>{x}</button>" for x in ("0.90", "1.35", "1.80", "2.70", "3.60", "4.05"))).encode()


class Hd(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def send(self, body, ct="application/json"):
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = up.urlparse(self.path); q = {k: v[0] for k, v in up.parse_qs(u.query).items()}
        j = lambda o: self.send(json.dumps(o, ensure_ascii=False).encode())
        if u.path == "/frame.jpg" and VIEW["jpg"]: return self.send(VIEW["jpg"], "image/jpeg")
        if u.path == "/stats": return j(VIEW["stat"])
        if u.path == "/rows": return j(rows())
        if u.path == "/rowcheck": return j(rowcheck())
        if u.path == "/sample": return j(sample(q.get("name", "test"), float(q.get("sec", 5)), q.get("note", "")))
        if u.path == "/rec":
            try: true_m = float(q["true"])
            except (KeyError, ValueError): return j({"ok": False, "msg": "true=거리(m) 필요"})
            return j(record(true_m, q.get("note", "")))
        if u.path == "/roi":
            k = float(q.get("k", 1)); x, y = float(q.get("x", 0)), float(q.get("y", 0))
            if k == 1: ROI.update(cx=min(max(x, 0), 1), cy=min(max(y, 0), 1))
            else: ROI["s"] = min(max(ROI["s"] * k, 0.02), 0.5)
            with LOCK: HIST.clear()
            return j(ROI)
        if u.path == "/exp":
            a.exposure = int(q.get("v", a.exposure))
            for p in (a.left, a.right): v4l(p, f"exposure_time_absolute={a.exposure}")
            log(f"노출 {a.exposure} (={a.exposure / 10:.1f} ms)")
            return self.send(f"exposure {a.exposure}".encode(), "text/plain")
        self.send(PAGE, "text/html")


threading.Thread(target=loop, daemon=True).start()
log(f"거리 측정 :{a.port}  왼쪽 {a.left}, 오른쪽 {a.right}, 기록 → {a.out}")
ThreadingHTTPServer(("0.0.0.0", a.port), Hd).serve_forever()
