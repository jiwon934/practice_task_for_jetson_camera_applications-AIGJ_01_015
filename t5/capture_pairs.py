"""과제 5-3: 스테레오 보정용 체커보드 쌍 자동 촬영.
- 시작 전에 두 웹캠의 자동 초점·자동 노출·자동 화이트밸런스를 끄고 같은 값으로 고정, 전원 주파수 60 Hz(형광등 깜빡임) 설정 (v4l2-ctl).
- 두 카메라를 각자 스레드로 계속 읽고, 시각 차이가 --sync-ms 안인 두 장을 한 쌍으로 씀 (프레임 어긋남 통계도 기록).
- 두 장 모두에서 체커보드(내부 코너 --cols x --rows)가 보이고, 이미 저장한 자세와 충분히 다르면 저장. --n 쌍 모이면 끝.
- 저장: --out/pair_NN_L.png, _R.png, corners.json, coverage.png(저장된 코너 분포), log.txt
사용: ~/eg2/bin/python capture_pairs.py [--left 2.4] [--right 2.2] [--cols 9 --rows 6] [--n 25]"""
import argparse, json, os, subprocess, threading, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--left", default="2.4"); ap.add_argument("--right", default="2.2")
ap.add_argument("--cols", type=int, default=9); ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--sync-ms", type=float, default=8)
ap.add_argument("--still-px", type=float, default=1.5, help="직전 검출과 비교해 코너가 이만큼 이하로 움직였을 때만 저장 (들고 있는 손 떨림·이동 중 흐림 방지)")
ap.add_argument("--focus", type=int, default=0, help="고정 초점 값 (0 = 무한대 쪽)")
ap.add_argument("--exposure", type=int, default=100, help="고정 노출 (exposure_time_absolute, 100 µs 단위)")
ap.add_argument("--gain", type=int, default=30)
ap.add_argument("--out", default=os.path.expanduser("~/eg2-lab/applab/runs/t5_calib2"))
ap.add_argument("--port", type=int, default=0, help="0이 아니면 이 포트에 실시간 화면(두 카메라, 검출 표시, 저장 개수)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "log.txt"), "a")
def log(m):
    s = time.strftime("%H:%M:%S ") + m; print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

dev = lambda p: f"/dev/v4l/by-path/platform-3610000.usb-usb-0:{p}:1.0-video-index0"
for p in (a.left, a.right):   # 자동 기능 끄고 같은 값으로 고정 (컨트롤 이름은 드라이버마다 조금 다를 수 있어 실패해도 계속)
    for c in ["focus_automatic_continuous=0", f"focus_absolute={a.focus}", "auto_exposure=1", f"exposure_time_absolute={a.exposure}",
              f"gain={a.gain}", "white_balance_automatic=0", "white_balance_temperature=4500", "power_line_frequency=2", "exposure_dynamic_framerate=0"]:
        r = subprocess.run(["v4l2-ctl", "-d", dev(p), "-c", c], capture_output=True, text=True)
        if r.returncode: log(f"[{p}] {c} 실패: {r.stderr.strip()[:80]}")
    log(f"[{p}] " + subprocess.run(["v4l2-ctl", "-d", dev(p), "-C", "focus_absolute,exposure_time_absolute,gain,white_balance_temperature"], capture_output=True, text=True).stdout.replace("\n", " "))


class Cam(threading.Thread):
    def __init__(self, p):
        super().__init__(daemon=True)
        self.c = cv2.VideoCapture(dev(p), cv2.CAP_V4L2)
        self.c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); self.c.set(3, 1280); self.c.set(4, 720); self.c.set(5, 30)
        self.c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.f, self.t, self.n = None, 0.0, 0
    def run(self):
        while True:
            ok, f = self.c.read()
            if ok: self.f, self.t, self.n = f, time.time(), self.n + 1


L, R = Cam(a.left), Cam(a.right); L.start(); R.start()
VIEW = {"data": None, "msg": ""}
if a.port:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    PAGE = """<!doctype html><meta charset=utf-8><title>t5 capture</title><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>
<img id=v style='width:100%'><h2 id=s></h2>
<p>노출: <button onclick="ex(-20)">어둡게 −</button> <button onclick="ex(20)">밝게 +</button> <span id=e></span></p><script>
const v=document.getElementById('v');let last=Date.now(),busy=false;
function next(){if(busy)return;busy=true;const im=new Image();im.onload=()=>{v.src=im.src;last=Date.now();busy=false;setTimeout(next,150)};
 im.onerror=()=>{busy=false;setTimeout(next,1000)};im.src='/frame.jpg?'+Date.now()}
setInterval(()=>{if(Date.now()-last>3000){busy=false;next()}},1000);next();
let E=EXP0;async function ex(d){E=Math.max(10,E+d);document.getElementById('e').textContent=await (await fetch('/exp?v='+E)).text()}
setInterval(async()=>{try{document.getElementById('s').textContent=(await (await fetch('/stats')).json()).msg}catch(e){}},700)
</script>"""
    PAGE = PAGE.replace("EXP0", str(a.exposure)).encode()
    class Hd(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def do_GET(self):
            if self.path.startswith("/exp"):
                import urllib.parse as up
                q = up.parse_qs(up.urlparse(self.path).query); v = int(q.get("v", [a.exposure])[0]); gn = int(q.get("g", [a.gain])[0])
                for p in (a.left, a.right):
                    subprocess.run(["v4l2-ctl", "-d", dev(p), "-c", f"exposure_time_absolute={v}", "-c", f"gain={gn}"])
                a.exposure, a.gain = v, gn; log(f"노출 {v} (={v / 10:.1f} ms), 게인 {gn}")
                self.send_response(200); self.end_headers(); return self.wfile.write(f"exposure {v} gain {gn}".encode())
            if self.path.startswith("/frame.jpg") and VIEW["data"]:
                self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store"); self.end_headers(); return self.wfile.write(VIEW["data"])
            body, ct = (json.dumps({"msg": VIEW["msg"]}, ensure_ascii=False).encode(), "application/json") if self.path == "/stats" else (PAGE, "text/html")
            self.send_response(200); self.send_header("Content-Type", ct); self.end_headers(); self.wfile.write(body)
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", a.port), Hd).serve_forever, daemon=True).start()


LAST = {"cl": None, "cr": None, "note": "", "t": 0}


def show(fl, fr, cl, cr, note):
    LAST.update(cl=cl, cr=cr, note=note, t=time.time())


def render():
    while True:
        time.sleep(0.1)
        if not a.port or L.f is None or R.f is None: continue
        fresh = time.time() - LAST["t"] < 1.0
        draw(L.f, R.f, LAST["cl"] if fresh else None, LAST["cr"] if fresh else None, LAST["note"])


def draw(fl, fr, cl, cr, note):
    tl, tr = cv2.resize(fl, (640, 360)), cv2.resize(fr, (640, 360))
    for t, c, nm in ((tl, cl, "LEFT"), (tr, cr, "RIGHT")):
        if c is not None: cv2.drawChessboardCorners(t, pat, c * 0.5, True)
        cv2.putText(t, nm + (" BOARD" if c is not None else ""), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if c is not None else (0, 0, 255), 2)
    cov_s = cv2.resize(cov, (1280, 360))
    img = np.vstack([np.hstack([tl, tr]), cv2.addWeighted(cov_s, 1.0, np.zeros_like(cov_s), 0, 0)])
    VIEW["data"] = cv2.imencode(".jpg", cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()
    VIEW["msg"] = f"저장 {len(saved)}/{a.n}쌍 — {note}  (아래: 지금까지 찍힌 코너 위치, 빈 곳을 채우세요)"
time.sleep(2)
pat = (a.cols, a.rows)
saved, feats, skews, tried = [], [], [], 0
PREV = {}
cov = np.zeros((720, 1280 * 2, 3), np.uint8)


CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)


def detect(g):
    """절반 크기 영상에서 찾고, 원본에서 cornerSubPix로 다듬기"""
    ok, c = cv2.findChessboardCornersSB(cv2.resize(g, (640, 360)), pat)
    if not ok: return False, None
    return True, cv2.cornerSubPix(g, (c * 2.0).astype(np.float32), (7, 7), (-1, -1), CRIT)


def feat(c):
    """자세 특징: 중심(x,y), 크기, 가로·세로 기울기 비율"""
    c = c.reshape(-1, 2); xs, ys = c[:, 0], c[:, 1]
    top = np.linalg.norm(c[a.cols - 1] - c[0]); bot = np.linalg.norm(c[-1] - c[-a.cols])
    lef = np.linalg.norm(c[-a.cols] - c[0]); rig = np.linalg.norm(c[-1] - c[a.cols - 1])
    return np.array([xs.mean() / 1280, ys.mean() / 720, (np.ptp(xs) * np.ptp(ys)) ** 0.5 / 720, top / max(bot, 1), lef / max(rig, 1)])


threading.Thread(target=render, daemon=True).start()
log(f"촬영 시작: 내부 코너 {a.cols}x{a.rows}, 목표 {a.n}쌍, 왼쪽={a.left}, 오른쪽={a.right}")
while len(saved) < a.n:
    fl, tl, fr, tr = L.f, L.t, R.f, R.t
    if fl is None or fr is None: time.sleep(0.05); continue
    sk = abs(tl - tr) * 1000; skews.append(sk)
    if sk > a.sync_ms: time.sleep(0.01); continue
    gl, gr = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    tried += 1
    okl, cl = detect(gl)
    okr, cr = detect(gr) if okl else (False, None)
    if not (okl and okr):
        show(fl, fr, cl if okl else None, cr if okr else None, "왼쪽만 보임" if okl else "체커보드 안 보임 (두 카메라 모두에 보이게)"); continue
    prev = PREV.get("c"); PREV["c"] = (cl, cr)
    if prev is None or np.abs(prev[0] - cl).max() > a.still_px or np.abs(prev[1] - cr).max() > a.still_px:
        show(fl, fr, cl, cr, "보드를 잠깐 멈춰 주세요 (움직이는 중)"); continue
    f = feat(cl)
    if feats and min(np.abs(np.array(feats) - f).max(1)) < 0.06:
        show(fl, fr, cl, cr, "이미 찍은 자세와 비슷함 — 위치·거리·기울기를 바꿔 주세요"); continue
    k = len(saved) + 1
    cv2.imwrite(os.path.join(a.out, f"pair_{k:02d}_L.png"), fl); cv2.imwrite(os.path.join(a.out, f"pair_{k:02d}_R.png"), fr)
    saved.append({"pair": k, "skew_ms": round(sk, 1), "L": cl.reshape(-1, 2).tolist(), "R": cr.reshape(-1, 2).tolist(), "feat": f.round(3).tolist()})
    feats.append(f)
    for (x, y) in cl.reshape(-1, 2): cv2.circle(cov, (int(x), int(y)), 3, (0, 255, 0), -1)
    for (x, y) in cr.reshape(-1, 2): cv2.circle(cov, (int(x) + 1280, int(y)), 3, (0, 200, 255), -1)
    cv2.imwrite(os.path.join(a.out, "coverage.png"), cov)
    json.dump({"pattern": pat, "pairs": saved}, open(os.path.join(a.out, "corners.json"), "w"))
    show(fl, fr, cl, cr, f"쌍 {k} 저장!")
    log(f"쌍 {k}/{a.n} 저장 (어긋남 {sk:.0f} ms, 중심 {f[0]:.2f},{f[1]:.2f} 크기 {f[2]:.2f})")
sk = np.array(skews)
log(f"끝: {len(saved)}쌍, 검사한 동기 쌍 {tried}, 두 카메라 프레임 시각 차이 평균 {sk.mean():.1f} ms, 중앙값 {np.median(sk):.1f}, 95% {np.percentile(sk, 95):.1f}")
os._exit(0)
