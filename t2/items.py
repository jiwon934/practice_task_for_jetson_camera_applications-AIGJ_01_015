"""과제 2-4: COCO에 없는 물체 5종을 세 방법으로 동시에 이름 붙여 비교 (화면 :8030, 2x2 격자).
 (1) YOLO11n TensorRT FP16 그대로 (COCO 라벨)
 (2) YOLO 박스 크롭 → EmbeddingGemma 2 이미지 임베딩 → 후보 문구(+/ref 참조 사진)와 유사도로 다시 이름 붙이기 (0.20 미만이면 YOLO 라벨 유지, app.py relabel과 같은 방식)
 (3) YOLOE-11s 오픈 어휘 (글로 준 클래스 이름을 바로 검출)
HTTP: /classes?names=a,b,c,d,e  (5종 이름 영문, 쉼표)   /ref?name=a  (지금 화면 가운데를 a의 참조 사진으로 저장)
      /trial?item=a  (5초 동안 세 방법의 가장 확실한 예측을 runs/t2_items.csv에 기록 + 방법별 스냅샷)   /stats
사용: ~/eg2/bin/python items.py [--src 웹캠 by-path] [--size 1280x720]"""
import argparse, collections, csv, json, os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.expanduser("~/eg2-lab/applab/t0"))
os.environ["YOLO_AUTOINSTALL"] = "False"
import cv2, numpy as np, torch
from ultralytics import YOLO, YOLOE
from sentence_transformers import SentenceTransformer
from camlib import LatestReader, safe_name

T2 = os.path.expanduser("~/eg2-lab/applab/t2"); RUNS = os.path.expanduser("~/eg2-lab/applab/runs")
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0")
ap.add_argument("--size", default="1280x720")
ap.add_argument("--classes", default="stapler,power strip,glasses,whiteboard eraser,tape")
ap.add_argument("--trial-sec", type=float, default=8)
ap.add_argument("--yoloe-conf", type=float, default=0.05)
ap.add_argument("--yoloe-model", default=os.path.expanduser("~/eg2-lab/yoloe-11s-seg.pt"))
ap.add_argument("--port", type=int, default=8030, choices=[8020, 8030])
a = ap.parse_args()
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
os.chdir(T2)                                  # YOLOE가 mobileclip_blt.ts를 여기서 찾음 (내려받기 없음)
LABEL_MIN, DIM, MIN_SIDE = 0.20, 512, 32
BIG = {"chair", "couch", "bed", "dining table", "tv", "laptop", "keyboard", "refrigerator", "oven", "person"}
# 재라벨 후보에 흔한 COCO 물건도 넣어 둠: 5종이 아닌 물건을 억지로 5종 중 하나로 부르지 않게
DISTRACT = ["person", "hand", "cup", "bottle", "cell phone", "laptop", "keyboard", "mouse", "book", "chair", "scissors", "remote"]

yolo = YOLO(os.path.join(T2, "yolo11n_fp16.engine"), task="detect")
yoloe = YOLOE(a.yoloe_model)
YTAG = os.path.basename(a.yoloe_model).replace("-seg.pt", "").replace("yoloe-", "")   # 11s / 11l: CSV 방법 이름에 붙여 모델별로 구분
emb = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16},
                          config_kwargs={"audio_config": None})
lock = threading.Lock()
S = {"classes": [], "cand": [], "text_vec": None, "refs": collections.defaultdict(list)}


def text_vecs(names):
    with torch.inference_mode():
        return np.asarray(emb.encode([f"a photo of a {n}" for n in names], prompt_name="SearchQuery",
                                     truncate_dim=DIM, normalize_embeddings=True), dtype=np.float32)


def img_vecs(crops):
    paths = []
    for i, c in enumerate(crops):
        p = f"/dev/shm/t2_items_{os.getpid()}_{i}.jpg"; cv2.imwrite(p, c); paths.append(p)
    with torch.inference_mode():
        v = emb.encode([{"image": p} for p in paths], truncate_dim=DIM, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


def set_classes(names):
    names = [n.strip() for n in names if n.strip()][:8]
    cand = names + [d for d in DISTRACT if d not in names]
    tv = text_vecs(cand)
    with lock:
        yoloe.set_classes(names, yoloe.get_text_pe(names))
        S.update(classes=names, cand=cand, text_vec=tv)
    log(f"클래스 {names}")


def relabel(crop_vecs):
    """후보 문구 유사도와 (있으면) 참조 사진 유사도 중 큰 값으로 라벨 결정"""
    sims = crop_vecs @ S["text_vec"].T
    for j, n in enumerate(S["cand"]):
        if S["refs"].get(n):
            sims[:, j] = np.maximum(sims[:, j], (crop_vecs @ np.stack(S["refs"][n]).T).max(1))
    return sims


set_classes(a.classes.split(","))
_rd = os.path.join(RUNS, "t2_items_refs")
for fn in sorted(os.listdir(_rd)) if os.path.isdir(_rd) else []:
    name = fn.rsplit("_", 1)[0].replace("_", " ")
    if name in S["cand"]:
        S["refs"][name].append(img_vecs([cv2.imread(os.path.join(_rd, fn))])[0])
log(f"참조 사진 불러옴 { {k: len(v) for k, v in S['refs'].items()} }")
w, h = map(int, a.size.split("x"))
rd = LatestReader(a.src, w, h, 30, log=log); rd.start()
cond, jpeg = threading.Condition(), {"seq": 0, "data": None}
st = {"fps": 0.0, "ms": {}, "top": {}}
trial = {"item": None, "end": 0, "rows": [], "snaps": {}}
CSV = os.path.join(RUNS, "t2_items.csv")
if not os.path.exists(CSV):
    csv.writer(open(CSV, "w", newline="")).writerow(["time", "item", "method", "frames", "frames_with_det", "top_label", "top_share", "correct_share", "mean_score", "snapshot"])
COL = {"yolo": (0, 200, 255), "relabel": (255, 160, 0), "yoloe": (0, 255, 0)}


def panel(img, title, dets, color):
    v = img.copy()
    for (x1, y1, x2, y2), lab, sc in dets:
        cv2.rectangle(v, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(v, f"{lab} {sc:.2f}", (int(x1), max(int(y1) - 8, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.rectangle(v, (0, 0), (v.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(v, title, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    return cv2.resize(v, (640, 360))


def finish():
    item = trial["item"]
    for m in ("yolo", "relabel", "yoloe"):
        rows = [r for r in trial["rows"] if r[0] == m]
        hit = [r for r in rows if r[1]]
        cnt = collections.Counter(r[1] for r in hit)
        top, k = cnt.most_common(1)[0] if cnt else ("-", 0)
        ok = sum(1 for r in hit if r[1].lower() == item.lower())
        snap = f"t2_items_{item.replace(' ', '_')}_{m}{'' if m != 'yoloe' or YTAG == '11s' else '_' + YTAG}.jpg"
        if m in trial["snaps"]:
            cv2.imwrite(os.path.join(RUNS, snap), trial["snaps"][m])
        row = [time.strftime("%H:%M:%S"), item, m if m != "yoloe" or YTAG == "11s" else f"yoloe_{YTAG}", len(rows), len(hit), top, round(k / max(len(rows), 1), 3),
               round(ok / max(len(rows), 1), 3), round(float(np.mean([r[2] for r in hit])), 3) if hit else 0, snap]
        csv.writer(open(CSV, "a", newline="")).writerow(row)
        st["top"][m] = {"label": top, "share": row[6], "correct": row[7]}
        log(f"시험 {item} {m}: {row}")
    trial["item"] = None


rl_in, rl_out = {"job": None}, {"dets": [], "ms": 0.0, "seq": 0}


def relabel_loop():
    while True:
        job = rl_in["job"]
        if job is None:
            time.sleep(0.01); continue
        rl_in["job"] = None; f, d1 = job
        t = time.perf_counter(); d2 = []
        hh, ww = f.shape[:2]
        # 가구·화면처럼 큰 배경 물체는 빼고, 작은 물건 박스부터 최대 3개 (책상 위 물건이 대상)
        crops = [x for x in d1 if x[1] not in BIG and min(x[0][2] - x[0][0], x[0][3] - x[0][1]) >= MIN_SIDE
                 and (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]) < 0.15 * ww * hh]
        crops = sorted(crops, key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))[:3]
        if crops:
            vecs = img_vecs([f[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b, _, _ in crops])
            with lock:
                sims = relabel(vecs); cand = S["cand"]
            for (b, lab, sc), sim in zip(crops, sims):
                j = int(sim.argmax())
                d2.append((b, cand[j], float(sim[j])) if sim[j] >= LABEL_MIN else (b, lab, float(sim[j])))
        rl_out.update(dets=d2, ms=(time.perf_counter() - t) * 1000, seq=rl_out["seq"] + 1)


def loop():
    last, n, t0 = -1, 0, time.time()
    while True:
        f, ts, _, seq = rd.latest()
        if f is None or seq == last:
            time.sleep(0.003); continue
        last = seq; ms = {}
        t = time.perf_counter()
        r1 = yolo.predict(f, conf=0.15, imgsz=640, device=0, verbose=False)[0]
        ms["yolo"] = (time.perf_counter() - t) * 1000
        d1 = [(b, yolo.names[int(c)], float(s)) for b, c, s in zip(r1.boxes.xyxy.tolist(), r1.boxes.cls.tolist(), r1.boxes.conf.tolist())
              if yolo.names[int(c)] != "person"]
        rl_in["job"] = (f, d1)            # 재라벨은 느려서(장당 약 0.4초) 별도 스레드가 최신 것만 처리, 화면엔 마지막 결과
        d2 = rl_out["dets"]; ms["relabel"] = rl_out["ms"]
        t = time.perf_counter()
        with lock:
            r3 = yoloe.predict(f, conf=a.yoloe_conf, imgsz=640, device=0, half=True, verbose=False)[0]
            names3 = S["classes"]
        ms["yoloe"] = (time.perf_counter() - t) * 1000
        d3 = [(b, names3[int(c)] if int(c) < len(names3) else str(int(c)), float(s))
              for b, c, s in zip(r3.boxes.xyxy.tolist(), r3.boxes.cls.tolist(), r3.boxes.conf.tolist())]
        P = {"yolo": panel(f, f"1 YOLO (COCO) {ms['yolo']:.0f}ms", d1, COL["yolo"]),
             "relabel": panel(f, f"2 relabel {ms['relabel']:.0f}ms", d2, COL["relabel"]),
             "yoloe": panel(f, f"3 YOLOE-{YTAG} {ms['yoloe']:.0f}ms", d3, COL["yoloe"])}
        info = np.zeros((360, 640, 3), np.uint8)
        lines = [f"classes: {', '.join(S['classes'])}", f"refs: {', '.join(f'{k}:{len(v)}' for k, v in S['refs'].items()) or '-'}",
                 f"{st['fps']:.1f} fps", ("TRIAL " + trial["item"] + f" {max(trial['end'] - time.time(), 0):.0f}s") if trial["item"] else "idle"]
        for m, v in st["top"].items():
            lines.append(f"last {m}: {v['label']} ok {v['correct']:.0%}")
        for i, l in enumerate(lines):
            cv2.putText(info, l[:52], (10, 34 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255) if l.startswith("TRIAL") else (230, 230, 230), 2)
        grid = np.vstack([np.hstack([P["yolo"], P["relabel"]]), np.hstack([P["yoloe"], info])])
        if trial["item"]:
            fresh = rl_out["seq"] != trial.get("rl_seq"); trial["rl_seq"] = rl_out["seq"]
            for m, d in (("yolo", d1), ("relabel", d2), ("yoloe", d3)):
                if m == "relabel" and not fresh:
                    continue
                dd = [x for x in d if x[1] not in BIG]
                best = max(dd, key=lambda x: x[2]) if dd else None
                trial["rows"].append((m, best[1] if best else "", best[2] if best else 0.0))
                if best and m not in trial["snaps"]:
                    trial["snaps"][m] = P[m]
            if time.time() >= trial["end"]:
                finish()
        ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with cond:
            jpeg["data"], jpeg["seq"] = buf.tobytes(), jpeg["seq"] + 1; cond.notify_all()
        st["ms"] = {k: round(v, 1) for k, v in ms.items()}
        n += 1; now = time.time()
        if now - t0 >= 1:
            st["fps"] = round(n / (now - t0), 1); n, t0 = 0, now


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def reply(self, obj, ct="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200); self.send_header("Content-Type", ct); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        arg = lambda k: q.get(k, [""])[0].strip()
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
        if u.path == "/classes" and arg("names"):
            set_classes(arg("names").split(",")); return self.reply({"classes": S["classes"]})
        if u.path == "/ref" and arg("name"):
            f, _, _, _ = rd.latest(); hh, ww = f.shape[:2]
            c = f[hh // 4: hh * 3 // 4, ww // 4: ww * 3 // 4]
            os.makedirs(os.path.join(RUNS, "t2_items_refs"), exist_ok=True)
            cv2.imwrite(os.path.join(RUNS, "t2_items_refs", f"{arg('name').replace(' ', '_')}_{int(time.time())}.jpg"), c)
            v = img_vecs([c])[0]
            with lock:
                S["refs"][arg("name")].append(v)
            log(f"참조 {arg('name')} {len(S['refs'][arg('name')])}장"); return self.reply({"ref": arg("name"), "n": len(S["refs"][arg("name")])})
        if u.path == "/trial" and arg("item"):
            if trial["item"]:
                return self.reply({"busy": trial["item"]})
            trial.update(item=arg("item"), end=time.time() + a.trial_sec, rows=[], snaps={}); log(f"시험 시작 {arg('item')}")
            return self.reply({"trial": arg("item"), "sec": a.trial_sec})
        if u.path == "/stats":
            return self.reply(st | {"classes": S["classes"], "refs": {k: len(v) for k, v in S["refs"].items()}})
        self.reply(b"<!doctype html><meta charset=utf-8><title>t2 items</title><body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>"
                   b"<img src=/stream style='max-width:100%'><p id=s></p><script>setInterval(async()=>{document.getElementById('s').textContent="
                   b"JSON.stringify(await (await fetch('/stats')).json())},1000)</script>", "text/html")


log(f"입력 {safe_name(a.src)}, 화면 :{a.port}")
threading.Thread(target=loop, daemon=True).start()
threading.Thread(target=relabel_loop, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
