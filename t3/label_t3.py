"""과제 3 정답 기입(T2-8): 스냅샷을 한 장씩 보여 주고 실제 인원을 버튼으로 입력. 예측 인원은 보여 주지 않음(정답이 예측에 끌려가지 않게).
- 하루에 고르게 퍼진 --n장을 고름 (시간 순으로 같은 간격 + 아직 안 뽑힌 것 중 예측 0명·최대 인원 쪽을 조금 더). 고른 목록과 결과는 eval_occ.py 형식의 gt.csv
- 구역 JSON(포함·제외·이름 구역)을 사진 위에 그림. 셀 사람: 포함 영역 안, 제외 영역 밖에 발이 있는 사람
- /door: 스냅샷 위에 두 점을 클릭해 문 가상선을 정하면 runs/t3_door_line.json에 저장 (line_count.py --line 값)
사진은 젯슨에서 브라우저로만 보냄.
사용: ~/eg2/bin/python label_t3.py [--snap-dir runs/t3_snaps_1030] [--n 40] [--port 8031]"""
import argparse, csv, json, os, random
import urllib.parse as up
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

A = os.path.expanduser("~/eg2-lab/applab")
ap = argparse.ArgumentParser()
ap.add_argument("--snap-dir", default=os.path.join(A, "runs/t3_snaps_1030"))
ap.add_argument("--zones", default=os.path.join(A, "t3/zones/room.json"))
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--port", type=int, default=8031)
ap.add_argument("--empty", default="11:41:00-11:57:30", help="빈 방 확인용 추가 세트: 이 시간대 스냅샷 전부 → gt_empty.csv (/?set=empty)")
a = ap.parse_args()
GT = os.path.join(a.snap_dir, "gt.csv")
COLS = ["file", "ts", "raw", "count", "gt", "note"]

if os.path.exists(GT):
    ROWS = list(csv.DictReader(open(GT)))
else:
    allr = [dict(zip(["ts", "file", "raw", "count"], r)) for r in csv.reader(open(os.path.join(a.snap_dir, "snaps.csv")))]
    allr = [r for r in allr if os.path.exists(os.path.join(a.snap_dir, r["file"]))]
    k = int(a.n * 0.8)
    idx = sorted({round(i * (len(allr) - 1) / (k - 1)) for i in range(k)})
    rest = [i for i in range(len(allr)) if i not in idx]
    random.seed(0)
    zero = [i for i in rest if int(allr[i]["raw"]) == 0]; busy = sorted(rest, key=lambda i: -int(allr[i]["count"]))
    extra = random.sample(zero, min(len(zero), (a.n - len(idx)) // 2))
    extra += [i for i in busy if i not in extra][:a.n - len(idx) - len(extra)]
    ROWS = [{"file": allr[i]["file"], "ts": allr[i]["ts"], "raw": allr[i]["raw"], "count": allr[i]["count"], "gt": "", "note": ""} for i in sorted(idx + extra)]
import time as _t
GTE = os.path.join(a.snap_dir, "gt_empty.csv")
if os.path.exists(GTE): EROWS = list(csv.DictReader(open(GTE)))
else:
    t0, t1 = a.empty.split("-")
    EROWS = [{"file": r[1], "ts": r[0], "raw": r[2], "count": r[3], "gt": "", "note": ""} for r in csv.reader(open(os.path.join(a.snap_dir, "snaps.csv")))
             if t0 <= _t.strftime("%H:%M:%S", _t.localtime(float(r[0]))) <= t1 and os.path.exists(os.path.join(a.snap_dir, r[1]))]
# 최종 시험 세트: 포함 억제 로거(14:43~)의 스냅샷에서 --final-n장을 시간 순으로 같은 간격 (설계에 안 쓴 사진)
FDIR = os.path.join(A, "runs/t3_snaps_1030b"); GTF = os.path.join(FDIR, "gt_final.csv")
if os.path.exists(GTF): FROWS = list(csv.DictReader(open(GTF)))
elif os.path.exists(os.path.join(FDIR, "snaps.csv")):
    fr = [r for r in csv.reader(open(os.path.join(FDIR, "snaps.csv"))) if os.path.exists(os.path.join(FDIR, r[1]))]
    k = min(30, len(fr)); idx = sorted({round(i * (len(fr) - 1) / max(1, k - 1)) for i in range(k)})
    FROWS = [{"file": fr[i][1], "ts": fr[i][0], "raw": fr[i][2], "count": fr[i][3], "gt": "", "note": ""} for i in idx]
else: FROWS = []
SETS = {"main": (GT, ROWS), "empty": (GTE, EROWS), "final": (GTF, FROWS)}
DIRS = {"main": a.snap_dir, "empty": a.snap_dir, "final": FDIR}
def save(name="main"):
    path, rows = SETS[name]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
save(); save("empty")
if FROWS: save("final")
Z = json.load(open(a.zones))


def overlay(fn, d=None):
    img = cv2.imread(os.path.join(d or a.snap_dir, fn)); h, w = img.shape[:2]
    P = lambda poly: (np.array(poly) * (w, h)).astype(np.int32)
    ov = img.copy()
    for poly in Z.get("exclude", []): cv2.fillPoly(ov, [P(poly)], (0, 0, 200))
    img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    for poly in Z.get("include", []): cv2.polylines(img, [P(poly)], True, (0, 255, 0), 2)
    for name, poly in Z.get("zones", {}).items():
        cv2.polylines(img, [P(poly)], True, (255, 200, 0), 1); x, y = P(poly)[0]; cv2.putText(img, name, (int(x) + 6, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>t3 labeling</title>
<body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>
<p style='margin:6px'><b id=p></b> · 초록 선 안, 빨간 칠 밖에 <b>발</b>이 있는 사람 수를 누르세요 (반쯤 가려져도 사람이면 셈). <a href=/door style='color:#8cf'>문 선 정하기</a> · <a href=/?set=empty style='color:#8cf'>빈 방 확인 세트</a> · <a href=/ style='color:#8cf'>본 세트</a></p>
<img id=v style='width:100%'>
<p style='margin:6px' id=b></p>
<p style='margin:6px'><button onclick=go(-1)>◀ 이전</button> <button onclick=go(1)>다음 ▶</button> <button onclick="lab('skip')">건너뛰기(판단 불가)</button> <span id=m></span></p>
<script>
let R=[],i=0;const SET=new URLSearchParams(location.search).get('set')||'main';
const B=document.getElementById('b');for(let k=0;k<=15;k++)B.innerHTML+=`<button style='font-size:1.4em;width:2.6em;margin:2px' onclick='lab(${k})'>${k}</button>`;
async function load(){R=await (await fetch('/list?set='+SET)).json();i=R.findIndex(r=>r.gt===''&&r.note!=='skip');if(i<0)i=0;show()}
function show(){const r=R[i];document.getElementById('v').src='/img?set='+SET+'&f='+r.file;
 const done=R.filter(r=>r.gt!==''||r.note==='skip').length;
 document.getElementById('p').textContent=(SET=='empty'?'[빈 방 확인 세트] ':SET=='final'?'[최종 시험 세트] ':'')+`${i+1} / ${R.length} (완료 ${done}) · ${r.time}`+(r.gt!==''?` · 입력함: ${r.gt}`:r.note==='skip'?' · 건너뜀':'')}
function go(d){i=Math.min(R.length-1,Math.max(0,i+d));show()}
async function lab(v){const r=R[i];const j=await (await fetch(`/set?set=${SET}&f=${r.file}&v=${v}`)).json();R[i]=j;
 document.getElementById('m').textContent=`저장: ${v}`;const n=R.findIndex((r,k)=>k>i&&r.gt===''&&r.note!=='skip');
 if(n>=0){i=n;show()}else{show();document.getElementById('m').textContent='모두 끝! 고맙습니다'}}
load()
</script>"""

DOOR = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>t3 door line</title>
<body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>
<p style='margin:6px'>문(출입구) 바닥을 가로지르는 선의 양 끝 두 점을 클릭하세요. 사람이 문을 지날 때 발이 이 선을 넘어야 해요. 문마다 따로 저장해요(앞문·뒷문). 그 문이 화면에 안 보이면 문을 고르고 <button onclick="fetch('/doorsave?none=1&name='+document.getElementById('n').value).then(r=>r.text()).then(t=>m.textContent=t)">문 안 보임</button> <a href=/ style='color:#8cf'>정답 기입으로</a></p>
<div id=c style='position:relative;touch-action:none;cursor:crosshair'><img id=v src='/doorimg' style='width:100%;display:block' draggable=false><svg id=s style='position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none'></svg></div>
<p style='margin:6px'>문: <select id=n><option value=front>앞문</option><option value=back>뒷문</option></select> <button onclick='P=[];draw()'>다시</button> 들어오는 방향: <select id=d><option value=down>화면 아래쪽(카메라 쪽)으로</option><option value=up>화면 위쪽으로</option></select>
 <button onclick=save()>저장</button> <span id=m></span></p><script>
let P=[];const v=document.getElementById('v'),s=document.getElementById('s'),m=document.getElementById('m');
document.getElementById('c').addEventListener('pointerdown',e=>{e.preventDefault();const r=v.getBoundingClientRect();if(P.length>=2)P=[];
 P.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);draw();m.textContent=P.length==1?'첫 점 찍음 — 두 번째 점을 찍으세요':'선 완성 — 문·방향을 고르고 저장'});
function draw(){s.innerHTML=P.map(p=>`<circle cx='${p[0]*100}%' cy='${p[1]*100}%' r=9 fill=yellow stroke=black stroke-width=2 />`).join('')+(P.length==2?`<line x1='${P[0][0]*100}%' y1='${P[0][1]*100}%' x2='${P[1][0]*100}%' y2='${P[1][1]*100}%' stroke=yellow stroke-width=4 />`:'')}
async function save(){if(P.length<2){m.textContent='두 점을 찍어 주세요';return}
 m.textContent='저장됨 ✔ '+await (await fetch(`/doorsave?name=${document.getElementById('n').value}&line=${P.flat().map(x=>x.toFixed(3)).join(',')}&dir=${document.getElementById('d').value}`)).text()}
</script>"""


class Hd(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def send(self, body, ct="text/html; charset=utf-8"):
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = up.urlparse(self.path); q = {k: v[0] for k, v in up.parse_qs(u.query).items()}
        import time
        pub = lambda r: {"file": r["file"], "gt": r["gt"], "note": r["note"], "time": time.strftime("%H:%M", time.localtime(float(r["ts"])))}   # 예측(raw/count)은 안 보냄
        sn = q.get("set", "main") if q.get("set") in SETS else "main"; RS = SETS[sn][1]
        if u.path == "/list": return self.send(json.dumps([pub(r) for r in RS]).encode(), "application/json")
        if u.path in ("/img", "/doorimg"):
            fn = q.get("f") if u.path == "/img" else ROWS[-1]["file"]
            if u.path == "/doorimg": return self.send(overlay(fn), "image/jpeg")
            if fn not in {r["file"] for r in RS}: return self.send(b"no", "text/plain")
            return self.send(overlay(fn, DIRS[sn]), "image/jpeg")
        if u.path == "/set":
            r = next(r for r in RS if r["file"] == q.get("f")); v = q.get("v", "")
            if v == "skip": r["gt"], r["note"] = "", "skip"
            else: r["gt"], r["note"] = str(int(v)), ""
            save(sn); return self.send(json.dumps(pub(r)).encode(), "application/json")
        if u.path == "/doorsave":
            path = os.path.join(A, "runs/t3_door_line.json")
            alld = json.load(open(path)) if os.path.exists(path) else {}
            if "lines" not in alld: alld = {"lines": {}}
            d = {"visible": False} if q.get("none") else {"visible": True, "line": q["line"], "in_dir": q.get("dir", "down")}
            alld["lines"][q.get("name", "front")] = d
            json.dump(alld, open(path, "w"), ensure_ascii=False, indent=1); d = alld; return self.send(("저장했어요: " + json.dumps(d, ensure_ascii=False)).encode(), "text/plain; charset=utf-8")
        if u.path == "/door": return self.send(DOOR.encode())
        self.send(PAGE.encode())


print(f"정답 기입 :{a.port}  {len(ROWS)}장 → {GT}", flush=True)
ThreadingHTTPServer(("0.0.0.0", a.port), Hd).serve_forever()
