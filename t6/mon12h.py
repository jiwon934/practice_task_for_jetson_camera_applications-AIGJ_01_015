"""과제 6 12시간 연속 운전 기록: 10초마다 /v1/health·서비스 PID·재시작 수·RSS·tegrastats·새 이미지 파일 수를 CSV 한 줄로. --hours 지나면 끝남.
사용: setsid nohup ~/eg2/bin/python mon12h.py [--hours 12] &"""
import argparse, csv, json, os, re, subprocess, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--hours", type=float, default=12); ap.add_argument("--every", type=float, default=10)
a = ap.parse_args()
R = os.path.expanduser("~/eg2-lab/applab/runs"); T0 = time.time(); tag = time.strftime("%Y%m%d_%H%M")
OUT = os.path.join(R, f"t6_12h_{tag}.csv"); TEG = os.path.join(R, "t6_12h_tegra.log")
MARK = os.path.join(R, f".t6_12h_{tag}.start"); open(MARK, "w").close()
TL = {"line": ""}
def teg():
    p = subprocess.Popen(["tegrastats", "--interval", str(int(a.every * 1000))], stdout=subprocess.PIPE, text=True)
    with open(TEG, "a") as f:
        for line in p.stdout:
            TL["line"] = line.strip(); f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line); f.flush()
            if time.time() - T0 > a.hours * 3600: p.terminate(); return
threading.Thread(target=teg, daemon=True).start()
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
def images():
    out = sh(f"find {os.path.expanduser('~/eg2-lab')} -newer {MARK} -type f \\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.mp4' -o -iname '*.avi' \\) 2>/dev/null")
    return [x for x in out.splitlines() if x]
COLS = ["ts", "time", "health", "pid", "nrestarts", "rss_mb", "ipcam_on", "ipcam_age", "ipcam_rc", "left_on", "left_age", "left_rc", "right_on", "right_age", "right_rc",
        "occ_age", "occ_fps", "occ_infer_ms", "prox_age", "prox_fps", "prox_state", "mqtt", "gpu_pct", "ram_mb", "vdd_in_mw", "cpu_temp", "gpu_temp", "tj_temp", "new_images"]
with open(OUT, "w", newline="") as f: csv.writer(f).writerow(COLS)
while time.time() - T0 < a.hours * 3600:
    t = time.time(); row = {"ts": round(t, 1), "time": time.strftime("%H:%M:%S")}
    try:
        h = json.load(urllib.request.urlopen("http://127.0.0.1:8030/v1/health", timeout=3)); row["health"] = h["status"]; row["mqtt"] = h["mqtt"]
        for k in ("ipcam", "left", "right"):
            c = h["cameras"].get(k, {}); row[f"{k}_on"], row[f"{k}_age"], row[f"{k}_rc"] = c.get("online"), c.get("age_s"), c.get("reconnects")
        row.update(occ_age=h["occupancy"]["age_s"], occ_fps=h["occupancy"]["fps"], occ_infer_ms=h["occupancy"].get("infer_ms"),
                   prox_age=h["proximity"]["age_s"], prox_fps=h["proximity"]["fps"], prox_state=h["proximity"]["state"])
    except Exception as e: row["health"] = f"ERR {type(e).__name__}"
    pid = sh("systemctl --user show t6-vision -p MainPID --value"); row["pid"] = pid; row["nrestarts"] = sh("systemctl --user show t6-vision -p NRestarts --value")
    if pid and pid != "0":
        rss = sh(f"ps -o rss= -p {pid}"); row["rss_mb"] = round(int(rss) / 1024) if rss.isdigit() else ""
    L = TL["line"]
    for k, pat in (("gpu_pct", r"GR3D_FREQ (\d+)%"), ("ram_mb", r"RAM (\d+)/"), ("vdd_in_mw", r"VDD_IN (\d+)mW"), ("cpu_temp", r"cpu@([\d.]+)C"), ("gpu_temp", r"gpu@([\d.]+)C"), ("tj_temp", r"tj@([\d.]+)C")):
        m = re.search(pat, L); row[k] = m.group(1) if m else ""
    if int(t) % 60 < a.every: row["new_images"] = len(images())   # 1분에 한 번만 찾음
    with open(OUT, "a", newline="") as f: csv.DictWriter(f, COLS).writerow(row)
    time.sleep(max(0, a.every - (time.time() - t)))
print("끝", OUT, flush=True)
