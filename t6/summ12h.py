"""12시간 기록 요약: 가동률, 카메라별 offline 구간·재연결, fps·추론 ms·GPU·RAM·전력·온도 통계, RSS 증가(메모리 누수), 재시작, 새 이미지 수.
사용: ~/eg2/bin/python summ12h.py runs/t6_12h_YYYYMMDD_HHMM.csv"""
import csv, sys, numpy as np
R = list(csv.DictReader(open(sys.argv[1]))); n = len(R)
f = lambda k: np.array([float(r[k]) for r in R if r.get(k) not in ("", None, "None")])
ok = sum(r["health"] == "ok" for r in R); err = sum(r["health"].startswith("ERR") for r in R)
print(f"기간 {R[0]['time']} ~ {R[-1]['time']} ({(float(R[-1]['ts']) - float(R[0]['ts'])) / 3600:.2f} h), 기록 {n}줄")
print(f"health ok {ok / n * 100:.2f}%, degraded {(n - ok - err) / n * 100:.2f}%, 응답 없음 {err / n * 100:.2f}%")
pids = sorted({r["pid"] for r in R}); print(f"서비스 PID {pids}, NRestarts 마지막 {R[-1]['nrestarts']}")
for k in ("ipcam", "left", "right"):
    on = [r[f"{k}_on"] == "True" for r in R]; segs, i = [], 0
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]: j += 1
            segs.append((R[i]["time"], R[j - 1]["time"], (j - i) * 10)); i = j
        else: i += 1
    rc = [int(r[f"{k}_rc"]) for r in R if r[f"{k}_rc"].isdigit()]
    print(f"{k}: online {np.mean(on) * 100:.2f}%, 재연결 {rc[-1] - rc[0] if rc else '?'}회, offline 구간 {len(segs)}개 {segs[:5]}")
for k, nm in (("occ_fps", "A1 fps"), ("occ_infer_ms", "A1 추론 ms"), ("prox_fps", "A4 fps"), ("gpu_pct", "GPU %"), ("ram_mb", "RAM MB"), ("vdd_in_mw", "전력 mW"), ("tj_temp", "Tj ℃"), ("rss_mb", "서비스 RSS MB")):
    v = f(k)
    if len(v): print(f"{nm}: 평균 {v.mean():.1f}, p5 {np.percentile(v, 5):.1f}, p95 {np.percentile(v, 95):.1f}, 최대 {v.max():.1f}")
rss = f("rss_mb")
if len(rss) > 20: print(f"RSS 처음 1시간 평균 {rss[:360].mean():.0f} MB → 마지막 1시간 {rss[-360:].mean():.0f} MB (증가 {rss[-360:].mean() - rss[:360].mean():+.0f} MB)")
st = [r["prox_state"] for r in R]; print("A4 상태 비율", {s: f"{st.count(s) / n * 100:.1f}%" for s in sorted(set(st))})
im = [int(r["new_images"]) for r in R if r["new_images"].isdigit()]; print(f"새 이미지 파일: 마지막 {im[-1] if im else '?'}개 (0이어야 함)")
