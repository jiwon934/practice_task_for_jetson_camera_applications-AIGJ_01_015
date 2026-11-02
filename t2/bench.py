"""과제 2-1: 같은 640 입력으로 PyTorch / TensorRT 엔진의 지연(평균·p95)·FPS·전력·메모리, (데이터가 있으면) mAP를 잼.
사용: ~/eg2/bin/python bench.py 이름 가중치(.pt/.engine) [--runs 300] [--half] [--val coco128.yaml] [--img 사진]
결과는 runs/t2_bench.jsonl에 한 줄 추가."""
import argparse, json, os, re, subprocess, threading, time
import numpy as np, torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("weights")
ap.add_argument("--runs", type=int, default=300)
ap.add_argument("--half", action="store_true", help="PyTorch를 FP16으로 (엔진은 이미 정해짐)")
ap.add_argument("--val", default=None)
ap.add_argument("--img", default=os.path.expanduser("~/eg2-lab/media/scene_people.jpg"))
a = ap.parse_args()
OUT = os.path.expanduser("~/eg2-lab/applab/runs/t2_bench.jsonl")


def ram_mb():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return (m["MemTotal"] - m["MemAvailable"]) / 1024


class Power:
    """tegrastats VDD_IN(보드 전체 입력 전력, mW)을 1초마다 읽어 평균"""
    def __init__(self):
        self.p = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=subprocess.PIPE, text=True)
        self.v, self.on = [], False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            m = re.search(r"VDD_IN (\d+)mW", line)
            if m and self.on:
                self.v.append(int(m.group(1)))

    def close(self):
        self.p.terminate()


pw = Power(); time.sleep(1.2)
pw.on = True; time.sleep(3); idle_mw = np.mean(pw.v) if pw.v else None; pw.v = []; pw.on = False
ram0 = ram_mb()
torch.cuda.reset_peak_memory_stats()
t = time.time(); model = YOLO(a.weights, task="detect"); load_s = time.time() - t
kw = dict(imgsz=640, device=0, verbose=False, half=a.half)
for _ in range(30):                       # 워밍업 (엔진 초기화·CUDA 커널 준비)
    model.predict(a.img, **kw)
inf, tot = [], []
pw.on = True
for _ in range(a.runs):
    t0 = time.perf_counter(); r = model.predict(a.img, **kw)[0]; t1 = time.perf_counter()
    inf.append(r.speed["inference"]); tot.append((t1 - t0) * 1000)
pw.on = False
ram1 = ram_mb()
res = {"case": a.name, "weights": os.path.basename(a.weights),
       "size_mb": round(os.path.getsize(a.weights) / 1e6, 1), "load_s": round(load_s, 1),
       "infer_avg_ms": round(float(np.mean(inf)), 2), "infer_p95_ms": round(float(np.percentile(inf, 95)), 2),
       "e2e_avg_ms": round(float(np.mean(tot)), 2), "e2e_p95_ms": round(float(np.percentile(tot, 95)), 2),
       "fps_infer": round(1000 / float(np.mean(inf)), 1), "fps_e2e": round(1000 / float(np.mean(tot)), 1),
       "power_mw": round(float(np.mean(pw.v))) if pw.v else None, "idle_mw": round(idle_mw) if idle_mw else None,
       "ram_delta_mb": round(ram1 - ram0), "torch_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6),
       "persons": int((r.boxes.cls == 0).sum())}
pw.close()
if a.val:
    m = model.val(data=a.val, imgsz=640, batch=1, device=0, half=a.half, plots=False, verbose=False)
    res |= {"map50_95": round(float(m.box.map), 4), "map50": round(float(m.box.map50), 4)}
print(json.dumps(res, ensure_ascii=False), flush=True)
open(OUT, "a").write(json.dumps(res, ensure_ascii=False) + "\n")
