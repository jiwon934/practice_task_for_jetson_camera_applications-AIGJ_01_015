"""과제 2-1: 엔진 하나 만들기. 사용: python export_one.py 가중치.pt fp16|int8
INT8 보정 데이터는 로컬 coco128 (coco128_local.yaml, 128장). 시스템 TensorRT 10.16 사용 (YOLO_AUTOINSTALL=False로 pip 자동 설치 막음)."""
import os, shutil, sys, time
os.environ["YOLO_AUTOINSTALL"] = "False"
from ultralytics import YOLO

w, tag = sys.argv[1], sys.argv[2]
kw = dict(format="engine", imgsz=640, device=0, workspace=4)
kw |= dict(int8=True, data="coco128_local.yaml") if tag == "int8" else dict(half=True)
t = time.time()
p = YOLO(w).export(**kw)
p = str(p); dst = p.replace(".engine", f"_{tag}.engine"); shutil.move(p, dst)
print("EXPORT_DONE", tag, dst, round(time.time() - t, 1), "s", round(os.path.getsize(dst) / 1e6, 1), "MB", flush=True)
