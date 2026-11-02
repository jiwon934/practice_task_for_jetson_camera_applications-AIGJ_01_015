#!/bin/bash
# 과제 2-1: 엔진 변환이 끝나면 4가지를 같은 조건(640, 사진 1장 300번 + coco128 mAP)으로 측정
cd ~/eg2-lab/applab/t2
export YOLO_AUTOINSTALL=False
while ! grep -q ALL_DONE export_all.log; do sleep 10; done
[ -f yolo11s.engine ] && mv yolo11s.engine yolo11s_fp16.engine
: > ~/eg2-lab/applab/runs/t2_bench.jsonl
B="~/eg2/bin/python bench.py"
~/eg2/bin/python bench.py pytorch_11n_fp32 yolo11n.pt --val coco128_local.yaml
~/eg2/bin/python bench.py trt_11n_fp16 yolo11n_fp16.engine --val coco128_local.yaml
~/eg2/bin/python bench.py trt_11n_int8 yolo11n_int8.engine --val coco128_local.yaml
~/eg2/bin/python bench.py trt_11s_fp16 yolo11s_fp16.engine --val coco128_local.yaml
~/eg2/bin/python bench.py pytorch_11n_fp16 yolo11n.pt --half --val coco128_local.yaml
echo BENCH_DONE
