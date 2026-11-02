# 과제 2-1 — YOLO PyTorch vs TensorRT FP16 / INT8 (2026-10-29 17:3x, team4)

조건: Jetson Orin Nano Super 8GB, MAXN_SUPER, 호스트 실행 (~/eg2: torch 2.14.1+cu132, ultralytics 8.4.184, 시스템 TensorRT 10.16.2), 입력 640x640, batch 1.
- 속도: 같은 사진(media/scene_people.jpg 1장)으로 워밍업 30번 뒤 300번 `model.predict`. "추론"은 ultralytics가 잰 모델 실행 시간, "전체"는 파일 읽기·전처리·추론·후처리(NMS)를 합친 한 번 호출 시간.
- mAP: coco128 128장 (`model.val`, 640, batch 1). coco128은 COCO train 일부라 절대값은 실제보다 높게 나옴 → 4가지 사이 비교용.
- 전력: tegrastats VDD_IN(보드 전체 입력) 0.5초 간격 평균, 측정 300번 동안. 같은 시각 :8020 IP 카메라 뷰어가 돌아 대기 전력이 약 7.3 W.
- 메모리: 모델 불러오기 전후 시스템 사용 메모리 차이(CUDA 문맥 포함, Jetson은 CPU·GPU 메모리 공유).
- 엔진: `model.export(format="engine", imgsz=640, half=True)` / `int8=True, data=coco128 (보정 128장)`. 스크립트 t2/export_one.py, t2/bench.py, 원자료 runs/t2_bench.jsonl.

| 모델 | 파일 | 추론 평균 / p95 (ms) | 전체 평균 / p95 (ms) | FPS 추론 / 전체 | mAP50-95 | mAP50 | 전력 (W) | 메모리 증가 (MB) | 변환 시간 |
|---|---|---|---|---|---|---|---|---|---|
| PyTorch YOLO11n (FP32) | .pt 5.6 MB | 25.2 / 26.8 | 36.0 / 38.0 | 39.6 / 27.8 | 0.508 | 0.675 | 10.0 | 623 | - |
| **TensorRT YOLO11n FP16** | .engine 8.1 MB | **4.0 / 4.3** | 15.5 / 16.7 | **250 / 64** | 0.504 | 0.674 | 10.5 | 553 | 약 7분 |
| TensorRT YOLO11n INT8 | .engine 5.6 MB | 3.3 / 3.5 | 14.6 / 15.9 | 307 / 69 | 0.482 | 0.653 | 10.1 | 569 | 약 10분 (보정 포함) |
| TensorRT YOLO11s FP16 (큰 모델) | .engine 22.0 MB | 6.6 / 7.1 | 20.4 / 22.2 | 151 / 49 | 0.576 | 0.745 | 11.7 | 542 | 약 7분 |
| (참고) PyTorch YOLO11n FP16 | .pt | 26.8 / 28.3 | 37.8 / 39.6 | 37.3 / 26.4 | 0.508 | 0.676 | 9.5 | 787 | - |

## 판정
- 합격 기준 "FP16 추론 ≤ 10 ms, 카메라 FPS 이상": FP16 추론 4.0 ms (p95 4.3) ✔, 전체 처리 64 fps > 카메라 25 fps(IP)·30 fps(웹캠) ✔ → **합격**.
- INT8 mAP 하락: FP16 대비 mAP50-95 0.504 → 0.482 (**−0.023, 약 −4.5%**), mAP50 −0.021. PyTorch 대비 −0.026.

## 해석
- TensorRT FP16은 PyTorch보다 추론 6.3배 빠름(25.2 → 4.0 ms), 정확도는 거의 같음(−0.004). PyTorch를 half로 돌려도 빨라지지 않음(26.8 ms) — batch 1에서는 PyTorch 쪽 커널 호출 부담이 커서 엔진 최적화(층 합치기·커널 고르기)가 차이를 만듦.
- INT8은 FP16보다 추론 0.7 ms(18%)만 더 빠르고 mAP는 4.5% 떨어짐. YOLO11n은 이미 작아 INT8 이득이 작음 → 이 보드에서는 FP16이 균형이 좋음. 보정 데이터가 coco128(일반 사진)이라 실제 강의실 장면으로 보정하면 하락이 줄 수 있음(IP 카메라 프레임 수백 장으로 재보정 예정).
- 전체 시간은 추론의 3~4배: 파일 읽기·전처리(크기 맞추기)·NMS가 CPU에서 약 11 ms. 실시간 앱에서 더 줄이려면 전처리를 GPU로 옮기거나 NVDEC→GPU 메모리 그대로 쓰기(B1 참고).
- YOLO11s FP16: 추론 6.6 ms로 여전히 10 ms 안, mAP50-95 +0.072(0.504→0.576). 실시간 49 fps라 카메라 25 fps에 여유 → 정확도가 필요하면 11s FP16도 가능.
- 전력: 대기 약 7.3 W에서 측정 중 10–11.7 W. 같은 일을 할 때 엔진이 빨리 끝나 1장당 에너지는 TensorRT가 훨씬 적음(전체 시간 기준 PyTorch 약 0.36 J/장 vs FP16 약 0.16 J/장).

## 가상환경 변경 기록 (~/eg2)
- 시스템 TensorRT 10.16 파이썬 모듈을 venv에서 쓰도록 링크: t2/systrt/ (심볼릭 링크) + site-packages/zz_system_tensorrt.pth.
- pip 설치: onnx 1.23.2, onnxslim 0.1.98 (ONNX 변환용), onnxruntime-gpu 1.31.1 + flatbuffers (ultralytics가 자동 설치, 엔진 변환에는 안 씀).
- ultralytics가 tensorrt-cu13(PyPI)도 자동 설치하려 해 중단 → 남은 것 없음 확인. 이후 YOLO_AUTOINSTALL=False로 실행.
- 확인: torch.cuda True, numpy 2.5.4 그대로, cv2 5.0.0 그대로.
