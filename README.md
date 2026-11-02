# practice_task_for_jetson_camera_applications-AIGJ_01_015

Jetson Orin Nano Super에서 IP 카메라(RTSP)·USB 웹캠으로 실시간 영상 응용을 만든 실습 과제입니다. 입력 안정화(최신 1장 리더·재연결)부터 YOLO TensorRT 최적화, 가상선 카운트, 재실 인원 추정, 자세 기반 동작·행동 판정, 스테레오 거리 측정, 그리고 이를 묶은 12시간 상시 서비스까지 과제 1~6과 심화 B1·B2의 코드와 측정 결과를 담았습니다.

## 폴더 구성

| 폴더 | 과제 | 내용 |
|---|---|---|
| `answers/` | 과제 1~6 답안 | `task1_answers.md` ~ `task6_answers.md`: 과제별 시험 결과, 확인 문제·생각해 볼 점 답안 |
| `P_check.md` | 준비 | 실습 전 점검(P1~P8: 전원 모드, GPU, USB 카메라, MQTT 등) 결과와 GPU 문제 진단 |
| `t0/` | 과제 1 (T0) 카메라 입력 | `camlib.py` 공통 입력(웹캠/RTSP/파일, 최신 1장 리더, 자동 재연결), `probe.py` 입력 점검, `latency.py` 지연 누적 비교, `reconn.py` 재연결 시험, `view.py`/`relay.py` 브라우저 MJPEG 뷰어. `runs/` 지연 측정 CSV |
| `t2/` | 과제 2 객체 인식 | `export_one.py`/`run_bench.sh`/`bench.py` YOLO11n PyTorch vs TensorRT FP16/INT8, `conf_probe.py` 조명×거리별 신뢰도, `line_count.py` ByteTrack 가상선 IN/OUT + MQTT, `items.py` COCO 밖 물체 3가지 방법 비교 |
| `t3/` | 과제 3 재실 인원 | `occupancy.py` 타일 검출 + NMS + 포함 억제 + 구역(`zones/room.json`) 필터로 인원 추정, `label_t3.py`/`eval_occ.py` 정답 기입·평가(MAE), `t3_*.py` 오차 분석 |
| `t4/` | 과제 4 자세·행동 | `pose_app.py` 관절 검출 → 동작(만세·손흔들기·악수) 및 행동 패턴(지나감·힐끗·관심·머무름) 판정, `flicker.py`/`summ.py` 결과 요약 |
| `t5/` | 과제 5 스테레오 거리 | `synth_stereo.py` 합성 장면 검증, `make_board.py`/`capture_pairs.py`/`calib_stereo.py` 체커보드 스테레오 보정, `measure_dist.py` SGBM 거리 측정, `person_dist.py` 사람별 거리, `*_offline.py` 실패 조건 재분석 |
| `t6/` | 과제 6 상시 서비스 | `service.py` 재실 인원(A1) + 근접 경고(A4)를 한 서비스로(REST :8030, MQTT), `t6-vision.service` systemd 유닛, `usb_setup.sh` 웹캠 설정, `mon12h.py`/`summ12h.py` 12시간 연속 운전 기록·요약. 자세한 운영 방법은 `t6/README.md` |
| `b1/` | 심화 B1 | 하드웨어 디코딩(NVDEC, GStreamer) vs CPU(OpenCV FFmpeg) 복원 비교 |
| `b2/` | 심화 B2 | PC 웹캠을 MediaMTX로 RTSP 송출 → 보드에서 실시간 사람 검출(`rtsp_infer.py`) |
| `runs/` | 결과 | 과제별 결과 보고서(`t2_*.md`, `t4_*.md`, `t5_*_report.md`, `b1_decode.md`, `b2_rtsp.md`, `test_log.md`)와 측정 CSV/JSON(재실 인원 하루 기록, 스테레오 보정값, 12시간 운전 기록 등) |

## 실행 방법 (Jetson, 가상환경 `~/eg2`)

```bash
source ~/eg2/bin/activate                # opencv, ultralytics, torch(CUDA), tensorrt, fastapi, paho-mqtt
# IP 카메라 계정은 ~/.config/eg2/ipcam.env (IPCAM_USER, IPCAM_PW, 권한 600)에 두고 코드에는 쓰지 않음
python t0/probe.py --url rtsp:stream1
python t0/view.py --url rtsp:stream1     # 브라우저 :8020
python t2/line_count.py                  # 화면 :8030, MQTT 발행
python t3/occupancy.py
python t5/measure_dist.py
# 상시 서비스
cp t6/t6-vision.service ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now t6-vision
curl -s localhost:8030/v1/health
```

## 올리지 않은 것

카메라 사진·스냅샷·영상, 모델 가중치·TensorRT 엔진(`*.pt`, `*.engine`), 압축 파일, 실행 로그(`*.log`, `*.jsonl`), 2MB가 넘는 기록(`runs/t4_trace.csv`), MediaMTX 키 파일과 IP 카메라 계정 같은 비밀 정보는 일부러 뺐습니다.
