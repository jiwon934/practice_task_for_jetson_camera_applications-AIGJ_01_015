# 심화 B2 — PC 웹캠을 RTSP로 송출 → 보드에서 실시간 사람 검출 (2026-10-29 14:43, team4)

구성 ("리눅스 PC" 역할도 젯슨이 맡음, 모두 젯슨 안):
```
코너 웹캠 C920 (MJPG 1280x720 30) ─ fake_camera_linux.sh (GStreamer: jpegdec → x264enc zerolatency ultrafast 2 Mbps, GOP 30 → flvmux → rtmp2sink)
   → MediaMTX v1.21.2 (127.0.0.1만: RTMP 1935 입력, RTSP 8554 출력, TCP만, HLS·WebRTC·SRT·MoQ·API 끔)
   → rtsp://127.0.0.1:8554/live/webcam
   → rtsp_infer.py (camlib 최신 1장 리더, FFmpeg CPU 복원 → YOLO11n FP16 GPU, 사람만, imgsz 640) → 결과 화면 :8030 (MJPEG)
```
- 젯슨 ffmpeg는 file/pipe 프로토콜만 있는 축소판이고, GStreamer에 rtspclientsink가 없어 RTSP로 바로 밀어 넣을 수 없음 → RTMP로 넣고 MediaMTX가 RTSP로 내보냄 (윈도·맥 PC라면 ffmpeg `-f rtsp rtsp://<젯슨>:8554/live/webcam`으로 바로 넣으면 됨).
- nvjpegdec는 이 웹캠 MJPG와 형식 협상 실패(not-negotiated) → 송신 쪽 MJPG 복원은 CPU(jpegdec).
- MediaMTX 받기: 공식 GitHub 배포 v1.21.2 linux_arm64 (배포 3주 된 안정판), 배포 checksums.sha256으로 검증 OK, sudo·서비스 등록 없음. 설정 b2/mediamtx/b2.yml.

## 측정 (30초 평균)
| 항목 | 값 |
|---|---|
| 받은 FPS (RTSP) | 29.9 |
| 처리(검출) FPS | 24.9 |
| 추론 시간 (YOLO11n FP16, 1280x720 → 640) | 33.1 ms |
| 프레임 나이 (받은 순간 → 검출·그리기 끝) | 50.4 ms |
| 송신 CPU (gst-launch: MJPG 복원 + x264) | 56% (코어 1개 기준) |
| MediaMTX CPU | 1% |
| 받는 쪽 CPU (FFmpeg 복원 + 전처리 + 그리기 + JPEG) | 109% |
| GPU (tegrastats GR3D_FREQ 평균) | 13.8% |
| 시스템 CPU (6코어 전체, :8020 IP 뷰어 포함) | 57% |

- 처리 FPS가 받은 FPS보다 낮은 이유: 한 장에 약 40 ms(추론 33 + 그리기·인코딩) → 최대 약 25 fps. 최신 1장 리더라 밀린 프레임은 버리고 지연은 쌓이지 않음(프레임 나이 50 ms 유지).
- 1080p 시험 영상에서는 추론 약 43 ms, 15 fps (같은 스크립트, 파일 입력).
- 더 빠르게: TensorRT 엔진(FP16)으로 바꾸면 추론 시간 단축 기대, 받는 쪽 복원을 NVDEC(GStreamer appsink)로 바꾸면 CPU 약 0.4코어 절약 (B1 참고).
- glass-to-glass(웹캠 앞 시계 → :8030 화면)는 아직 안 잼.

## 검출 기준값 conf 0.4 → 0.3 (16:55, 받는 쪽만 다시 시작, 송신·MediaMTX 그대로, 웹캠 각도는 현재가 한계라 그대로)
| conf | 1초 평균 사람 수 (최소–최대) | 처리 FPS | 추론 ms | 프레임 나이 ms | 기록 |
|---|---|---|---|---|---|
| 0.4 | 5.2 (3–8) | 21.7 | 37.1 | 56.3 | runs/b2_conf04_last60.log |
| 0.3 | 6.0 (4–8) | 22.7 | 35.1 | 53.7 | runs/b2_conf03_last60.log |

- 각 60초. 속도 차이는 측정 오차 범위(기준값은 후처리만 바뀜).
- 0.3 화면 확인(16:57): 7개 박스 중 6명은 맞음. 0.4에서 놓치던 왼쪽 검은 후드 사람을 잡지만 같은 사람에 박스가 2개 겹침(0.55, 큰 박스 + 작은 박스) → 오검출 1. 모니터 뒤에 머리만 보이는 사람, 화면 오른쪽 끝 손만 보이는 사람은 여전히 못 잡음.
- 결론: 0.3이 사람 수를 약 15% 더 잡고 겹침 박스가 가끔 생김. 겹침은 NMS iou를 낮추거나(예: 0.5→0.4) 추적(track)으로 줄일 수 있음.
