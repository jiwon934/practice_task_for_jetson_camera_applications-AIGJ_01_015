# 심화 B1 — 하드웨어 복원(NVDEC) vs CPU 복원 (2026-10-29 14:0x, team4)

조건: Jetson Orin Nano Super 8GB, MAXN_SUPER, 호스트 실행. 스크립트 ~/eg2-lab/applab/b1/ (gst_run.py, cv_run.py, run_all.sh), 원자료 runs/b1_raw.jsonl.
입력: (1) 1080p 시험 영상 2개 — IP 카메라 강의실 화면 60초를 1920x1080 25fps로 줄여 만든 H.264(x264 superfast 8 Mbps, GOP 50, cls1080_h264.mkv)와 같은 영상의 H.265(x265 superfast 6 Mbps, cls1080_h265.mkv), 각 1,531장. (2) IP 카메라 stream1 RTSP 그대로 (H.265 2560x1440 25fps, 두 번째 세션).
복원 방법 (모두 결과를 BGR(x) 메모리 프레임까지 만든 뒤 버림):
- HW: GStreamer `h26xparse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! fakesink` (NVDEC + VIC 색 변환)
- FF: OpenCV 5.0.0 `VideoCapture(CAP_FFMPEG).read()` → BGR (CPU, FFmpeg 멀티스레드)
- de265: GStreamer `libde265dec ! videoconvert ! BGRx` (H.265 CPU 두 번째 기준. H.264 CPU 복원기는 GStreamer에 없음: gst-libav 미설치)
CPU%: 측정 프로세스가 쓴 CPU 시간 ÷ 경과 시간 (코어 1개 = 100%, 괄호는 6코어 기준). 시스템 CPU%는 /proc/stat 전체 — 같은 시각 :8020 뷰어(약 1.7코어)·:8030 웹캠 뷰어·다른 작업이 함께 돌아 바닥값이 약 45%.

## 1) 최대 속도 (파일, 동기화 없음)
| 입력 | 방법 | FPS | 프로세스 CPU | 1프레임당 CPU 시간 |
|---|---|---|---|---|
| 1080p H.264 | HW | 168.8 | 57% (9.5%) | 3.4 ms |
| 1080p H.264 | FF | 196.4 | 279% (46.5%) | 14.2 ms |
| 1080p H.265 | HW | 164.2 | 58% (9.6%) | 3.5 ms |
| 1080p H.265 | FF | 134.6 | 241% (40.2%) | 17.9 ms |
| 1080p H.265 | de265 | 41.7 | 193% (32.1%) | 46 ms |

## 2) 실시간 25 fps (파일을 재생 속도로, 60초)
| 입력 | 방법 | FPS | 프로세스 CPU (코어 1개 / 6코어) | 시스템 CPU |
|---|---|---|---|---|
| 1080p H.264 | HW | 24.9 | **10.1% / 1.7%** | 46.1% |
| 1080p H.264 | FF | 25.0 | 43.6% / 7.3% | 49.4% |
| 1080p H.265 | HW | 24.9 | **10.4% / 1.7%** | 45.9% |
| 1080p H.265 | FF | 25.0 | 53.0% / 8.8% | 48.8% |
| 1080p H.265 | de265 | 24.9 | 126.8% / 21.1% | 55.3% |

## 3) IP 카메라 RTSP stream1 (H.265 2560x1440 25 fps, 60초)
| 방법 | 받은 프레임 | FPS | 프로세스 CPU (코어 1개 / 6코어) | 시스템 CPU |
|---|---|---|---|---|
| HW | 1,496 | 25.1 | **15.2% / 2.5%** | 46.3% |
| FF | 1,517 | 25.3 | 56.0% / 9.3% | 52.0% |
| de265 | 1,472 | 24.7 | 122.8% / 20.5% | 60.1% |

## 해석
- 같은 25 fps를 낼 때 NVDEC은 CPU를 1080p에서 약 0.1코어, 1440p에서 0.15코어만 씀 → FFmpeg CPU 복원의 1/4~1/5, libde265의 1/8~1/12. 남는 CPU를 추론 전·후처리에 쓸 수 있음.
- 최대 속도는 HW가 H.264·H.265 모두 약 165–170 fps로 같음 → 한계는 NVDEC이 아니라 BGRx 변환·메모리 복사(nvvidconv → 시스템 메모리) 쪽. FFmpeg H.264는 6코어를 거의 다 써서 196 fps까지 나오지만 그동안 다른 일을 못 함(시스템 CPU 86%).
- H.265는 CPU로 풀면 H.264보다 비쌈(FF 1프레임 14→18 ms). libde265는 FFmpeg보다 약 2.5배 느려 1440p에서는 25 fps를 겨우 맞춤(24.7 fps, 프레임 일부 놓침).
- tegrastats에 NVDEC 사용률 줄은 안 나옴(일반 사용자 권한에서 표시 안 됨). 대신 nvv4l2decoder가 NvMMLite BlockType 261(NVDEC)로 열리는 것을 로그로 확인.
- 남은 과제: OpenCV(~/eg2)는 GStreamer 지원 없이 빌드돼 cv2에서 바로 NVDEC을 못 씀 → 추론 코드에서 HW 복원을 쓰려면 GStreamer appsink(시스템 python3 + gi)로 프레임을 받거나 GStreamer 포함 OpenCV가 필요.
