# Jetson 카메라 응용 실습 1.5 준비 확인 (P1–P8) — 2026-10-28 15:31, team4 (jiwon934, sudo 안 씀)

| 항목 | 결과 | 실제 출력 / 근거 |
|---|---|---|
| P1 전원 모드 | OK (단 경고) | `nvpmodel -q` → `NV Power Mode: MAXN_SUPER` (pmode 2). 함께 `NVPM ERROR: Error opening /sys/devices/platform/17000000.gpu/devfreq_dev/min_freq` → GPU 주파수 장치가 없음 (P2와 같은 원인) |
| P2 GPU | **FAIL** | 호스트 ~/eg2: `torch 2.14.1+cu132, cuda.is_available() = False`, `nvidia-smi` → `NvRmGpuLibOpen failed, error=4 / No devices were found`, tegrastats GR3D_FREQ 0%. 실습 폴더 /ssd/lab 없음 (/ssd 자체가 없음), run_container.sh 못 찾음. `docker ps` → `permission denied ... /var/run/docker.sock` (jiwon934가 docker 그룹 아님) |
| P3 USB 카메라 | 부분 OK | `v4l2-ctl --list-devices`: C920 2대 (/dev/video0·1 = usb 2.4, /dev/video2·3 = usb 2.2). `lsusb -t`: 두 카메라 모두 **480M** (Realtek USB 2.0 허브 0bda:5489 아래). 기대(5000M)와 다름. tools/usb_camera.sh 없음 |
| P4 TensorRT 엔진 | **FAIL** | IP카메라 객체인식 실습 폴더가 없어 yolo26n_fp16_640 / fp16_960 / int8_640 .engine 없음 (찾은 .engine은 ~/withio/task7의 다른 과제 것뿐) |
| P5 00_probe_stream.py | **FAIL** | 파일 없음 (/, /home/jiwon934, /mnt, /media, /opt, /data 검색) |
| P6 05_view.py | **FAIL** | 파일 없음 |
| P7 MQTT | OK | `/usr/bin/mosquitto_pub`, `/usr/bin/mosquitto_sub` 있음. 1883 포트 LISTEN 중 = 팀원(prettygeun) timeseries의 amqtt 브로커 (PID 2475). 같이 쓸지 팀원과 정해야 함 |
| P8 체커보드 | 건너뜀 | 직접 측정 |

## GPU가 안 되는 원인 (읽기 전용 진단)
- 부팅: 2026-10-28 10:01 (uptime 5:27). 10-23 10:22 부팅 때는 GPU가 됐음 (bench6 GPU 사용률 기록).
- 커널 모듈은 올라와 있음: `nvgpu 6082560 0`, nvmap, host1x 정상. 드라이버도 붙어 있음 (17000000.gpu → gk20a).
- 하지만 `/dev/nvgpu/igpu0/`에 `power` 하나만 있고 ctrl·as·channel 같은 장치 파일이 없음, GPU devfreq(`/sys/devices/platform/17000000.gpu/devfreq_dev`)도 없음, `power/runtime_status = suspended`.
  → 이번 부팅에서 nvgpu 초기화(probe)가 중간에 실패한 상태로 보임. 사용 권한 문제는 아님 (jiwon934는 video·render 그룹, /dev/nvmap 등은 root:video).
- 10-23 이후 apt로 바뀐 NVIDIA·커널·CUDA 패키지 없음 (설치 기록: ffmpeg·v4l-utils·alsa·nmap뿐). nvgpu.ko 날짜 08-07, L4T R39.2.1.
- GPU를 붙잡은 다른 프로세스·컨테이너 없음 (dockerd만 실행, 팀원 infer.py는 GPU 장치 자체가 없어서 못 씀). dmesg·커널 로그는 권한 없어 못 읽음.
- 관리자(sudo)가 할 일: `sudo dmesg | grep -iE "nvgpu|gk20a"`로 초기화 오류 확인 → 재부팅 (안 되면 전원 완전히 껐다 켜기). 계속되면 nvgpu 펌웨어(/lib/firmware) 확인.

## 실습 진행에 필요한 것
1. GPU 복구 (관리자 재부팅) — P2·P4 TensorRT 엔진 만들기에 필수.
2. 실습 코드(/ssd/lab: IP카메라 객체인식 실습/, 스테레오 거리 실습/, tools/) 받기 — 지금 젯슨에 없음. SD 카드(mmcblk0p1 117.7G)가 마운트 안 된 채 있음, 거기 있는지 관리자 확인 필요.
3. docker 쓰려면 jiwon934를 docker 그룹에 추가 (관리자).
4. 카메라 2대가 USB 2.0 허브(480M)에 함께 있음: 640x480 MJPG는 충분했지만 고해상도 2대 동시면 대역폭 부족 가능 → USB 3.0 포트/허브(0bda:0489, Bus 002)로 옮기는 것 권장.
