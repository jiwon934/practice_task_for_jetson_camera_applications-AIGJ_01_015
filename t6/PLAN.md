# 과제 6 설계 초안 (2026-10-30 15:05): A1 재실 인원 + A4 근접 경고를 한 서비스로

## 확인한 환경
- linger: `loginctl show-user jiwon934 -p Linger` → Linger=yes. systemd --user가 재부팅 뒤 로그인 없이 시작됨 (sudo 필요 없음)
- 자원 측정: tegrastats는 sudo 없이 됨 (/usr/bin/tegrastats). jtop은 없음 (jtop 모듈·/run/jtop.sock 없음, 설치하려면 sudo) → tegrastats로 측정
- 메모리: 전체 7485 MB 중 사용 약 5.1 GB (팀원 서비스 포함), 사용 가능 약 2.4 GB
- 이미 쓰는 포트: 1883(팀원 브로커), 8000, 8020, 8030, 8031, 18001, 18002 등. 8011은 쓰지 않음
- ~/.config/systemd/user에 collector·infer·influxd 유닛이 있음 → 건드리지 않음. 새 유닛 이름은 t6-*로

## 프로세스 구성 (제안)
- t6-vision.service: 한 프로세스, TensorRT 엔진·CUDA 컨텍스트 하나를 공유 (컨텍스트마다 약 0.5 GB라 메모리가 빠듯해서)
  - 스레드 A1: IP 카메라 → yolo11n fp16 960, 2x2 타일 + 포함 억제(--contain 0.6), 구역 JSON, 10초 median
  - 스레드 A4: USB 스테레오(2.4/2.2) → 정류 → yolo11n fp16 640 + SGBM(CPU) → 사람별 거리 (무늬 게이트·유효 비율·1초 중앙값) → near(<1.0 m)/mid/far(>3.0 m) 상태
  - 카메라마다 독립 읽기 스레드 + 자동 재연결. 한 카메라를 뽑아도 다른 파이프라인은 계속 돌고, status=offline로 표시
  - REST :8020 (같은 프로세스): GET /v1/occupancy, /v1/proximity, /v1/health — ts, age_s, confidence(유효 비율), 카메라 상태
  - MQTT 127.0.0.1:1883 (발행만): jiwon934/t6/occupancy (retain, 10초마다), jiwon934/t6/proximity (retain, 상태가 바뀔 때), jiwon934/t6/status/<cam> (retain, LWT offline)
- ExecStartPre: v4l2-ctl로 두 USB 카메라 AF 끔·초점 0·노출·화이트밸런스 고정·전원 주파수 60 Hz
- Restart=always, RestartSec=5. 재부팅 뒤 3분 안에 올라오는지 측정
- --ops: 이미지 저장 0 (스냅샷·스테레오 원본·실패 시험 원본·화면 JPEG·디버그 크롭·/frame.jpg 화면 서버 모두 끔)

## GPU 예산 (추정, 실측으로 확인할 것)
- 단독 측정(bench): yolo11n TRT fp16 640 추론 4.0 ms, 전처리 포함 15.5 ms
- 지금 같이 돌 때 과제 3(960, 타일 4장): 프레임당 약 77~80 ms (10 fps)
- A1: 5 fps × 타일 4장 × 약 9 ms ≈ 180 ms/s → GPU 약 18% (10 fps면 약 36%)
- A4: 8 fps × 640 1장 × 약 4~6 ms ≈ 40 ms/s → GPU 약 4%. SGBM은 CPU (640x360에서 25~60 ms → 10 fps면 코어 하나의 25~60%)
- 합계 GPU 약 22~40%, 메모리: 프로세스 하나 + 엔진 2개 약 0.7~0.9 GB → 사용 가능 2.4 GB 안. LLM(+0.5~1.9 GB)은 같이 띄우면 부족 → 운영 모드에서는 LLM 없음
