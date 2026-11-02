# 과제 6 운영 메모

## 파일
- service.py: A1+A4 서비스 (REST :8030, MQTT jiwon934/t6/*)
- usb_setup.sh: 시작 전 웹캠 설정 (AF 끔, 60 Hz). 웹캠이 없어도 통과
- t6-vision.service: systemd --user 유닛 (~/.config/systemd/user/에 복사)

## 설치·시작 (과제 3 로거를 내린 뒤; 같은 :8030과 IP 카메라 스트림을 씀)
    cp ~/eg2-lab/applab/t6/t6-vision.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now t6-vision
    journalctl --user -u t6-vision -f

## 멈추기 (웹캠을 팀원에게 돌려줄 때)
    systemctl --user stop t6-vision          # 지금만 멈춤 (재부팅하면 다시 시작)
    systemctl --user disable --now t6-vision # 재부팅 뒤에도 안 시작

## 확인
    curl -s localhost:8030/v1/health
    mosquitto_sub -h 127.0.0.1 -v -t 'jiwon934/t6/#' -C 5

## 이미지가 저장될 수 있는 곳 (과제 6 운영 모드에서는 모두 안 씀)
- t6/service.py: 디스크에 이미지 저장 코드 없음, 화면(jpg) 엔드포인트 없음
- t3/occupancy.py --snap-every (runs/t3_snaps*), :8030 화면·/snapshot.jpg
- t5/measure_dist.py /sample (runs/t5_fail, runs/t5_dist_snaps), person_dist.py /sample (runs/t5_person), capture_pairs.py (runs/t5_calib*)
- t0/view.py --snap (runs/ipcam*), t2/line_count.py 화면 :8030
- 운영 중에는 위 프로그램을 띄우지 않음 → 이미지 저장 0
