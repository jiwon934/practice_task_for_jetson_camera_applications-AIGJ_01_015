#!/bin/bash
# B2: 리눅스 PC(또는 젯슨)의 USB 웹캠을 H.264로 인코딩해 MediaMTX로 보냄 → 다른 장비는 rtsp://<이 PC>:8554/live/<이름> 으로 받음 (rtmp2sink는 앱/스트림 두 단계 주소가 필요).
# macOS fake_camera.sh(webcam 모드)의 리눅스판. 젯슨의 ffmpeg는 파일·파이프만 되는 축소판이라 GStreamer로 보냄.
# 이 젯슨 GStreamer에는 rtspclientsink가 없어 RTMP(rtmp2sink → MediaMTX :1935)로 넣고, MediaMTX가 같은 경로를 RTSP(:8554)로 내보냄.
# 사용: ./fake_camera_linux.sh [-d 장치] [-s 1280x720] [-f 30] [-b 4000] [-n webcam] [-H 127.0.0.1]
#   -d  웹캠 장치 (기본: 첫 번째 /dev/v4l/by-path/*video-index0)
#   -s  해상도, -f fps, -b 비트레이트(kbps), -n 스트림 이름, -H MediaMTX 주소
# 끊기면 2초 뒤 자동으로 다시 시작 (웹캠을 뽑았다 꽂아도 계속 시도). Ctrl+C로 끝.
set -u
DEV=$(ls /dev/v4l/by-path/*video-index0 2>/dev/null | head -1); SIZE=1280x720; FPS=30; KBPS=4000; NAME=webcam; HOST=127.0.0.1
while getopts "d:s:f:b:n:H:" o; do case $o in
  d) DEV=$OPTARG;; s) SIZE=$OPTARG;; f) FPS=$OPTARG;; b) KBPS=$OPTARG;; n) NAME=$OPTARG;; H) HOST=$OPTARG;;
  *) sed -n 2,9p "$0"; exit 1;; esac; done
W=${SIZE%x*}; H=${SIZE#*x}
[ -e "$DEV" ] || { echo "웹캠 장치 없음: $DEV"; exit 1; }
# 이미 다른 프로그램이 쓰는 중이면 열리지 않음 (예: view.py :8030)
if fuser "$(readlink -f "$DEV")" >/dev/null 2>&1; then echo "주의: $DEV 를 다른 프로세스가 쓰는 중"; fi
DEC="jpegdec ! videoconvert"   # nvjpegdec은 이 웹캠 MJPG와 형식 협상이 안 돼(not-negotiated) CPU 복원 사용
trap "echo 끝; exit 0" INT TERM
echo "$(date +%T) 송출: $DEV ${W}x${H}@${FPS} ${KBPS}kbps → rtmp://$HOST:1935/live/$NAME  (받는 쪽: rtsp://<이 PC>:8554/live/$NAME)"
while true; do
  gst-launch-1.0 -e v4l2src device="$DEV" do-timestamp=true \
    ! image/jpeg,width=$W,height=$H,framerate=$FPS/1 ! $DEC ! video/x-raw,format=I420 \
    ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=$KBPS key-int-max=$FPS \
    ! video/x-h264,profile=baseline ! h264parse ! flvmux streamable=true ! rtmp2sink location="rtmp://$HOST:1935/live/$NAME"
  echo "$(date +%T) 송출 끊김 (종료 코드 $?), 2초 뒤 다시 시작"; sleep 2
done
