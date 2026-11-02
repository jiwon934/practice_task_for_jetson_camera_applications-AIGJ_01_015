#!/bin/sh
# t6 시작 전: 두 웹캠 자동 초점 끄고 초점·노출·화이트밸런스 고정, 전원 주파수 60 Hz. 웹캠이 없어도 실패로 끝내지 않음(서비스는 계속 시작)
for p in 2.4 2.2; do
  d=/dev/v4l/by-path/platform-3610000.usb-usb-0:$p:1.0-video-index0
  [ -e "$d" ] || { echo "웹캠 $p 없음 (건너뜀)"; continue; }
  v4l2-ctl -d "$d" -c focus_automatic_continuous=0 -c auto_exposure=1 -c exposure_time_absolute=60 -c gain=30 \
    -c white_balance_automatic=0 -c white_balance_temperature=4500 -c power_line_frequency=2 -c exposure_dynamic_framerate=0 2>&1 || true
  v4l2-ctl -d "$d" -c focus_absolute=0 2>&1 || true   # 자동 초점을 끈 다음에 따로 넣어야 받아들여짐
  echo "웹캠 $p: $(v4l2-ctl -d "$d" -C focus_automatic_continuous,power_line_frequency,exposure_time_absolute 2>/dev/null | tr '\n' ' ')"
done
exit 0
