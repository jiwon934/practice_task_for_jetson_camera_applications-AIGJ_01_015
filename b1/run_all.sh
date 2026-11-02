#!/bin/bash
# B1 전체 실행: 결과 JSON 한 줄씩 runs/b1_raw.jsonl, tegrastats는 runs/b1_tegrastats.log
cd ~/eg2-lab/applab/b1
OUT=~/eg2-lab/applab/runs/b1_raw.jsonl; : > $OUT
tegrastats --interval 1000 > ~/eg2-lab/applab/runs/b1_tegrastats.log 2>&1 & TG=$!
P=/usr/bin/python3; CV=~/eg2/bin/python
F4="filesrc location=cls1080_h264.mkv ! matroskademux ! h264parse"
F5="filesrc location=cls1080_h265.mkv ! matroskademux ! h265parse"
R5="rtspsrc name=src location=rtsp://192.168.0.145:554/stream1 protocols=tcp latency=200 ! rtph265depay ! h265parse"
HW="nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx"
SW5="libde265dec ! videoconvert ! video/x-raw,format=BGRx"
mark(){ echo "$(date +%T) START $1" >> ~/eg2-lab/applab/runs/b1_tegrastats.log; }
g(){ mark $1; $P gst_run.py $1 "$2 ! fakesink name=sink sync=$3" $4 2>/dev/null | grep '^{' >> $OUT; sleep 3; }
c(){ mark $1; $CV cv_run.py $1 $2 $3 $4 2>/dev/null | grep '^{' >> $OUT; sleep 3; }
# 1) 파일 최대 속도
g max_h264_HW  "$F4 ! $HW" false 120
c max_h264_FF  cls1080_h264.mkv 120
g max_h265_HW  "$F5 ! $HW" false 120
g max_h265_de265 "$F5 ! $SW5" false 120
c max_h265_FF  cls1080_h265.mkv 120
# 2) 파일 실시간 25fps
g rt_h264_HW   "$F4 ! $HW" true 70
c rt_h264_FF   cls1080_h264.mkv 70 --rt
g rt_h265_HW   "$F5 ! $HW" true 70
g rt_h265_de265 "$F5 ! $SW5" true 70
c rt_h265_FF   cls1080_h265.mkv 70 --rt
# 3) IP 카메라 RTSP 2560x1440 H.265 실시간 60초
g rtsp_HW      "$R5 ! $HW" false 60
g rtsp_de265   "$R5 ! $SW5" false 60
c rtsp_FF      rtsp:stream1 60
kill $TG; echo DONE >> $OUT
