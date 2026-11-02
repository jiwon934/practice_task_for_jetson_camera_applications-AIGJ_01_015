#!/bin/bash
# 사용: beh_wait.sh <mark> : 그 구간에서 '요약' 이벤트(사람이 3초 이상 사라져 최종 판정)가 나올 때까지 최대 60초 기다린 뒤 구간 이벤트 출력
E=~/eg2-lab/applab/runs/t4_events.csv
for i in $(seq 1 60); do grep -q ",요약.*,$1\$" $E 2>/dev/null && break; awk -F, -v m="$1" '$5==m && $4 ~ /요약/' $E | grep -q . && break; sleep 1; done
awk -F, -v m="$1" '$5==m {print $2, $3, $4}' $E
