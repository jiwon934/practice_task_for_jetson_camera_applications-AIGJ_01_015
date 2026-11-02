# 과제 2-4 — COCO 밖 물체 × 3가지 방법 (2026-10-29 18:41–19:00, team4)

입력: 코너 웹캠 C920 (포트 2.2, 1280x720), 책상 위에 물체 1개씩, 각 시험 8초. 스크립트 t2/items.py (화면 :8030 2x2), 원자료 runs/t2_items.csv.
방법과 고정 설정 (5종 모두 같음):
- ① YOLO11n TensorRT FP16 (COCO 80종), conf 0.15. 시험 기록은 사람·의자·책상·노트북 등 배경 큰 물체를 뺀 가장 확실한 박스 (18:41 첫 마이크 시험만 이 제외 전이라 chair가 기록됨).
- ② 재라벨: ①의 작은 물체 박스(최대 3개) 크롭 → EmbeddingGemma 2 이미지 임베딩 → "a photo of a <이름>" 문구(5종 + 흔한 COCO 물건 12개) 유사도 최고값, 0.20 미만이면 YOLO 라벨 유지 (app.py relabel과 같은 방식). 재라벨은 약 0.7–1.4초/회라 8초에 8–13회 기록.
  - 변경 18:51: 처음엔 /ref로 화면 가운데 절반을 참조 사진으로 더했는데, 대부분 책상 바닥이라 어떤 물체든 참조와 비슷하게 나오는 문제(크림 → tape dispenser)를 확인하고 참조 사진을 뺌. **마이크·테이프 ② 결과는 참조 사진 포함 상태**, 크림(18:53)·에어팟은 문구만.
- ③ YOLOE-11s-seg (젯슨에 있는 유일한 YOLOE, 내려받기 없음), 텍스트 프롬프트 = 5종 이름, conf 0.05 (18:46부터; 그 전 0.10).
- 프롬프트 문구 변경: microphone → handheld microphone (18:46), roll of tape → tape dispenser (18:49), airpods case → white earbuds charging case (19:00). 같은 문구를 ②·③ 모두에 사용.

## 시험 기록 (각 방법: 8초 동안 가장 많이 나온 답, 평균 점수, 정답 비율)
| 시각 | 물체 / 프롬프트 | ① YOLO | ② 재라벨 | ③ YOLOE |
|---|---|---|---|---|
| 18:41 | 마이크 / microphone (YOLOE conf 0.10) | chair 0.66 ✗ (배경 제외 전) | microphone 0.86 ✓ 100% | 검출 없음 ✗ |
| 18:42 | 마이크 / microphone (YOLOE conf 0.10) | remote 0.49 ✗ | microphone 0.88 ✓ 100% | 검출 없음 ✗ (기본 설정 기준선) |
| 18:46 | 마이크 / handheld microphone | remote 0.52 ✗ | handheld microphone 0.85 ✓ 100% | handheld microphone 0.08 ✓ 100% |
| 18:48 | 테이프 / roll of tape | remote 0.46 ✗ | roll of tape 0.84 ✓ 78% | hand cream tube 0.46 ✗ |
| 18:49 | 테이프 / tape dispenser | remote 0.44 ✗ | tape dispenser 0.91 ✓ 100% | hand cream tube 0.46 ✗ (정답 3%) |
| 18:51 | 핸드크림 / hand cream tube (참조 사진 포함) | remote 0.39 ✗ | tape dispenser 0.73 ✗ | hand cream tube 0.19 ✓ 100% |
| 18:53 | 핸드크림 / hand cream tube (참조 사진 뺌) | remote 0.44 ✗ | remote 0.67 ✗ | hand cream tube 0.20 ✓ 100% |
| 18:59 | 에어팟 / airpods case | mouse 0.84 ✗ | airpods case 0.73 ✓ 100% | hand cream tube 0.40 ✗ |
| 19:00 | 에어팟 / white earbuds charging case | mouse 0.87 ✗ | white earbuds charging case 0.78 ✓ 100% | hand cream tube 0.36 ✗ |
| 19:05 | 화장품(쿠션 팩트) / cushion compact | remote 0.42 ✗ | cushion compact 0.68 ✓ 70% | white earbuds charging case 0.06 ✗ (정답 3%) |

## 물체별 최종 (각 물체 마지막 시험 기준, 5종)
| 방법 | 마이크 | 테이프 | 핸드크림 | 에어팟 | 화장품 | 정답 |
|---|---|---|---|---|---|---|
| ① YOLO | ✗ remote | ✗ remote | ✗ remote | ✗ mouse | ✗ remote | **0/5** |
| ② 재라벨 | ✓* | ✓* | ✗ remote | ✓ | ✓ | **4/5** |
| ③ YOLOE-11s | ✓ | ✗ (hand cream) | ✓ | ✗ (hand cream) | ✗ (earbuds case) | **2/5** |
(*참조 사진 포함 상태)

- 합격 기준(오픈 어휘 ≥ 4/5): **미달 (2/5)**. 화장품은 점수 0.06으로 거의 못 잡고 잡을 때도 earbuds case로 부름. YOLOE-11s는 작고 밝은 물체(테이프 디스펜서, 에어팟)를 계속 "hand cream tube"로 부름(점수 0.36–0.46으로 오히려 정답보다 높음). 문구를 바꿔도(roll of tape → tape dispenser, airpods case → white earbuds charging case) 그대로 → 작은 모델의 시각-문구 정렬 한계로 판단. 더 큰 yoloe-11l로 재시험은 하지 않음.
- YOLO 그대로는 COCO 밖 물체를 비슷한 모양의 COCO 물건(remote, mouse)으로 부름 — 4종 모두 오인식.
- 재라벨이 4/5로 가장 좋음: YOLO가 박스만 잡아 주면 문구 비교로 대부분 바로잡지만, 후보에 비슷한 COCO 문구(remote)가 있으면 그쪽으로 감(핸드크림).
- 참고: 8단계(10/26, ~/eg2-lab/step8_result_1026.md)의 COCO 밖 4종(건전지 낱개·포장, 드럼·통돌이 세탁기)에서도 YOLOE는 문구 수정 후 3/4, 재라벨은 건전지만 해결.

## 오인식 스냅샷 (14장, runs/ 안, 얼굴 없음)
같은 물체·방법을 다시 시험하면 파일을 덮어써서 아래는 각 파일의 마지막 시험 화면.
- ① YOLO (8): t2_items_cushion_compact_yolo.jpg (remote), t2_items_microphone_yolo.jpg (remote, 18:42), t2_items_handheld_microphone_yolo.jpg (remote), t2_items_roll_of_tape_yolo.jpg (remote), t2_items_tape_dispenser_yolo.jpg (remote), t2_items_hand_cream_tube_yolo.jpg (remote), t2_items_airpods_case_yolo.jpg (mouse), t2_items_white_earbuds_charging_case_yolo.jpg (mouse)
- ② 재라벨 (1): t2_items_hand_cream_tube_relabel.jpg (remote, 18:53)
- ③ YOLOE (5): t2_items_roll_of_tape_yoloe.jpg, t2_items_tape_dispenser_yoloe.jpg, t2_items_airpods_case_yoloe.jpg, t2_items_white_earbuds_charging_case_yoloe.jpg (모두 hand cream tube), t2_items_cushion_compact_yoloe.jpg (white earbuds charging case)

## 2차: YOLOE-11l (19:12–19:18, 같은 5종·같은 문구·conf 0.05, 재라벨은 참조 사진 없이 문구만)
yoloe-11l-seg.pt (Ultralytics 공식 배포, 사용자가 직접 내려받음 67.7 MB). 텍스트 인코더(mobileclip_blt)는 11s와 같은 것 사용. CSV에는 방법 이름 yoloe_11l로 따로 기록, 스냅샷 *_yoloe_11l.jpg.

| 시각 | 물체 / 프롬프트 | ① YOLO | ② 재라벨 (문구만) | ③ YOLOE-11l |
|---|---|---|---|---|
| 19:12 | 마이크 / handheld microphone | remote 0.59 ✗ | remote 0.67 ✗ | handheld microphone 0.15 ✓ 100% |
| 19:13 | 테이프(투명 디스펜서) / tape dispenser | remote 0.47 ✗ | remote 0.67 ✗ | white earbuds charging case 0.60 ✗ |
| 19:16 | 핸드크림 / hand cream tube | remote 0.31 ✗ | hand cream tube 0.77 ✓ 100% | hand cream tube 0.07 ✓ 100% |
| 19:17 | 에어팟 / white earbuds charging case | mouse 0.74 ✗ | white earbuds charging case 0.79 ✓ 100% | white earbuds charging case 0.19 ✓ 100% |
| 19:18 | 화장품 / cushion compact | remote 0.56 ✗ | remote 0.67 ✗ | cushion compact 0.44 ✓ 100% |
| | **합계** | **0/5** | **2/5** | **4/5 → 합격** |

### YOLOE 11s vs 11l (같은 문구·기준값)
| | 마이크 | 테이프 | 핸드크림 | 에어팟 | 화장품 | 정답 | 추론 시간 (화면 처리 중) |
|---|---|---|---|---|---|---|---|
| YOLOE-11s | ✓ 0.08 | ✗ hand cream | ✓ 0.20 | ✗ hand cream | ✗ earbuds case | 2/5 | 약 50 ms |
| YOLOE-11l | ✓ 0.15 | ✗ earbuds case | ✓ 0.07 | ✓ 0.19 | ✓ 0.44 | **4/5** | 약 87 ms |
- 큰 모델(11l)은 시각-문구 정렬이 좋아 11s가 모두 "hand cream tube"로 몰던 물체를 구분함. 대신 추론 약 1.7배 느림(그래도 약 11 fps 가능).
- 남은 실패: 투명 테이프 디스펜서 → 투명·반사가 강한 물체는 둘 다 어려움.
- 재라벨(문구만)은 2/5: 마이크·테이프·화장품을 후보 중 "remote"로 보냄 — 흔한 COCO 문구를 후보에 같이 넣은 설정의 부작용. 1차에서 마이크·테이프를 맞힌 것은 참조 사진 덕(위 18:51 메모).

## 결론
- 오픈 어휘(YOLOE-11l) 4/5 → **합격 기준 충족**. YOLO 그대로는 5종 모두 오인식(0/5).
- 오인식 스냅샷 (최종 파일 기준 17장): ① YOLO 8 (t2_items_{microphone,handheld_microphone,roll_of_tape,tape_dispenser,hand_cream_tube,cushion_compact}_yolo.jpg = remote, t2_items_{airpods_case,white_earbuds_charging_case}_yolo.jpg = mouse), ② 재라벨 3 (t2_items_{handheld_microphone,tape_dispenser,cushion_compact}_relabel.jpg = remote), ③ YOLOE-11s 5 (t2_items_{roll_of_tape,tape_dispenser,airpods_case,white_earbuds_charging_case}_yoloe.jpg = hand cream tube, t2_items_cushion_compact_yoloe.jpg = earbuds case), YOLOE-11l 1 (t2_items_tape_dispenser_yoloe_11l.jpg = earbuds case).
- 참고: 위 "오인식 스냅샷 (14장)" 목록은 1차 시점 기준. 2차에서 일부 파일이 덮어써짐(예: t2_items_hand_cream_tube_relabel.jpg는 이제 정답 화면).
