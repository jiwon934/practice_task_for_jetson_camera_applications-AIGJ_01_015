import time, threading, collections, argparse, csv
import cv2, numpy as np, uvicorn, torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
from pose_model import (KP_CONF, NOSE, LS, RS, LW, RW, body, facing_score,
                        frame_features, load_model, predict)

# applab 과제 4용 (팀원 pose_app_v6.py 복사본): 0.0.0.0:8030, 웹캠 by-path·RTSP(camlib) 입력, --event-log(동작 기록 CSV),
#   --trace(프레임마다 사람별 라벨·보이는 관절 수 CSV, 깜빡임·관절 시험용), GET /mark?name=... (시험 구간 표시)
# 개선판(v6): 문지기 모델(--gate-model)이 '동작 아님(idle/other)'이라고 하면 GRU 판정을 버림
# 개선판(v5): GRU 판정 거부(--ml-th/--ml-margin/--ml-vote), 라벨 제한(--ml-only), 규칙 확인(--ml-rule), other 클래스
# 개선판(v4): 악수 판정에서 박수 치는 사람·손 흔드는 사람 제외, 두 사람이 모두 카메라 정면을 보고 있으면 제외(나란히 서서 박수 치는 청중), 맞잡은 손은 두 사람 사이·팔꿈치보다 많이 위가 아님
# 개선판(v3): 손흔들기를 팔꿈치 기준 아래팔 움직임으로 판정(측면·어깨 높이 흔들기도 잡음, 박수 자세 제외). --wave v2면 이전 방식
# 개선판(v2): 손흔들기 방향 전환 3번 + 손 올림 유지 80%, 손흔들기 1초 유지, --low-wave(어깨 높이 흔들기),
#   악수는 두 손목을 상대 쪽으로 뻗고 0.5초 이상 이어질 때만, 같은 자리 2초 안 재카운트 안 함
# ---------------- 실행 옵션 ----------------
# --raw          : 프레임마다 판정한 결과를 그대로 표시 (9-1 문제 확인용)
# (기본)         : 0.6초 다수결 안정화 + 손 흔들기(시간 패턴) 판정 (9-2 개선)
# --record F.csv : 관절 특징을 CSV로 저장 (9-3 학습 데이터 수집)
# --behavior     : 정면 응시·이동 속도로 행동 패턴(지나감/힐끗 봄/관심 있음/머무름) 판정 (9-5)
# --model M.pt   : 학습한 GRU 모델로 판정 (9-6). 여러 번 지정 가능 (동작용, 행동용)
ap = argparse.ArgumentParser()
ap.add_argument("--cam", default="http://127.0.0.1:8081/stream",
                help="카메라: 숫자면 USB(/dev/videoN), 아니면 영상 주소 (기본: IP 카메라 뷰어)")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--raw", action="store_true")
ap.add_argument("--record", default=None, help="예: data/wave.csv")
ap.add_argument("--behavior", action="store_true")
ap.add_argument("--ml-th", type=float, default=0.7, help="GRU 최고 확률 기준")
ap.add_argument("--ml-margin", type=float, default=0.0, help="GRU 1·2위 확률 차이 기준")
ap.add_argument("--ml-vote", type=int, default=1, help="최근 N번 예측이 모두 같을 때만 인정")
ap.add_argument("--ml-only", default="", help="GRU로 판정할 라벨 (쉼표로 구분, 비우면 전부)")
ap.add_argument("--gate-model", default=None, help="other 클래스가 있는 문지기 모델")
ap.add_argument("--gate-th", type=float, default=0.5, help="문지기가 idle/other라고 할 확률 기준")
ap.add_argument("--ml-rule", action="store_true", help="박수·팔짱을 규칙으로 한 번 더 확인")
ap.add_argument("--ml-rule-labels", default="clap,arms_crossed", help="--ml-rule을 적용할 라벨 (예: clap)")
ap.add_argument("--wave", choices=["v3", "v2"], default="v3", help="손흔들기 판정 방식")
ap.add_argument("--low-wave", action="store_true", help="어깨 높이에서 흔드는 손도 손흔들기로 봄")
ap.add_argument("--model", action="append", default=[], help="예: models/action.pt")
ap.add_argument("--port", type=int, default=8030)
ap.add_argument("--event-log", default=None, help="동작 기록 CSV (ts, time, who, label)")
ap.add_argument("--trace", default=None, help="프레임마다 사람별 라벨 CSV")
args = ap.parse_args()
MODE = "raw" if args.raw else "smooth"

W, H = 1280, 720
SMOOTH_S = 0.6        # 다수결 창(초)
WAVE_S = 1.5          # 손 흔들기 판정 창(초)
BEH_WIN = 15.0        # 행동 패턴 판정 창(초)
FACE_TH = 0.6         # 정면으로 볼 facing 점수
LOOK_S, INTEREST_S, LINGER_S = 1.0, 3.0, 10.0
MOVE_TH = 0.5         # 이동 판정 속도 (어깨 너비/초)
ML_TH = 0.7           # 모델 확률 기준
WAVE_HOLD, HS_MIN_S, HS_COOL_S = 1.0, 0.5, 2.0
WAVE3_KIND = "angle"     # angle: 아래팔 각도 진동, traj: 팔꿈치 기준 손목 궤적 주 방향 진동 (pose_eval_v3.py 비교로 선택)
WAVE3_AMP = 15.0 if WAVE3_KIND == "angle" else 0.2   # 방향 전환으로 볼 최소 진폭(도 / 아래팔 길이)
WAVE3_UP_FRAC, WAVE3_CHANGES = 0.7, 2
LE, RE, LH, RH = 7, 8, 11, 12   # 손흔들기 유지, 악수 최소 지속, 악수 재카운트 쿨다운(초)
EN = {"만세": "BOTH ARMS UP", "한팔들기": "ONE ARM UP", "손흔들기": "WAVE", "악수": "HANDSHAKE",
      "기본": "-", "지나감": "PASSING", "힐끗 봄": "GLANCE", "관심 있음": "INTERESTED",
      "머무름": "LINGER", "관찰 중": "..."}
BEH_RANK = {"관찰 중": 0, "지나감": 1, "머무름": 2, "힐끗 봄": 3, "관심 있음": 4}

det = YOLO("yolo11n-pose.pt")         # 첫 실행 시 자동 다운로드
models = [load_model(p) for p in args.model]
gate = load_model(args.gate_model) if args.gate_model else None
SEQ_LEN = max([m["window"] for m in models] + ([gate["window"]] if gate else []), default=1)
print(f"[pose mode={MODE}] conf={args.conf} behavior={args.behavior} "
      f"models={[(m['task'], m['labels']) for m in models]}")

frame_lock, ev_lock = threading.Lock(), threading.Lock()
latest_jpeg = None
tracks = {}
pairs = {}            # (tid_a, tid_b) -> deque[(t, bool)]
pair_on = set()
pair_start, pair_mid, hs_log = {}, {}, collections.deque(maxlen=50)
events = collections.deque(maxlen=50)
counts = collections.Counter()
beh_counts = collections.Counter()   # 사람(추적 ID) 단위 최종 행동
stats = {"fps": 0.0, "people": 0}
rec = None
if args.record:
    rec_f = open(args.record, "a", newline="")
    rec = csv.writer(rec_f)

def new_track(now):
    return {"raw": collections.deque(maxlen=90), "wave": collections.deque(maxlen=90),
            "beh": collections.deque(maxlen=600), "feat": collections.deque(maxlen=SEQ_LEN),
            "label": "기본", "bstate": "관찰 중", "best": "관찰 중", "ml": {},
            "first": now, "seen": now, "prev": None, "look": 0.0, "n": 0}

def pose_label(k, b):
    (cx, cy), sw = b
    raised = lambda w: k[w, 2] > KP_CONF and k[w, 1] < cy - 0.3 * sw
    lu, ru = raised(LW), raised(RW)
    if lu and ru:
        return "만세"
    if lu or ru:
        return "한팔들기"
    return "기본"

def is_wave(hist, n_frames=0):
    """올린 손목의 x 좌표(어깨 너비로 정규화)가 좌우로 3번 이상 방향을 바꾸고, 창 안 프레임의 80% 이상에서
    손이 올라가 있으면 손 흔들기 (v2: 팔을 내리는 동작 제외)"""
    xs = [x for _, x in hist]
    if len(xs) < 8:
        return False
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > 0.15:
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes >= 3 and len(xs) >= 0.8 * n_frames

def behavior(st, now):
    """정면 연속 시간·누적 시간, 이동 여부, 체류 시간으로 행동 패턴 판정"""
    h = [x for x in st["beh"] if now - x[0] <= BEH_WIN]
    if len(h) < 5:
        return "관찰 중"
    run = 0.0                                   # 끝에서부터 정면 연속 시간
    for i in range(len(h) - 1, 0, -1):
        if h[i][1] < FACE_TH:
            break
        run = h[-1][0] - h[i - 1][0]
    look = sum(h[i][0] - h[i - 1][0] for i in range(1, len(h)) if h[i][1] >= FACE_TH)
    moving = np.median([s for t, _, s in h if now - t < 2.0]) > MOVE_TH
    if run >= INTEREST_S and not moving:
        return "관심 있음"
    if moving and LOOK_S <= look < INTEREST_S:
        return "힐끗 봄"
    if moving:
        return "지나감"
    if now - st["first"] >= LINGER_S:
        return "머무름"
    return "관찰 중"

def wave_v3(st, now, n_frames):
    """팔마다: 흔드는 자세(손목이 팔꿈치보다 위 + 어깨선보다 몸통 0.25배 아래까지, 박수 자세 제외)가 창의 70% 이상이고
    아래팔 움직임이 진폭 이상으로 2~15번 방향을 바꾸면 손흔들기 (15번 초과는 떨림으로 봄)"""
    for arm in (LW, RW):
        h = [(d, a) for t, d, a in st.setdefault("arm3", {}).get(arm, ()) if now - t <= WAVE_S]
        if len(h) < 6 or len(h) < WAVE3_UP_FRAC * n_frames:
            continue
        if WAVE3_KIND == "angle":
            vals = list(np.degrees(np.unwrap(np.radians([a for _, a in h]))))
        else:
            P = np.array([d for d, _ in h], dtype=np.float32)
            P -= P.mean(0)
            vals = list(P @ np.linalg.svd(P, full_matrices=False)[2][0])
        changes, last_dir, anchor = 0, 0, vals[0]
        for x in vals[1:]:
            if abs(x - anchor) > WAVE3_AMP:
                cur = 1 if x > anchor else -1
                if last_dir and cur != last_dir:
                    changes += 1
                last_dir, anchor = cur, x
        if WAVE3_CHANGES <= changes <= 15:
            return True
    return False

UPPER_OFF = list(range(22, 34)) + list(range(45, 51))

@torch.inference_mode()
def predict_full(ck, seq):
    """(1위 라벨, 1위 확률, 1·2위 확률 차이). 상반신 모델(upper)이면 하체 특징을 지움"""
    x = torch.tensor(np.asarray(seq), device="cuda")[None]
    if ck.get("upper"):
        x[..., UPPER_OFF] = 0
    x = (x - ck["mean"]) / ck["std"]
    pr = torch.softmax(ck["model"](x), -1)[0]
    top = torch.topk(pr, min(2, len(pr)))
    second = float(top.values[1]) if len(top.values) > 1 else 0.0
    return ck["labels"][int(top.indices[0])], float(top.values[0]), float(top.values[0]) - second

def rule_ok(st, k, lab):
    """--ml-rule: 박수는 박수 치는 사람일 때만, 팔짱은 손목이 반대쪽 팔꿈치 근처(아래팔 0.7배 안)일 때만"""
    if lab not in args.ml_rule_labels.split(","):
        return True
    if lab == "clap":
        return st.get("clapping", False)
    if lab == "arms_crossed":
        near = []
        for w, e_other, e_own in ((LW, RE, LE), (RW, LE, RE)):
            if k[w, 2] > KP_CONF and k[e_other, 2] > KP_CONF and k[e_own, 2] > KP_CONF:
                fl = max(np.linalg.norm(k[w, :2] - k[e_own, :2]), 5.0)
                near.append(np.linalg.norm(k[w, :2] - k[e_other, :2]) < 0.7 * fl)
        return any(near)
    return True

def majority(dq, now, window):
    vals = [v for t, v in dq if now - t <= window]
    if not vals:
        return None, 0.0
    v, n = collections.Counter(vals).most_common(1)[0]
    return v, n / len(vals)

EV_F = open(args.event_log, "a", newline="") if args.event_log else None
EV = csv.writer(EV_F) if EV_F else None
TR_F = open(args.trace, "a", newline="") if args.trace else None
TR = csv.writer(TR_F) if TR_F else None
MARK = {"name": ""}


def log_event(now, who, label):
    if EV:
        EV.writerow([round(now, 3), time.strftime("%H:%M:%S"), who, label, MARK["name"]]); EV_F.flush()
    with ev_lock:
        events.appendleft({"ts": now, "who": who, "label": label})
        counts[label] += 1

def close_track(tid, st, now):
    """추적이 끝난 사람의 요약을 남긴다"""
    if not args.behavior or st["best"] == "관찰 중":
        return
    dwell = st["seen"] - st["first"]
    with ev_lock:
        events.appendleft({"ts": now, "who": f"#{tid}",
                           "label": f"요약: 체류 {dwell:.0f}초, 정면 {st['look']:.0f}초 → {st['best']}"})
        beh_counts[st["best"]] += 1

def not_raised(k, w):
    """손목이 팔꿈치보다 아래팔 0.3배 넘게 위에 있지 않음 (악수는 손을 앞으로 뻗은 자세)"""
    e = LE if w == LW else RE
    if k[e, 2] <= KP_CONF:
        return True
    fl = max(np.linalg.norm(k[w, :2] - k[e, :2]), 5.0)
    return k[w, 1] >= k[e, 1] - 0.3 * fl

def handshake_pairs(kps, ids, now):
    """두 사람의 손목이 가까이(어깨 너비 0.5배 이내) 있고 높이가 가슴~허리 사이면 악수 후보"""
    found = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = kps[i], kps[j]
            ba, bb = body(a), body(b)
            if ba is None or bb is None:
                continue
            sw = (ba[1] + bb[1]) / 2
            best, bw = 1e9, None
            for wa in (LW, RW):
                for wb in (LW, RW):
                    if a[wa, 2] > KP_CONF and b[wb, 2] > KP_CONF:
                        mid_y = (a[wa, 1] + b[wb, 1]) / 2
                        if ba[0][1] < mid_y < ba[0][1] + 2.5 * ba[1]:
                            d = np.linalg.norm(a[wa, :2] - b[wb, :2])
                            if d < best:
                                best, bw = d, (wa, wb)
            # v4: 두 사람이 떨어져 있고(어깨 너비 0.5배 이상), 각자 손목을 상대 쪽으로 뻗었고, 맞잡은 손이 두 사람 사이에 있고,
            #     손목이 팔꿈치보다 많이 위가 아니고, 두 사람 모두 손흔들기·박수 중이 아닐 때만
            s = 1 if bb[0][0] > ba[0][0] else -1
            hit = False
            if bw is not None and best < 0.5 * sw:
                lo, hi = sorted((ba[0][0], bb[0][0]))
                mx = (a[bw[0], 0] + b[bw[1], 0]) / 2
                busy = lambda t: (now - tracks.get(t, {}).get("wave_t", -9.0) < 1.0) or tracks.get(t, {}).get("clapping", False)
                hit = (abs(ba[0][0] - bb[0][0]) >= 0.5 * sw
                       and (a[bw[0], 0] - ba[0][0]) * s >= 0.3 * ba[1] and (b[bw[1], 0] - bb[0][0]) * -s >= 0.3 * bb[1]
                       and lo + 0.15 * (hi - lo) <= mx <= hi - 0.15 * (hi - lo)
                       and not_raised(a, bw[0]) and not_raised(b, bw[1])
                       and not busy(ids[i]) and not busy(ids[j])
                       and not (facing_score(a) >= 0.6 and facing_score(b) >= 0.6))
            key = tuple(sorted((ids[i], ids[j])))
            pairs.setdefault(key, collections.deque(maxlen=60)).append((now, hit))
            pair_mid[key] = ((a[bw[0], :2] + b[bw[1], :2]) / 2, sw) if bw is not None else None
            found.add(key)
    return found

def update_person(tid, k, now):
    st = tracks.setdefault(tid, new_track(now))
    dt = now - st["seen"]
    st["seen"] = now
    b = body(k)
    if b is None:
        return st
    raw = pose_label(k, b)
    fa = facing_score(k)
    spd = 0.0 if st["prev"] is None else float(np.linalg.norm(b[0] - st["prev"]) / b[1] / max(dt, 1e-3))
    st["prev"] = b[0]
    st["raw"].append((now, raw))
    st["beh"].append((now, fa, spd))
    if fa >= FACE_TH and st["n"] > 0:
        st["look"] += dt
    st["n"] += 1
    w = LW if k[LW, 1] < k[RW, 1] else RW
    low_up = args.low_wave and k[w, 2] > KP_CONF and k[w, 1] < b[0][1] + 0.2 * b[1]
    if raw in ("한팔들기", "만세") or low_up:  # 손 흔들기용: 올라간 손목의 x (어깨 중심 기준, 어깨 너비 단위)
        st["wave"].append((now, (k[w, 0] - b[0][0]) / b[1]))
    # v3: 팔꿈치 기준 아래팔 (아래팔 길이로 나눔, 측면에서도 0이 되지 않음)
    torso = abs((k[LH, 1] + k[RH, 1]) / 2 - b[0][1]) if k[LH, 2] > KP_CONF and k[RH, 2] > KP_CONF else 2.5 * b[1]
    fal = [max(np.linalg.norm(k[a, :2] - k[e, :2]), 5.0) for a, e in ((LW, LE), (RW, RE))]
    together = k[LW, 2] > KP_CONF and k[RW, 2] > KP_CONF and np.linalg.norm(k[LW, :2] - k[RW, :2]) < 0.6 * np.mean(fal)
    tog = st.setdefault("tog", collections.deque(maxlen=90))
    tog.append((now, bool(together)))
    h = [x for t, x in tog if now - t <= WAVE_S]
    st["clapping"] = len(h) >= 5 and sum(h) >= 0.25 * len(h)      # v4: 박수 치는 사람
    for (wj, ej), fl in zip(((LW, LE), (RW, RE)), fal):
        if (not together and k[wj, 2] > KP_CONF and k[ej, 2] > KP_CONF
                and k[wj, 1] < k[ej, 1] and k[wj, 1] < b[0][1] + 0.25 * torso):
            d = (k[wj, :2] - k[ej, :2]) / fl
            st.setdefault("arm3", {}).setdefault(wj, collections.deque(maxlen=90)).append(
                (now, tuple(d), float(np.degrees(np.arctan2(d[1], d[0])))))
    feat = frame_features(k, b, fa, spd)
    st["feat"].append(feat)
    if rec:
        rec.writerow([round(now, 3), tid, raw, round(fa, 3), round(spd, 3)] + feat[:51].round(3).tolist())

    # ---- 동작 (규칙) ----
    if MODE == "raw":
        label = raw
    else:
        v, frac = majority(st["raw"], now, SMOOTH_S)
        label = v if frac >= 0.6 else st["label"]
        recent = [(t, x) for t, x in st["wave"] if now - t <= WAVE_S]
        n_win = sum(1 for t, _ in st["raw"] if now - t <= WAVE_S)
        if args.wave == "v3":
            hit = label != "만세" and wave_v3(st, now, n_win)
        else:
            hit = (label == "한팔들기" or (args.low_wave and label == "기본")) and is_wave(recent, n_win)
        if hit:
            label, st["wave_t"] = "손흔들기", now
        if st["label"] == "손흔들기" and label != "만세" and now - st.get("wave_t", -9.0) < WAVE_HOLD:
            label = "손흔들기"                # v2: 1초 유지 (한팔들기↔손흔들기 깜빡임 방지)

    # ---- 학습 모델 (5프레임마다) ----
    if models and st["n"] % 5 == 0:
        seq = list(st["feat"])
        for m in models:
            if len(seq) >= m["window"]:
                lab, p, margin = predict_full(m, seq[-m["window"]:])
                only = [x for x in args.ml_only.split(",") if x]
                ok = (p >= args.ml_th and margin >= args.ml_margin and (not only or lab in only)
                      and (not args.ml_rule or rule_ok(st, k, lab)))
                if ok and gate is not None and len(seq) >= gate["window"]:
                    glab, gp, _ = predict_full(gate, seq[-gate["window"]:])
                    if glab in ("idle", "other") and gp >= args.gate_th:
                        ok = False
                votes = st.setdefault("votes", collections.deque(maxlen=max(args.ml_vote, 1)))
                votes.append(lab if ok else None)
                st["ml"][m["task"]] = (lab, p) if ok and len(votes) == votes.maxlen and all(v == lab for v in votes) else None
    ml_act = st["ml"].get("action")
    if ml_act and ml_act[0] not in ("idle", "other"):
        label = ml_act[0]

    if label != st["label"]:
        if label != "기본":
            log_event(now, f"#{tid}", label)
        st["label"] = label

    # ---- 행동 패턴 ----
    if args.behavior:
        bs = behavior(st, now)
        ml_beh = st["ml"].get("behavior")
        if ml_beh:
            bs = ml_beh[0]
        if bs != st["bstate"]:
            if bs != "관찰 중":
                log_event(now, f"#{tid}", bs)
            st["bstate"] = bs
        if BEH_RANK.get(bs, 1) > BEH_RANK.get(st["best"], 0):
            st["best"] = bs
    return st

def capture_loop():
    global latest_jpeg
    if args.cam.startswith("rtsp:"):      # IP 카메라 (camlib: 계정은 ~/.config/eg2/ipcam.env에서)
        import sys, os; sys.path.insert(0, os.path.expanduser("~/eg2-lab/applab/t0"))
        from camlib import open_cap
        cap = open_cap(args.cam)
    elif args.cam.isdigit() or args.cam.startswith("/dev/"):   # USB 웹캠 (번호 또는 by-path)
        cap = cv2.VideoCapture(int(args.cam) if args.cam.isdigit() else args.cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    else:                                 # IP 카메라 (MJPEG/RTSP 주소)
        cap = cv2.VideoCapture(args.cam)
    assert cap.isOpened(), "카메라를 열 수 없음 (Q11 참고)"
    t0, n = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1); continue
        r = det.track(frame, persist=True, conf=args.conf, half=True, verbose=False)[0]
        now = time.time()
        vis = r.plot(labels=False)        # 박스 + 관절 뼈대
        ids, kps, boxes = [], [], []
        if r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().tolist()
            kps = np.concatenate([r.keypoints.xy.cpu().numpy(),
                                  r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)
            boxes = r.boxes.xyxy.int().tolist()
        stats["people"] = len(ids)

        for tid, k, (x1, y1, x2, y2) in zip(ids, kps, boxes):
            st = update_person(tid, k, now)
            if TR:
                vis_n = int((k[:, 2] > KP_CONF).sum())
                TR.writerow([round(now, 3), tid, st["label"], st.get("bstate", ""), vis_n,
                             round(float(k[[5, 7, 9, 11, 13, 15], 2].mean()), 3), round(float(k[[6, 8, 10, 12, 14, 16], 2].mean()), 3),
                             round(float(facing_score(k)), 2), MARK["name"], st["raw"][-1][1] if st["raw"] else ""])
            txt = f"#{tid} {EN.get(st['label'], st['label'])}"
            if args.behavior:
                txt += f" | {EN.get(st['bstate'], st['bstate'])}"
            color = (0, 200, 255) if st["label"] != "기본" or st["bstate"] == "관심 있음" else (0, 255, 0)
            cv2.putText(vis, txt, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ---- 2인 동작 (악수) ----
        active = handshake_pairs(kps, ids, now) if len(ids) >= 2 else set()
        for key in list(pairs):
            if key not in active:
                pairs.pop(key); pair_on.discard(key); pair_start.pop(key, None); continue
            if MODE == "raw":
                on = pairs[key][-1][1]
            else:
                v, frac = majority(pairs[key], now, SMOOTH_S)
                on = bool(v) and frac >= 0.6
            if on:
                pair_start.setdefault(key, now)
            else:
                pair_start.pop(key, None)
            on = on and now - pair_start.get(key, now) >= HS_MIN_S        # v2: 0.5초 이상 이어져야
            if on and key not in pair_on:
                pm = pair_mid.get(key)
                near = pm is not None and any(now - t < HS_COOL_S and np.linalg.norm(m - pm[0]) < 2 * pm[1]
                                              for t, m in hs_log)            # v2: 같은 자리 2초 안 재카운트 안 함
                if not near:
                    log_event(now, f"#{key[0]}+#{key[1]}", "악수")
                if pm is not None:
                    hs_log.append((now, pm[0]))
            (pair_on.add if on else pair_on.discard)(key)
        for a, b_ in pair_on:
            if a in ids and b_ in ids:
                ba, bb = boxes[ids.index(a)], boxes[ids.index(b_)]
                x, y = (ba[0] + bb[2]) // 2 - 80, min(ba[1], bb[1]) + 30
                cv2.putText(vis, EN["악수"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

        # 3초 이상 안 보인 사람은 추적 종료 + 요약
        for tid in [t for t, s in tracks.items() if now - s["seen"] > 3]:
            close_track(tid, tracks.pop(tid), now)

        n += 1
        if now - t0 >= 1:
            stats["fps"], t0, n = n / (now - t0), now, 0
            if rec:
                rec_f.flush()
            if TR_F:
                TR_F.flush()
        cv2.putText(vis, f"{stats['fps']:.1f} fps | {MODE} | people {stats['people']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()

app = FastAPI()

@app.get("/stream")
def stream():
    def gen():
        while True:
            with frame_lock:
                f = latest_jpeg
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
def get_events():
    with ev_lock:
        total = sum(beh_counts.values())
        return {"mode": MODE + (" +behavior" if args.behavior else "") + (" +model" if models else ""),
                "stats": stats, "counts": dict(counts), "events": list(events)[:20],
                "beh": dict(beh_counts),
                "interest_rate": round(beh_counts["관심 있음"] / total, 2) if total else None,
                "now": {f"#{t}": s["label"] + (f" / {s['bstate']}" if args.behavior else "")
                        for t, s in tracks.items()}}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 동작 인식</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img{width:100%;max-width:960px;border-radius:8px}table{border-collapse:collapse;margin-top:8px}
td,th{border:1px solid #444;padding:4px 10px;font-size:14px}.chip{display:inline-block;background:#333;
border-radius:12px;padding:4px 10px;margin:2px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img src="/stream">
<h2>현재 상태</h2><div id=now></div>
<h2>동작별 횟수</h2><div id=cnt></div>
<h2>관심도 통계 (사람 단위)</h2><div id=beh></div>
<h2>동작 기록</h2><table><thead><tr><th>시각</th><th>대상</th><th>동작</th></tr></thead><tbody id=ev></tbody></table>
<script>
const chips=o=>Object.entries(o).map(([k,v])=>`<span class=chip>${k} ${v}</span>`).join('')||'-';
async function poll(){const r=await (await fetch('/events')).json();
 document.getElementById('mode').textContent=`[${r.mode}] ${r.stats.fps.toFixed(1)} fps, ${r.stats.people}명`;
 document.getElementById('now').innerHTML=Object.entries(r.now).map(([k,v])=>`<span class=chip>${k}: ${v}</span>`).join('')||'사람 없음';
 document.getElementById('cnt').innerHTML=chips(r.counts);
 document.getElementById('beh').innerHTML=chips(r.beh)+(r.interest_rate!==null?` <b>관심 비율 ${(r.interest_rate*100).toFixed(0)}%</b>`:'');
 document.getElementById('ev').innerHTML=r.events.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${e.who}</td><td>${e.label}</td></tr>`).join('');}
setInterval(poll,1000);poll();
</script>"""

@app.get("/mark")
def mark(name: str = ""):
    MARK["name"] = name
    if EV:
        EV.writerow([round(time.time(), 3), time.strftime("%H:%M:%S"), "-", "MARK", name])
    return {"mark": name}

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)   # applab: 8020/8030만 씀
