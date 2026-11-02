"""과제 1 공통: 입력 열기(웹캠/RTSP/파일), 최신 1장 리더(자동 재연결), CPU 측정
--url 형식
  /dev/v4l/by-path/...   웹캠 (V4L2, MJPG)
  rtsp:stream1 / rtsp:stream2   IP 카메라 (계정은 ~/.config/eg2/ipcam.env 의 IPCAM_USER, IPCAM_PW — 파일·로그·화면에 비밀번호를 남기지 않음)
  rtsp://... 또는 동영상 파일 경로
"""
import os, threading, time, urllib.parse
import cv2

CRED_FILE = os.path.expanduser("~/.config/eg2/ipcam.env")
HOST_FILE = os.path.expanduser("~/.config/eg2/ipcam_host")   # 비밀 아님: "192.168.0.145:554" 한 줄
IPCAM_HOST = os.environ.get("IPCAM_HOST") or (open(HOST_FILE).read().strip() if os.path.exists(HOST_FILE) else "192.168.0.144:554")


def _ipcam_cred():
    user, pw = os.environ.get("IPCAM_USER"), os.environ.get("IPCAM_PW")
    if pw is None and os.path.exists(CRED_FILE):
        for line in open(CRED_FILE):
            k, _, v = line.strip().partition("=")
            if k == "IPCAM_USER":
                user = v
            elif k == "IPCAM_PW":
                pw = v
    if pw is None:
        raise SystemExit(f"IP 카메라 비밀번호가 없음: {CRED_FILE} 에 IPCAM_USER=, IPCAM_PW= 를 넣거나 환경변수로 주세요")
    return user or "admin", pw


def resolve(url):
    """실제 열 주소와, 로그에 남길 이름(비밀번호 없음)"""
    if url.startswith("rtsp:") and not url.startswith("rtsp://"):
        user, pw = _ipcam_cred()
        stream = url.split(":", 1)[1]
        q = lambda s: urllib.parse.quote(s, safe="")   # 비밀번호에 @ : / # 같은 문자가 있어도 주소가 깨지지 않게
        return f"rtsp://{q(user)}:{q(pw)}@{IPCAM_HOST}/{stream}", f"rtsp://{IPCAM_HOST}/{stream}"
    return url, url


def safe_name(url):
    """로그·화면용 이름 (비밀번호를 읽지도 넣지도 않음)"""
    if url.startswith("rtsp:") and not url.startswith("rtsp://"):
        return f"rtsp://{IPCAM_HOST}/{url.split(':', 1)[1]}"
    if url.startswith("rtsp://") and "@" in url:
        return "rtsp://***@" + url.split("@", 1)[1]
    return url


def is_device(url):
    return url.startswith("/dev/")


def open_cap(url, w=None, h=None, fps=None):
    real, _ = resolve(url)
    if is_device(url):
        cap = cv2.VideoCapture(real, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if w:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
    elif real.startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        # 기본 읽기 타임아웃 30초 → 3초 (랜선이 빠져도 TCP가 바로 안 끊겨 read()가 30초 멈추던 문제, T0-5)
        cap = cv2.VideoCapture(real, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    else:
        cap = cv2.VideoCapture(real)
    return cap


class LatestReader(threading.Thread):
    """별도 스레드가 계속 읽고 최신 1장만 남김. 읽기 실패가 이어지면 다시 연다(0.5→8초 백오프)."""

    def __init__(self, url, w=None, h=None, fps=None, log=print):
        super().__init__(daemon=True)
        self.url, self.args, self.log = url, (w, h, fps), log
        self.name_safe = safe_name(url)
        self.lock = threading.Lock()
        self.frame, self.ts, self.pos_ms, self.seq = None, 0.0, None, 0
        self.reconnects, self.read_count, self.stop_flag = 0, 0, False
        self.cap = None

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = open_cap(self.url, *self.args)
        return self.cap.isOpened()

    def run(self):
        backoff, fails, ever_ok = 0.5, 0, False
        while not self.stop_flag:
            if self.cap is None or not self.cap.isOpened():
                if self._open():
                    if ever_ok:
                        self.reconnects += 1
                        self.log(f"재연결 {self.reconnects}회 ({self.name_safe})")
                    backoff, fails = 0.5, 0
                else:
                    time.sleep(backoff); backoff = min(backoff * 2, 8.0)
                    continue
            ok, f = self.cap.read()
            if not ok or f is None:
                fails += 1
                if fails >= (10 if is_device(self.url) else 2) or (ever_ok and is_device(self.url) and not os.path.exists(self.url)):
                    self.log(f"입력 끊김 ({self.name_safe}), 다시 연결 시도")
                    self.cap.release(); self.cap = None
                    ever_ok = True
                else:
                    time.sleep(0.05)
                continue
            fails, ever_ok = 0, True
            pos = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            with self.lock:
                self.frame, self.ts, self.pos_ms = f, time.time(), pos
                self.seq += 1; self.read_count += 1
        if self.cap is not None:   # read() 도중 다른 스레드가 release하면 FFmpeg가 죽으므로 이 스레드에서 닫음
            self.cap.release()

    def latest(self):
        with self.lock:
            return self.frame, self.ts, self.pos_ms, self.seq

    def stop(self):
        self.stop_flag = True
        self.join(timeout=5)


def cpu_sampler():
    """호출할 때마다 (이 프로세스 CPU%, 시스템 전체 CPU%) — 직전 호출 이후 평균"""
    import psutil
    p = psutil.Process()
    p.cpu_percent(None); psutil.cpu_percent(None)
    return lambda: (p.cpu_percent(None), psutil.cpu_percent(None))
