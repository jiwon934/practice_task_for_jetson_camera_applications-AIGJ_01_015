"""B1: GStreamer 파이프라인 하나를 돌리며 프레임 수·FPS·이 프로세스 CPU·시스템 CPU를 잼.
RTSP 계정은 ~/.config/eg2/ipcam.env에서 읽어 rtspsrc 속성으로만 넣음 (명령줄·로그에 안 남김).
사용: python3 gst_run.py NAME 'PIPELINE (rtspsrc는 name=src, 마지막 sink는 name=sink)' [초]"""
import json, os, sys, time
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)
name, desc = sys.argv[1], sys.argv[2]
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0


def cred():
    d = {}
    for line in open(os.path.expanduser("~/.config/eg2/ipcam.env")):
        k, _, v = line.strip().partition("=")
        d[k] = v
    return d.get("IPCAM_USER", "admin"), d["IPCAM_PW"]


def sys_cpu():
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    return sum(v), v[3] + v[4]


pipe = Gst.parse_launch(desc)
src = pipe.get_by_name("src")
if src is not None and src.find_property("user-pw"):
    u, p = cred(); src.set_property("user-id", u); src.set_property("user-pw", p)
n, t_first = [0], [None]


def probe(pad, info):
    if t_first[0] is None:
        t_first[0] = time.time()
    n[0] += 1
    return Gst.PadProbeReturn.OK


pipe.get_by_name("sink").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe)
loop, err = GLib.MainLoop(), [None]


def on_msg(bus, m):
    if m.type == Gst.MessageType.EOS:
        loop.quit()
    elif m.type == Gst.MessageType.ERROR:
        err[0] = m.parse_error()[0].message; loop.quit()


bus = pipe.get_bus(); bus.add_signal_watch(); bus.connect("message", on_msg)
GLib.timeout_add(int(dur * 1000), lambda: loop.quit() or False)
t0, c0, s0 = time.time(), os.times(), sys_cpu()
pipe.set_state(Gst.State.PLAYING); loop.run()
t1, c1, s1 = time.time(), os.times(), sys_cpu()
pipe.set_state(Gst.State.NULL)
wall = t1 - t0
proc = (c1.user + c1.system - c0.user - c0.system) / wall * 100
tot, idle = s1[0] - s0[0], s1[1] - s0[1]
fps = n[0] / (t1 - t_first[0]) if t_first[0] and t1 > t_first[0] else 0
print(json.dumps({"case": name, "frames": n[0], "sec": round(wall, 1), "fps": round(fps, 1),
                  "proc_cpu_1core": round(proc, 1), "proc_cpu_6core": round(proc / 6, 1),
                  "sys_cpu": round(100 * (tot - idle) / tot, 1), "error": err[0]}, ensure_ascii=False), flush=True)
