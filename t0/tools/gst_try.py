# RTSP 계정 확인용 1회 시도: 비밀번호를 주소에 넣지 않고 rtspsrc 속성으로 전달 (~/.config/eg2/ipcam.env)
import os, gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
kv = dict(l.strip().split("=", 1) for l in open(os.path.expanduser("~/.config/eg2/ipcam.env")) if "=" in l)
host = open(os.path.expanduser("~/.config/eg2/ipcam_host")).read().strip()
p = Gst.parse_launch(f"rtspsrc name=src location=rtsp://{host}/stream2 protocols=tcp latency=200 ! fakesink num-buffers=20")
src = p.get_by_name("src"); src.set_property("user-id", kv["IPCAM_USER"]); src.set_property("user-pw", kv["IPCAM_PW"])
p.set_state(Gst.State.PLAYING)
msg = p.get_bus().timed_pop_filtered(15 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
print("RESULT:", "timeout" if msg is None else ("OK (buffers received)" if msg.type == Gst.MessageType.EOS else "ERROR " + msg.parse_error()[0].message))
p.set_state(Gst.State.NULL)
