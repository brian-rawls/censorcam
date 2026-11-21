"""Local control API (Stream Deck, browser dashboard) and optional OBS scene bridge.

Endpoints (GET, bound to 127.0.0.1 only):
  /mode/normal  /mode/people  /mode/all     set mode
  /toggle/all                                BLUR ALL on/off (returns to previous mode)
  /toggle/people                             BLUR PEOPLE <-> NORMAL
  /status        JSON {mode, failsafe, ...}
  /status/text   plain text: NORMAL | BLUR PEOPLE | BLUR ALL | FAILSAFE   (for key titles)
  /metrics       JSON performance metrics
  /probe/on  /probe/off                      latency probe strip (testing only!)
  /report/downstream?ms=123                  used by tools/measure_latency.py
  /                                          dashboard page
If an API key is configured, append ?key=YOUR_KEY to every request.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

from .state import Mode, OperatorState

DASHBOARD = """<!doctype html><html><head><meta charset="utf-8"><title>CensorCam</title>
<style>
body{font:15px system-ui,sans-serif;background:#111;color:#eee;margin:24px;max-width:860px}
.btns{display:flex;gap:12px;margin:12px 0 20px}
button{flex:1;font-size:20px;padding:22px 8px;border:2px solid #444;border-radius:10px;background:#222;color:#eee;cursor:pointer}
button.on{border-color:#fff}#b-normal.on{background:#1f6f3a}#b-people.on{background:#8a6d00}#b-all.on{background:#9b1c1c}
#banner{font-size:28px;font-weight:700;padding:14px;border-radius:10px;background:#333;text-align:center}
#banner.fs{background:#c00;animation:blink 1s infinite}@keyframes blink{50%{opacity:.6}}
pre{background:#1b1b1b;padding:12px;border-radius:8px;overflow:auto;font-size:13px}
</style></head><body>
<div id="banner">...</div>
<div class="btns">
<button id="b-normal" onclick="go('/mode/normal')">NORMAL</button>
<button id="b-people" onclick="go('/mode/people')">BLUR PEOPLE</button>
<button id="b-all" onclick="go('/mode/all')">BLUR ALL</button></div>
<label><input type="checkbox" id="probe" onchange="go(this.checked?'/probe/on':'/probe/off')"> latency probe strip (visible on stream - testing only)</label>
<pre id="m"></pre>
<script>
const K=new URLSearchParams(location.search).get('key');const q=K?('?key='+encodeURIComponent(K)):'';
function go(p){fetch(p+q).then(tick)}
function tick(){fetch('/metrics'+q).then(r=>r.json()).then(d=>{
 const b=document.getElementById('banner');b.textContent=d.status_text;b.className=d.failsafe?'fs':'';
 for(const m of ['normal','people','all'])document.getElementById('b-'+m).className=(d.mode==m?'on':'');
 document.getElementById('probe').checked=d.latency_probe;
 document.getElementById('m').textContent=JSON.stringify(d,null,2)})}
setInterval(tick,500);tick();
</script></body></html>"""


class ControlServer:
    def __init__(self, state: OperatorState, status_fn: Callable[[], Dict],
                 metrics_fn: Callable[[], Dict], report_fn: Callable[[float], None],
                 host: str = "127.0.0.1", port: int = 8765, api_key: Optional[str] = None):
        self.state = state
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, code: int, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else (
                    body.encode() if isinstance(body, str) else json.dumps(body).encode())
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                u = urlparse(self.path)
                qs = parse_qs(u.query)
                path = u.path.rstrip("/") or "/"
                if api_key and path != "/" and qs.get("key", [""])[0] != api_key:
                    return self._send(403, {"error": "bad or missing key"})
                try:
                    if path == "/":
                        return self._send(200, DASHBOARD, "text/html; charset=utf-8")
                    if path.startswith("/mode/"):
                        outer.state.set_mode(Mode.parse(path[6:]), "http")
                        return self._send(200, status_fn())
                    if path == "/toggle/all":
                        outer.state.toggle_all("http")
                        return self._send(200, status_fn())
                    if path == "/toggle/people":
                        outer.state.toggle_people("http")
                        return self._send(200, status_fn())
                    if path == "/status":
                        return self._send(200, status_fn())
                    if path == "/status/text":
                        return self._send(200, status_fn()["status_text"], "text/plain")
                    if path == "/metrics":
                        return self._send(200, metrics_fn())
                    if path in ("/probe/on", "/probe/off"):
                        outer.state.latency_probe = path.endswith("on")
                        return self._send(200, {"latency_probe": outer.state.latency_probe})
                    if path == "/report/downstream":
                        report_fn(float(qs["ms"][0]))
                        return self._send(200, {"ok": True})
                    return self._send(404, {"error": "unknown endpoint"})
                except ValueError as e:
                    return self._send(400, {"error": str(e)})

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="Control", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class ObsSceneBridge:
    """Follows OBS's program scene: when you switch to a scene whose name is in
    `scene_map`, CensorCam switches mode. Lets you use the native Stream Deck
    OBS 'Scene' buttons (which light up to show the active scene).

    Requires OBS 28+ (obs-websocket v5 built in) and `pip install obsws-python`.
    """

    def __init__(self, state: OperatorState, scene_map: Dict[str, Mode],
                 host: str = "localhost", port: int = 4455, password: str = ""):
        import obsws_python as obs

        self.state = state
        self.scene_map = {k.lower(): v for k, v in scene_map.items()}
        self.ev = obs.EventClient(host=host, port=port, password=password)
        self.ev.callback.register(self.on_current_program_scene_changed)
        # apply the scene that is live right now
        req = obs.ReqClient(host=host, port=port, password=password)
        self._apply(req.get_current_program_scene().current_program_scene_name)
        req.disconnect() if hasattr(req, "disconnect") else None

    def _apply(self, scene: str):
        m = self.scene_map.get(scene.lower())
        if m is not None:
            self.state.set_mode(m, f"obs scene '{scene}'")

    def on_current_program_scene_changed(self, data):
        self._apply(data.scene_name)

    def stop(self):
        try:
            self.ev.disconnect()
        except Exception:
            pass
