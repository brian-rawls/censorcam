import json
import urllib.request

import cv2
import numpy as np
import pytest

from censorcam import probe
from censorcam.control import ControlServer
from censorcam.state import Mode, OperatorState


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
        return r.read().decode()


def _server(state, key=None):
    status = lambda: {"mode": state.mode.value, "status_text": state.mode.label}
    return ControlServer(state, status, lambda: {"ok": 1}, lambda ms: None,
                         port=0, api_key=key).start()


def test_http_mode_control():
    st = OperatorState(Mode.ALL)
    srv = _server(st)
    try:
        assert json.loads(_get(srv.port, "/mode/people"))["mode"] == "people"
        assert st.mode == Mode.PEOPLE
        _get(srv.port, "/toggle/all")
        assert st.mode == Mode.ALL
        _get(srv.port, "/toggle/all")          # back to the previous mode
        assert st.mode == Mode.PEOPLE
        assert _get(srv.port, "/status/text") == "BLUR PEOPLE"
        _get(srv.port, "/mode/normal")
        assert st.mode == Mode.NORMAL
        assert "<html>" in _get(srv.port, "/")
    finally:
        srv.stop()


def test_http_api_key():
    st = OperatorState(Mode.NORMAL)
    srv = _server(st, key="s3cret")
    try:
        with pytest.raises(urllib.error.HTTPError):
            _get(srv.port, "/mode/all")
        assert st.mode == Mode.NORMAL
        _get(srv.port, "/mode/all?key=s3cret")
        assert st.mode == Mode.ALL
    finally:
        srv.stop()


@pytest.mark.parametrize("scale,jpeg", [(1.0, False), (0.55, True), (0.33, True)])
def test_probe_roundtrip(scale, jpeg):
    f = np.full((1080, 1920, 3), 90, np.uint8)
    value = 0xA5C3F1
    probe.stamp(f, value)
    cell = 1080 // 60
    w, h = probe.GRID_W * cell, probe.GRID_H * cell
    x0, y0 = (1920 - w) // 2, (1080 - h) // 2
    crop = f[y0:y0 + h, x0:x0 + w]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if jpeg:
        crop = cv2.imdecode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], 1)
    assert probe.decode(crop) == value


def test_probe_age_wraps():
    assert probe.age_ms(probe.MOD - 10, now=5) == 15
