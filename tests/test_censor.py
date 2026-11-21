# CensorCam - Copyright (C) 2026 Aeternum Logic. Developed by Brian Rawls.
# Licensed under the GNU Affero General Public License v3.0 - see LICENSE.
import numpy as np

from censorcam.censor import Detection, MaskBuilder, censor_people, obscure


def _block_detail(img, block):
    """Mean per-block std-dev: high for real detail, ~0 when information is destroyed."""
    h, w = img.shape[:2]
    g = img[: h // block * block, : w // block * block].astype(np.float32).mean(axis=2)
    g = g.reshape(h // block, block, w // block, block)
    return g.std(axis=(1, 3)).mean()


def test_obscure_destroys_detail(frame):
    for style in ("blur", "pixelate"):
        out = obscure(frame, 48, style)
        assert out.shape == frame.shape
        assert _block_detail(frame, 48) > 30
        assert _block_detail(out, 48) < 6, style


def test_people_censors_only_inside_mask(frame):
    det = Detection(box=[800, 300, 1000, 900], conf=0.9,
                    polygons=[np.array([[850, 320], [950, 320], [980, 880], [820, 880]], np.float32)])
    b = MaskBuilder("mask", dilate_px=16, hold=0)
    out, mask = censor_people(frame, [det], b, block=24)
    inside = mask > 0
    assert inside[600, 900] and not inside[100, 100]
    # every masked pixel replaced, every unmasked pixel untouched
    assert np.array_equal(out[~inside], frame[~inside])
    changed = np.any(out != frame, axis=2)
    assert changed[inside].mean() > 0.95
    # polygon fully covered (dilation means mask is a superset)
    assert inside[320:880, 850:950].all()


def test_box_fallback_when_no_polygon(frame):
    det = Detection(box=[100, 100, 300, 500], conf=0.5, polygons=[])
    b = MaskBuilder("mask", dilate_px=0, hold=0)
    mask = b.build([det], frame.shape)
    assert mask[100:500, 100:300].all()


def test_hold_only_adds_coverage(frame):
    b = MaskBuilder("box", dilate_px=0, hold=1)
    m1 = b.build([Detection([100, 100, 200, 200], 0.9)], frame.shape)
    m2 = b.build([], frame.shape)          # detection dropped out for one frame
    assert (m2[m1 > 0] > 0).all()           # previous coverage retained
    m3 = b.build([], frame.shape)
    assert not m3.any()                     # ...but only for `hold` frames
