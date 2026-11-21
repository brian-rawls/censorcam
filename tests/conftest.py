import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def frame():
    """Busy 1080p test frame with high-frequency detail everywhere."""
    rng = np.random.default_rng(0)
    f = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    return f


def frame_with_id(i: int):
    rng = np.random.default_rng(i + 1)
    return rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
