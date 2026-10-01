import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))

from fixed_memory_wisig_preprocessing_api import (  # noqa: E402
    make_wisig_segment,
    normalize_wisig_packets,
)


def test_wisig_1024_frontend_shape_and_finiteness():
    rng = np.random.default_rng(7)
    packets = rng.normal(size=(4, 256, 2)).astype(np.float32)
    features = make_wisig_segment(normalize_wisig_packets(packets), 0, 1024)
    assert features.shape == (3, 1024)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_packet_rms_normalization():
    packets = np.ones((4, 256, 2), dtype=np.float32)
    normalized = normalize_wisig_packets(packets)
    rms = np.sqrt(np.mean(np.sum(normalized.astype(np.float64) ** 2, axis=-1), axis=-1))
    np.testing.assert_allclose(rms, np.ones(4), rtol=1e-6)

