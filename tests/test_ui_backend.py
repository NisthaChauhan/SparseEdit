"""Smoke tests for the UI adapter layer."""

import numpy as np
import pytest
from PIL import Image

import mlx.core as mx

from ui_backend import EditRequest, pil_to_mx, mx_to_pil


def _dummy_image(size=128):
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_pil_to_mx_shape_and_range():
    img = _dummy_image(64)
    out = pil_to_mx(img, 64, 64)
    assert isinstance(out, mx.array)
    assert out.shape == (1, 64, 64, 3)
    assert out.dtype == mx.float32
    arr = np.array(out)
    assert arr.min() >= -1.0 - 1e-5
    assert arr.max() <=  1.0 + 1e-5


def test_mx_to_pil_roundtrip():
    img = _dummy_image(64)
    mx_arr = pil_to_mx(img, 64, 64)
    out = mx_to_pil(mx_arr)
    assert isinstance(out, Image.Image)
    assert out.size == (64, 64)


def test_edit_request_defaults():
    req = EditRequest(source_image=_dummy_image(), prompt="test")
    assert req.num_steps == 30
    assert req.tau == 0.02
    assert req.lambda_sparsity == 0.05
    assert req.height == req.width == 1024


@pytest.mark.slow
def test_edit_image_runs_end_to_end():
    """Only run when weights are present; skip on CI without them."""
    pytest.importorskip("sparse_edit.pipeline")
    from ui_backend import edit_image
    req = EditRequest(
        source_image=_dummy_image(256),
        prompt="a cute cat",
        num_steps=4,        # tiny for smoke
        height=256, width=256,
    )
    result = edit_image(req)
    assert result.edited_image.size == (256, 256)
