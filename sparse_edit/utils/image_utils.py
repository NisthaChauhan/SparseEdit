"""Image loading, resizing, and normalisation utilities.

Memory: O(H * W * C) per image. ~3 MB for 512x512x3 float32.
Throughput: < 1 ms per image for resize + normalise on CPU.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def load_image(path: str, target_size: tuple[int, int] = (512, 512)) -> npt.NDArray[np.float32]:
    """Load an image from disk, resize, and return as float32 [0, 1].

    Parameters
    ----------
    path : str
        Path to image file (PNG, JPEG, BMP).
    target_size : tuple[int, int]
        (height, width) to resize to.

    Returns
    -------
    ndarray, shape (H, W, 3), dtype float32, range [0, 1]
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def save_image(image: npt.NDArray[np.floating], path: str) -> None:
    """Save a float [0, 1] image to disk.

    Parameters
    ----------
    image : ndarray, shape (H, W, 3), range [0, 1]
    path : str
    """
    from PIL import Image

    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def normalise_for_vae(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalise image from [0, 1] to [-1, 1] for VAE input.

    Parameters
    ----------
    image : ndarray, shape (H, W, 3), range [0, 1]

    Returns
    -------
    ndarray, shape (H, W, 3), range [-1, 1]
    """
    return image * 2.0 - 1.0


def denormalise_from_vae(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Denormalise image from [-1, 1] to [0, 1].

    Parameters
    ----------
    image : ndarray, shape (H, W, 3), range [-1, 1]

    Returns
    -------
    ndarray, shape (H, W, 3), range [0, 1]
    """
    return np.clip((image + 1.0) / 2.0, 0.0, 1.0)


def make_batch(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Add batch dimension: (H, W, C) -> (1, H, W, C).

    Parameters
    ----------
    image : ndarray, shape (H, W, C)

    Returns
    -------
    ndarray, shape (1, H, W, C)
    """
    return image[np.newaxis, ...]


def resize_mask(
    mask: npt.NDArray[np.floating],
    target_size: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Resize a binary mask using nearest-neighbour interpolation.

    Parameters
    ----------
    mask : ndarray, shape (H, W) or (H, W, 1)
    target_size : (target_H, target_W)

    Returns
    -------
    ndarray, shape (target_H, target_W, 1), float32
    """
    from PIL import Image

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    m_uint8 = (mask * 255).astype(np.uint8)
    m_pil = Image.fromarray(m_uint8, mode="L")
    m_resized = m_pil.resize(
        (target_size[1], target_size[0]), Image.NEAREST
    )
    arr = np.array(m_resized, dtype=np.float32) / 255.0
    return arr[:, :, np.newaxis]


# ── Backward-compat wrappers for sparse_edit.editing.pipeline ──
# pipeline.py calls preprocess_image(np_array, target_size=..., dtype=...)
# expecting an MLX array of shape [1, H, W, 3] in [-1, 1], and calls
# postprocess_image(mx_array) expecting an HWC uint8 numpy array.
import mlx.core as _mx_pp
from PIL import Image as _PIL_Image


def preprocess_image(
    image,
    target_size=(512, 512),
    dtype=None,
):
    """Resize -> normalise -> MLX array with batch dim.

    Parameters
    ----------
    image : np.ndarray (HWC uint8) or PIL.Image
    target_size : tuple[int, int]  (H, W)
    dtype : mlx dtype, optional (defaults to mx.float32)

    Returns
    -------
    mx.array of shape [1, H, W, 3] in [-1, 1]
    """
    # Accept PIL or numpy input.
    if hasattr(image, "convert"):  # PIL.Image
        image = np.array(image.convert("RGB"), dtype=np.uint8)

    if image.ndim == 4:
        image = image[0]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    # Resize via PIL (target_size is (H, W); PIL expects (W, H))
    H, W = int(target_size[0]), int(target_size[1])
    pil = _PIL_Image.fromarray(image).convert("RGB")
    pil = pil.resize((W, H), _PIL_Image.LANCZOS)
    arr = np.asarray(pil, dtype=np.float32)

    # Normalise to [-1, 1]
    arr = arr / 127.5 - 1.0

    # Add batch dim -> [1, H, W, 3]
    arr = arr[None, ...]

    # To MLX
    mlx_dtype = dtype if dtype is not None else _mx_pp.float32
    return _mx_pp.array(arr, dtype=mlx_dtype)


def postprocess_image(image):
    """Convert MLX/NumPy [-1, 1] tensor -> HWC uint8 NumPy array."""
    if hasattr(image, "shape") and not isinstance(image, np.ndarray):
        # Likely an mx.array
        arr = np.array(image)
    else:
        arr = np.asarray(image)

    if arr.ndim == 4:
        arr = arr[0]

    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return arr
