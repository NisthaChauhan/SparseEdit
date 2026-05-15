"""CLIP Score metric for SparseEdit.

    CLIPScore(I, C) = max(100 * cos_sim(E_I, E_C), 0)

Memory
------
    Precomputed mode: O(D) where D=768 — negligible.
    Full mode: ~400 MB for CLIP ViT-L/14 in fp16.

Throughput
----------
    Precomputed cosine similarity: < 0.01 ms.
    Full mode: ~10 ms on M4 Pro.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def cosine_similarity(
    a: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a, b : ndarray, shape (D,)

    Returns
    -------
    float in [-1, 1]
    """
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()

    dot = float(np.dot(a_f, b_f))
    norm_a = float(np.linalg.norm(a_f))
    norm_b = float(np.linalg.norm(b_f))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def compute_clip_score(
    image_embedding: npt.NDArray[np.floating],
    text_embedding: npt.NDArray[np.floating],
) -> float:
    """Compute CLIPScore from precomputed embeddings.

    Parameters
    ----------
    image_embedding : ndarray, shape (D,)
    text_embedding : ndarray, shape (D,)

    Returns
    -------
    float in [0, 100]
    """
    cos_sim = cosine_similarity(image_embedding, text_embedding)
    return max(100.0 * cos_sim, 0.0)


def compute_clip_score_batch(
    image_embeddings: npt.NDArray[np.floating],
    text_embeddings: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Compute CLIPScore for a batch of image-text pairs.

    Parameters
    ----------
    image_embeddings : ndarray, shape (B, D)
    text_embeddings : ndarray, shape (B, D)

    Returns
    -------
    ndarray, shape (B,) in [0, 100]

    Memory
    ------
        O(B * D). For B=256, D=768: ~3 MB.
    """
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            f"Shape mismatch: images {image_embeddings.shape} "
            f"vs text {text_embeddings.shape}"
        )
    if image_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2-D arrays (B, D), got {image_embeddings.ndim}-D"
        )

    a = image_embeddings.astype(np.float64)
    b = text_embeddings.astype(np.float64)

    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)

    cos_sims = np.sum(a_norm * b_norm, axis=1)
    scores = np.clip(100.0 * cos_sims, 0.0, None)
    return scores.astype(np.float64)
