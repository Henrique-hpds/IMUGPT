"""Rigid rotation estimation utilities used by the IMU alignment module."""

from __future__ import annotations

from typing import Optional

import numpy as np


def estimate_rotation_procrustes(V: np.ndarray, W: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate a proper rotation matrix solving ``W ~= R @ V``.

    ``V`` and ``W`` are interpreted as row-major batches of vectors with shape
    ``[N, 3]``. The returned matrix is guaranteed to satisfy ``det(R) = +1``.
    """

    virtual = _validate_vector_batch(V, "V")
    real = _validate_vector_batch(W, "W")
    if virtual.shape != real.shape:
        raise ValueError("estimate_rotation_procrustes expects V and W with the same shape.")

    if weights is None:
        weighted_virtual = virtual
        weighted_real = real
    else:
        weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weight_array.shape[0] != virtual.shape[0]:
            raise ValueError("weights must have shape [N].")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative.")
        if float(np.sum(weight_array)) <= 0.0:
            raise ValueError("weights must contain at least one positive value.")
        sqrt_weights = np.sqrt(weight_array)[:, None]
        weighted_virtual = virtual * sqrt_weights
        weighted_real = real * sqrt_weights

    covariance = weighted_real.T @ weighted_virtual
    U, _, Vt = np.linalg.svd(covariance, full_matrices=True)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    rotation = U @ correction @ Vt
    if float(np.linalg.det(rotation)) < 0.0:
        correction[-1, -1] *= -1.0
        rotation = U @ correction @ Vt
    return rotation.astype(np.float32, copy=False)


def apply_rotation(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a rotation matrix to a batch of row-major vectors."""

    vectors = np.asarray(values, dtype=np.float32)
    matrix = np.asarray(rotation, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("values must have shape [N, 3].")
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3].")
    return (vectors @ matrix.T).astype(np.float32, copy=False)


def _validate_vector_batch(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3].")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two vectors.")
    finite_mask = np.isfinite(array).all(axis=1)
    array = array[finite_mask]
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two finite vectors.")
    return array
