"""Synchronized image and camera preprocessing."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_crop_intrinsics(
    intrinsics: np.ndarray,
    *,
    input_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    crop_top_left: Tuple[int, int],
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Return RGB-pixel intrinsics after resize followed by a top-left crop.

    Sizes use ``(height, width)`` and crop coordinates use ``(top, left)``.
    """

    matrices = np.asarray(intrinsics, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("intrinsics must have shape [T, 3, 3]")
    input_h, input_w = input_size
    resized_h, resized_w = resized_size
    output_h, output_w = output_size
    top, left = crop_top_left
    if min(input_h, input_w, resized_h, resized_w, output_h, output_w) <= 0:
        raise ValueError("image sizes must be positive")
    if top < 0 or left < 0 or top + output_h > resized_h or left + output_w > resized_w:
        raise ValueError("crop lies outside the resized image")

    result = matrices.copy()
    scale_x = resized_w / input_w
    scale_y = resized_h / input_h
    result[:, 0, 0] *= scale_x
    result[:, 1, 1] *= scale_y
    result[:, 0, 2] = result[:, 0, 2] * scale_x - left
    result[:, 1, 2] = result[:, 1, 2] * scale_y - top
    return result


def resize_crop_frames(
    frames: np.ndarray,
    *,
    resized_size: Tuple[int, int],
    crop_top_left: Tuple[int, int],
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Resize and crop RGB frames stored as ``[T, H, W, C]``."""

    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("frames must have shape [T, H, W, 3]")
    resized_h, resized_w = resized_size
    output_h, output_w = output_size
    top, left = crop_top_left
    if top < 0 or left < 0 or top + output_h > resized_h or left + output_w > resized_w:
        raise ValueError("crop lies outside the resized image")
    resized = np.stack(
        [cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA) for frame in array]
    )
    return resized[:, top : top + output_h, left : left + output_w]
