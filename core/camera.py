"""Camera conventions shared by data, model, rollout, and evaluation code.

External camera poses enter this package as camera-to-world transforms (``c2w``):
a homogeneous point expressed in camera coordinates is left-multiplied by ``c2w``
to obtain world coordinates. Code that requires world-to-camera transforms must
perform one explicit matrix inverse at the adapter boundary and name the result
``w2c``. Stored conditions remain ``c2w``.

Intrinsics use the standard pixel-coordinate matrix with focal lengths in
``K[0, 0]`` and ``K[1, 1]`` and principal point in ``K[0, 2]`` and ``K[1, 2]``.
Every condition declares whether those coordinates refer to the RGB image or a
latent grid. Resize and crop operations must produce a new matrix and append a
description to ``preprocessing``; callers must never reinterpret RGB intrinsics as
latent-grid intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple


class IntrinsicsSpace(str, Enum):
    """The pixel/grid coordinate system in which an intrinsic matrix is defined."""

    RGB_PIXELS = "rgb_pixels"
    LATENT_GRID = "latent_grid"


@dataclass(frozen=True)
class CameraCondition:
    """A batched camera trajectory with explicit geometry metadata.

    Tensor-like fields deliberately use ``Any`` so this contract does not depend on
    PyTorch, NumPy, Lightning, datasets, or evaluation packages.

    Shapes:
        c2w: ``[B, T, 4, 4]`` camera-to-world homogeneous transforms.
        intrinsics: ``[B, T, 3, 3]`` matrices in ``intrinsics_space``.
        timestamps_seconds: ``[B, T]`` physical timestamps.
    """

    c2w: Any
    intrinsics: Any
    timestamps_seconds: Any
    intrinsics_space: IntrinsicsSpace
    length_unit: str = "meter"
    reference_frame: str = "world"
    preprocessing: Tuple[str, ...] = ()

    @property
    def extrinsics_convention(self) -> str:
        """Return the fixed storage convention for consumers and provenance."""

        return "camera_to_world"
