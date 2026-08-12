"""Global camera-motion compensation.

Why this exists
---------------
The tracker matches boxes between frames by overlap. That assumes the
world moves and the camera does not. On handheld footage the camera moves
too, so a person standing perfectly still has a box that jumps between
frames, the overlap collapses, the track dies, and a new identity is
minted from nothing.

Measuring the sample clip showed this is the *dominant* cause of inflated
person counts — larger than scene cuts, larger than the tracker's matching
rules. See docs/FINDINGS.md #3.

The fix is to estimate how the whole frame moved, then move the tracks by
the same amount before matching, so comparison happens in a common frame
of reference. Trackers like BoT-SORT call this CMC (camera motion
compensation); it is one of the cheapest large wins available.

Method: sparse optical flow. Corner features are found on the previous
frame, followed into the current frame with Lucas-Kanade, and a partial
affine transform (translation, rotation, uniform scale) is fitted to the
matched pairs with RANSAC.

One important detail: feature points that land *on people* are masked out.
People move independently of the camera, so including them biases the
estimate toward the crowd's motion rather than the camera's — which is
precisely backwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection

# Lucas-Kanade and feature-detection settings. Deliberately few points:
# this runs on every frame and precision matters less than stability.
_MAX_CORNERS = 200
_QUALITY = 0.01
_MIN_DISTANCE = 30
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


@dataclass(frozen=True)
class Transform:
    """A 2x3 affine transform mapping previous-frame points to this frame."""

    matrix: np.ndarray | None

    @property
    def is_identity(self) -> bool:
        return self.matrix is None

    @property
    def translation(self) -> tuple[float, float]:
        if self.matrix is None:
            return (0.0, 0.0)
        return (float(self.matrix[0, 2]), float(self.matrix[1, 2]))

    @property
    def magnitude(self) -> float:
        """Pixels of translation — a rough 'how much did the camera move'."""
        dx, dy = self.translation
        return float(np.hypot(dx, dy))

    def apply_to_box(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Move a box into the current frame's coordinates."""
        if self.matrix is None:
            return box

        x, y, w, h = box
        corners = np.array([[x, y], [x + w, y + h]], dtype=np.float32).reshape(-1, 1, 2)
        moved = cv2.transform(corners, self.matrix).reshape(-1, 2)

        x1, y1 = moved[0]
        x2, y2 = moved[1]
        return (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))


class CameraMotionEstimator:
    """Estimates frame-to-frame global motion, ignoring detected people."""

    def __init__(self, enabled: bool = True, downscale: float = 0.5) -> None:
        self.enabled = enabled
        self.downscale = downscale
        self._prev_gray: np.ndarray | None = None
        self.last: Transform = Transform(None)

    def _mask_people(self, shape: tuple[int, int], detections: list[Detection]) -> np.ndarray:
        """255 where features may be sampled, 0 over people."""
        mask = np.full(shape, 255, dtype=np.uint8)
        scale = self.downscale
        for det in detections:
            x, y, w, h = det.box
            # Pad slightly: box edges usually clip a little background.
            x0 = max(0, int((x - w * 0.1) * scale))
            y0 = max(0, int((y - h * 0.1) * scale))
            x1 = min(shape[1], int((x + w * 1.1) * scale))
            y1 = min(shape[0], int((y + h * 1.1) * scale))
            mask[y0:y1, x0:x1] = 0
        return mask

    def update(self, frame: np.ndarray, detections: list[Detection] | None = None) -> Transform:
        """Estimate motion from the previous frame to this one."""
        if not self.enabled:
            self.last = Transform(None)
            return self.last

        small = cv2.resize(frame, None, fx=self.downscale, fy=self.downscale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            self.last = Transform(None)
            return self.last

        mask = self._mask_people(gray.shape, detections or [])
        prev_points = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=_MAX_CORNERS,
            qualityLevel=_QUALITY,
            minDistance=_MIN_DISTANCE,
            mask=mask,
        )

        matrix = None
        if prev_points is not None and len(prev_points) >= 6:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, prev_points, None, **_LK_PARAMS
            )
            if next_points is not None and status is not None:
                keep = status.flatten() == 1
                src = prev_points[keep].reshape(-1, 2)
                dst = next_points[keep].reshape(-1, 2)
                if len(src) >= 6:
                    estimated, _ = cv2.estimateAffinePartial2D(
                        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    if estimated is not None:
                        # Undo the downscale: translation scales with it,
                        # rotation and scale do not.
                        matrix = estimated.copy()
                        matrix[0, 2] /= self.downscale
                        matrix[1, 2] /= self.downscale

        self._prev_gray = gray
        self.last = Transform(matrix)
        return self.last
