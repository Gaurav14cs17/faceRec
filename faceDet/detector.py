"""Face detection and landmark extraction (InsightFace RetinaFace + landmarks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from config import (
    CTX_ID,
    DET_SIZE,
    DET_THRESH,
    INSIGHTFACE_ROOT,
    MODEL_NAME,
    resolve_providers,
    WEIGHT_DET_DIR,
)
from weight_sync import ensure_weights, weight_dirs_ready


@dataclass(frozen=True)
class DetectedFace:
    """One face in an image."""

    bbox: np.ndarray  # (4,) x1, y1, x2, y2
    det_score: float
    kps: np.ndarray  # (5, 2) landmarks
    raw: Any  # underlying insightface Face object for advanced use


class FaceDetector:
    """High-accuracy detector using InsightFace ``buffalo_*`` detection stack."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        det_size: Tuple[int, int] = DET_SIZE,
        det_thresh: float = DET_THRESH,
        ctx_id: int = CTX_ID,
        providers: Optional[Sequence[str]] = None,
        app: Any = None,
    ) -> None:
        from insightface.app import FaceAnalysis

        self._det_size = det_size
        self._det_thresh = det_thresh
        provider_list = (
            list(providers) if providers is not None else resolve_providers()
        )

        if app is not None:
            self._app = app
        else:
            if not weight_dirs_ready():
                ensure_weights()
            if weight_dirs_ready():
                model_path = str(WEIGHT_DET_DIR.resolve())
            else:
                model_path = model_name
            kwargs = {
                "allowed_modules": ["detection", "landmark"],
                "providers": provider_list,
            }
            if weight_dirs_ready():
                self._app = FaceAnalysis(name=model_path, **kwargs)
            else:
                self._app = FaceAnalysis(
                    name=model_path,
                    root=str(INSIGHTFACE_ROOT),
                    **kwargs,
                )
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    @property
    def det_size(self) -> Tuple[int, int]:
        return self._det_size

    def detect(
        self,
        image: np.ndarray,
        max_faces: int = 0,
    ) -> List[DetectedFace]:
        """
        Run detection on a BGR uint8 image (OpenCV format).

        Args:
            image: HxWx3 BGR array.
            max_faces: If > 0, keep only the top faces by detection score.
        """
        if image is None or image.size == 0:
            return []

        faces = self._app.get(image)
        out: List[DetectedFace] = []
        for f in faces:
            score = float(f.det_score)
            if score < self._det_thresh:
                continue
            out.append(
                DetectedFace(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    det_score=score,
                    kps=np.asarray(f.kps, dtype=np.float32),
                    raw=f,
                )
            )

        out.sort(key=lambda x: x.det_score, reverse=True)
        if max_faces > 0:
            out = out[:max_faces]
        return out

    def detect_largest(self, image: np.ndarray) -> Optional[DetectedFace]:
        faces = self.detect(image, max_faces=1)
        return faces[0] if faces else None
