"""Load detection and recognition from separate weight folders."""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config import (
    CTX_ID,
    DET_SIZE,
    DET_THRESH,
    INSIGHTFACE_ROOT,
    MODEL_NAME,
    effective_ctx_id,
    resolve_providers,
    WEIGHT_DET_DIR,
    WEIGHT_REC_DIR,
)
from faceDet.detector import DetectedFace, FaceDetector
from faceRec.recognizer import FaceRecognizer
from weight_sync import load_recognition_model, sync_weights, weight_dirs_ready

logger = logging.getLogger(__name__)


def _prepare_app(app, ctx_id: int, det_size: Tuple[int, int]) -> None:
    try:
        app.prepare(ctx_id=ctx_id, det_size=det_size)
    except Exception:
        if ctx_id >= 0:
            logger.warning("GPU prepare failed; falling back to CPU (ctx_id=-1)")
            app.prepare(ctx_id=-1, det_size=det_size)
        else:
            raise


def _create_det_app(
    provider_list: Sequence[str],
    model_name: str,
) -> object:
    from insightface.app import FaceAnalysis

    if weight_dirs_ready():
        path = str(WEIGHT_DET_DIR.resolve())
        logger.info("Loading detection weights from %s", path)
        return FaceAnalysis(
            name=path,
            allowed_modules=["detection", "landmark"],
            providers=list(provider_list),
        )
    sync_weights()
    if weight_dirs_ready():
        path = str(WEIGHT_DET_DIR.resolve())
        return FaceAnalysis(
            name=path,
            allowed_modules=["detection", "landmark"],
            providers=list(provider_list),
        )
    return FaceAnalysis(
        name=model_name,
        root=str(INSIGHTFACE_ROOT),
        allowed_modules=["detection", "landmark"],
        providers=list(provider_list),
    )


class FaceEngine:
    """``weight/det`` for detector; ``weight/rec`` for ArcFace recognizer."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        det_size: Tuple[int, int] = DET_SIZE,
        det_thresh: float = DET_THRESH,
        ctx_id: int = CTX_ID,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self._det_thresh = det_thresh
        provider_list = (
            list(providers) if providers is not None else resolve_providers()
        )
        ctx_id = effective_ctx_id(ctx_id)

        t0 = time.perf_counter()
        det_app = _create_det_app(provider_list, model_name)
        _prepare_app(det_app, ctx_id, det_size)

        if weight_dirs_ready():
            logger.info("Loading recognition weights from %s", WEIGHT_REC_DIR)
        rec_model = load_recognition_model(provider_list, model_name)

        self._det_app = det_app
        self._rec_model = rec_model
        self.detector = FaceDetector(
            model_name=model_name,
            det_size=det_size,
            det_thresh=det_thresh,
            ctx_id=ctx_id,
            providers=provider_list,
            app=det_app,
        )
        self.recognizer = FaceRecognizer(
            model_name=model_name,
            ctx_id=ctx_id,
            providers=provider_list,
            rec_model=rec_model,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Models ready in %.1fs (weight/det + weight/rec)", elapsed)

    @property
    def det_app(self):
        return self._det_app

    @property
    def rec_model(self):
        return self._rec_model

    def analyze(
        self,
        image: np.ndarray,
        max_faces: int = 0,
    ) -> List[Tuple[DetectedFace, np.ndarray]]:
        """Detect faces and compute embeddings in one forward pass."""
        faces = self.detector.detect(image, max_faces=max_faces)
        out: List[Tuple[DetectedFace, np.ndarray]] = []
        for face in faces:
            emb = self.recognizer.embed(image, face)
            out.append((face, emb))
        return out
