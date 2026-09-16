"""ArcFace embeddings and gallery matching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from config import (
    CTX_ID,
    INSIGHTFACE_ROOT,
    MATCH_THRESH,
    MODEL_NAME,
    resolve_providers,
    WEIGHT_REC_DIR,
)
from weight_sync import ensure_weights, load_recognition_model, weight_dirs_ready
from faceDet.detector import DetectedFace


@dataclass(frozen=True)
class MatchResult:
    identity: Optional[str]
    score: float
    matched: bool


class FaceRecognizer:
    """Recognition via InsightFace ArcFace (``buffalo_*`` recognition module)."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        match_thresh: float = MATCH_THRESH,
        ctx_id: int = CTX_ID,
        providers: Optional[Sequence[str]] = None,
        app: Any = None,
        rec_model: Any = None,
    ) -> None:
        self._match_thresh = match_thresh
        provider_list = (
            list(providers) if providers is not None else resolve_providers()
        )
        self._app: Any = None
        self._rec_model: Any = None

        if rec_model is not None:
            self._rec_model = rec_model
        elif app is not None:
            self._app = app
        else:
            if not weight_dirs_ready():
                ensure_weights()
            self._rec_model = load_recognition_model(provider_list, model_name)

        self._gallery: Dict[str, np.ndarray] = {}

    @property
    def match_threshold(self) -> float:
        return self._match_thresh

    @property
    def gallery(self) -> Dict[str, np.ndarray]:
        return dict(self._gallery)

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Compute L2-normalized embedding for a detected face."""
        raw = face.raw
        pre = getattr(raw, "normed_embedding", None)
        if pre is not None and len(pre) > 0:
            emb = np.asarray(pre, dtype=np.float32).reshape(-1)
            return emb

        if self._rec_model is not None:
            rec_model = self._rec_model
        else:
            rec_model = self._app.models["recognition"]
        emb = rec_model.get(image, raw)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb

    def enroll(self, identity: str, embedding: np.ndarray) -> None:
        identity = identity.strip()
        if not identity:
            raise ValueError("identity must be non-empty")
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        self._gallery[identity] = emb

    def remove(self, identity: str) -> bool:
        return self._gallery.pop(identity, None) is not None

    def clear_gallery(self) -> None:
        self._gallery.clear()

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        return float(np.dot(a, b))

    def match(self, embedding: np.ndarray) -> MatchResult:
        if not self._gallery:
            return MatchResult(identity=None, score=0.0, matched=False)

        best_id: Optional[str] = None
        best_score = -1.0
        for name, ref in self._gallery.items():
            s = self.cosine_similarity(embedding, ref)
            if s > best_score:
                best_score = s
                best_id = name

        matched = best_score >= self._match_thresh
        return MatchResult(
            identity=best_id if matched else None,
            score=best_score,
            matched=matched,
        )

    def match_all(
        self, embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        scores = [
            (name, self.cosine_similarity(embedding, ref))
            for name, ref in self._gallery.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save_gallery(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"identities": list(self._gallery.keys())}
        (directory / "gallery.json").write_text(json.dumps(meta, indent=2))
        for name, emb in self._gallery.items():
            safe = name.replace("/", "_")
            np.save(directory / f"{safe}.npy", emb)

    def load_gallery(self, directory: Union[str, Path]) -> int:
        directory = Path(directory)
        meta_path = directory / "gallery.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No gallery at {directory}")

        meta = json.loads(meta_path.read_text())
        self._gallery.clear()
        for name in meta.get("identities", []):
            safe = name.replace("/", "_")
            path = directory / f"{safe}.npy"
            if path.is_file():
                self._gallery[name] = np.load(path).astype(np.float32)
        return len(self._gallery)
