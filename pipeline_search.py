"""Large-scale pipeline: faceDet + faceRec + FAISS index."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from config import FAISS_INDEX_DIR, MATCH_THRESH, SEARCH_MARGIN
from engine import FaceEngine
from faceDet import DetectedFace
from faceRec.bulk_enroll import enroll_from_folders, enroll_from_manifest, migrate_numpy_gallery
from faceRec.faiss_gallery import FaissGallery, IndexMatch
from pipeline import FacePipeline, PipelineResult


class SearchPipeline(FacePipeline):
    """Same as FacePipeline but matches against a FAISS index (100k+)."""

    def __init__(
        self,
        index_dir: Optional[Union[str, Path]] = None,
        match_thresh: float = MATCH_THRESH,
        margin: float = SEARCH_MARGIN,
        engine: Optional[FaceEngine] = None,
        load_existing_index: bool = True,
        gallery_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._index_dir = Path(index_dir) if index_dir else FAISS_INDEX_DIR
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._margin = margin
        index_path = self._index_dir / "faces.index"
        if load_existing_index and index_path.is_file():
            self._faiss = FaissGallery.load(
                self._index_dir, match_thresh=match_thresh, margin=margin
            )
        else:
            self._faiss = FaissGallery(match_thresh=match_thresh, margin=margin)

        super().__init__(
            gallery_dir=gallery_dir,
            match_thresh=match_thresh,
            engine=engine,
            load_small_gallery=False,
        )

    @property
    def faiss(self) -> FaissGallery:
        assert self._faiss is not None
        return self._faiss

    def enroll_index(self, identity: str, image: np.ndarray) -> np.ndarray:
        face = self.detector.detect_largest(image)
        if face is None:
            raise RuntimeError("No face detected for enrollment")
        emb = self.recognizer.embed(image, face)
        self.faiss.add(identity, emb)
        return emb

    def enroll_index_file(self, identity: str, image_path: Union[str, Path]) -> np.ndarray:
        return self.enroll_index(identity, self.read_image(image_path))

    def save_index(self) -> None:
        self.faiss.save(self._index_dir)

    def process_image(
        self,
        image: np.ndarray,
        max_faces: int = 0,
    ) -> List[PipelineResult]:
        from faceRec import MatchResult

        results: List[PipelineResult] = []
        for face, emb in self._engine.analyze(image, max_faces=max_faces):
            hit = self.faiss.search(emb)
            match = MatchResult(
                identity=hit.identity,
                score=hit.score,
                matched=hit.matched,
            )
            results.append(PipelineResult(face=face, embedding=emb, match=match))
        return results

    def bulk_enroll_folders(self, root: Union[str, Path]) -> tuple[int, int]:
        def embed_file(img: np.ndarray) -> np.ndarray:
            face = self.detector.detect_largest(img)
            if face is None:
                raise RuntimeError("no face")
            return self.recognizer.embed(img, face)

        return enroll_from_folders(
            self.faiss,
            root,
            embed_fn=embed_file,
            load_image=FacePipeline.read_image,
        )

    def bulk_enroll_manifest(self, manifest: Union[str, Path]) -> tuple[int, int]:
        def embed_file(img: np.ndarray) -> np.ndarray:
            face = self.detector.detect_largest(img)
            if face is None:
                raise RuntimeError("no face")
            return self.recognizer.embed(img, face)

        return enroll_from_manifest(
            self.faiss,
            manifest,
            embed_fn=embed_file,
            load_image=FacePipeline.read_image,
        )
