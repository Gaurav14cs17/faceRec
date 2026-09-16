"""End-to-end face detection + recognition. Prefer: python run.py <command> (see README.md)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from config import GALLERY_DIR, MATCH_THRESH
from engine import FaceEngine
from faceDet import DetectedFace
from faceRec import MatchResult


@dataclass
class PipelineResult:
    face: DetectedFace
    embedding: np.ndarray
    match: MatchResult


class FacePipeline:
    """Orchestrates detection, embedding, and gallery matching."""

    def __init__(
        self,
        gallery_dir: Optional[Union[str, Path]] = None,
        match_thresh: float = MATCH_THRESH,
        engine: Optional[FaceEngine] = None,
        load_small_gallery: bool = True,
    ) -> None:
        self._engine = engine or FaceEngine()
        self.detector = self._engine.detector
        self.recognizer = self._engine.recognizer
        self.recognizer._match_thresh = match_thresh
        self._gallery_dir = Path(gallery_dir) if gallery_dir else GALLERY_DIR
        if (
            load_small_gallery
            and self._gallery_dir.is_dir()
            and (self._gallery_dir / "gallery.json").is_file()
        ):
            self.recognizer.load_gallery(self._gallery_dir)

    @staticmethod
    def read_image(path: Union[str, Path]) -> np.ndarray:
        path = Path(path)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img

    def enroll_image(self, identity: str, image: np.ndarray) -> np.ndarray:
        face = self.detector.detect_largest(image)
        if face is None:
            raise RuntimeError("No face detected for enrollment")
        emb = self.recognizer.embed(image, face)
        self.recognizer.enroll(identity, emb)
        return emb

    def enroll_file(self, identity: str, image_path: Union[str, Path]) -> np.ndarray:
        return self.enroll_image(identity, self.read_image(image_path))

    def process_image(
        self,
        image: np.ndarray,
        max_faces: int = 0,
    ) -> List[PipelineResult]:
        results: List[PipelineResult] = []
        for face, emb in self._engine.analyze(image, max_faces=max_faces):
            match = self.recognizer.match(emb)
            results.append(
                PipelineResult(face=face, embedding=emb, match=match)
            )
        return results

    def process_file(
        self,
        image_path: Union[str, Path],
        max_faces: int = 0,
    ) -> List[PipelineResult]:
        return self.process_image(self.read_image(image_path), max_faces=max_faces)

    def save_gallery(self) -> None:
        self.recognizer.save_gallery(self._gallery_dir)

    def draw_results(
        self,
        image: np.ndarray,
        results: List[PipelineResult],
    ) -> np.ndarray:
        out = image.copy()
        for r in results:
            x1, y1, x2, y2 = r.face.bbox.astype(int)
            color = (0, 200, 0) if r.match.matched else (0, 0, 255)
            label = (
                f"{r.match.identity} ({r.match.score:.2f})"
                if r.match.matched
                else f"unknown ({r.match.score:.2f})"
            )
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return out


def _main() -> int:
    from cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(_main())
