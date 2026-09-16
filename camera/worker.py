"""Multi-camera RTSP / USB ingest with FAISS search."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import yaml

from camera.tracker import SimpleTracker
from config import (
    CAMERA_EMBED_INTERVAL_SEC,
    CAMERA_FRAME_SKIP,
    CAMERA_MAX_STREAMS,
    CAMERAS_CONFIG,
    FAISS_INDEX_DIR,
    MATCH_THRESH,
    SEARCH_MARGIN,
)
from engine import FaceEngine
from faceDet.detector import DetectedFace
from faceRec.faiss_gallery import FaissGallery, IndexMatch

LOG = logging.getLogger(__name__)

HitCallback = Callable[[str, IndexMatch, DetectedFace, np.ndarray], None]


@dataclass
class CameraSpec:
    camera_id: str
    url: Union[str, int]


def load_cameras_config(path: Union[str, Path] = CAMERAS_CONFIG) -> List[CameraSpec]:
    path = Path(path)
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    items = data.get("cameras", data if isinstance(data, list) else [])
    specs: List[CameraSpec] = []
    for item in items:
        if isinstance(item, dict):
            cid = str(item.get("id", item.get("camera_id", "cam")))
            url = item.get("url", item.get("source", 0))
            if isinstance(url, str) and url.isdigit():
                url = int(url)
            specs.append(CameraSpec(camera_id=cid, url=url))
    return specs[:CAMERA_MAX_STREAMS]


class CameraStreamWorker:
    def __init__(
        self,
        spec: CameraSpec,
        engine: Optional[FaceEngine],
        gallery: FaissGallery,
        on_hit: Optional[HitCallback] = None,
        frame_skip: int = CAMERA_FRAME_SKIP,
        embed_interval: float = CAMERA_EMBED_INTERVAL_SEC,
        show_window: bool = False,
    ) -> None:
        self.spec = spec
        self._engine = engine or FaceEngine()
        self._gallery = gallery
        self._on_hit = on_hit
        self._frame_skip = max(1, frame_skip)
        self._embed_interval = embed_interval
        self._show = show_window
        self._tracker = SimpleTracker()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=self.spec.camera_id, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _open_capture(self) -> cv2.VideoCapture:
        url = self.spec.url
        if isinstance(url, int):
            return cv2.VideoCapture(url)
        return cv2.VideoCapture(str(url))

    def _run(self) -> None:
        cap = self._open_capture()
        if not cap.isOpened():
            LOG.error("[%s] Cannot open source %s", self.spec.camera_id, self.spec.url)
            return

        frame_idx = 0
        LOG.info("[%s] Stream started", self.spec.camera_id)
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            frame_idx += 1
            if frame_idx % self._frame_skip != 0:
                continue

            faces = self._engine.detector.detect(frame)
            bboxes = [f.bbox for f in faces]
            tracks = self._tracker.update(bboxes)
            now = time.time()

            for face in faces:
                cx = (face.bbox[0] + face.bbox[2]) / 2
                cy = (face.bbox[1] + face.bbox[3]) / 2
                track_id = None
                for tid, tb in tracks:
                    if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]:
                        track_id = tid
                        break
                if track_id is None:
                    continue
                tr = self._tracker.get_track(track_id)
                if tr is None:
                    continue
                if now - tr.last_embed_ts < self._embed_interval:
                    continue
                tr.last_embed_ts = now

                emb = self._engine.recognizer.embed(frame, face)
                match = self._gallery.search(emb)
                if match.matched and self._on_hit:
                    self._on_hit(self.spec.camera_id, match, face, frame)

                if self._show:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    color = (0, 200, 0) if match.matched else (0, 0, 255)
                    label = match.identity if match.matched else "unknown"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {match.score:.2f}",
                        (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

            if self._show:
                cv2.imshow(f"faceRec-{self.spec.camera_id}", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    self._stop.set()

        cap.release()
        if self._show:
            cv2.destroyWindow(f"faceRec-{self.spec.camera_id}")
        LOG.info("[%s] Stream stopped", self.spec.camera_id)


class MultiCameraRunner:
    def __init__(
        self,
        cameras: List[CameraSpec],
        index_dir: Path = FAISS_INDEX_DIR,
        match_thresh: float = MATCH_THRESH,
        margin: float = SEARCH_MARGIN,
        show_window: bool = False,
    ) -> None:
        self._gallery = FaissGallery.load(index_dir, match_thresh=match_thresh, margin=margin)
        self._workers = [
            CameraStreamWorker(
                spec=spec,
                engine=None,
                gallery=self._gallery,
                on_hit=self._log_hit,
                show_window=show_window,
            )
            for spec in cameras
        ]

    @staticmethod
    def _log_hit(
        camera_id: str,
        match: IndexMatch,
        face: DetectedFace,
        frame: np.ndarray,
    ) -> None:
        LOG.info(
            "HIT cam=%s id=%s score=%.3f bbox=%s",
            camera_id,
            match.identity,
            match.score,
            face.bbox.astype(int).tolist(),
        )

    def run_until_interrupt(self) -> None:
        for w in self._workers:
            w.start()
        print(f"Watching {len(self._workers)} camera(s). Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            for w in self._workers:
                w.stop()
