"""Lightweight IOU tracker to limit embed rate per face track."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    last_embed_ts: float = 0.0
    misses: int = 0


class SimpleTracker:
    def __init__(self, iou_thresh: float = 0.35, max_misses: int = 8) -> None:
        self._iou_thresh = iou_thresh
        self._max_misses = max_misses
        self._next_id = 1
        self._tracks: List[Track] = []

    def update(self, bboxes: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        """Return (track_id, bbox) for each detection."""
        assigned: List[Tuple[int, np.ndarray]] = []
        if not bboxes:
            for t in self._tracks:
                t.misses += 1
            self._tracks = [t for t in self._tracks if t.misses <= self._max_misses]
            return assigned

        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_dets = set(range(len(bboxes)))

        pairs: List[Tuple[float, int, int]] = []
        for ti in unmatched_tracks:
            for di in unmatched_dets:
                pairs.append((_iou(self._tracks[ti].bbox, bboxes[di]), ti, di))
        pairs.sort(reverse=True)

        det_to_track: dict[int, int] = {}
        for score, ti, di in pairs:
            if score < self._iou_thresh:
                break
            if ti not in unmatched_tracks or di not in unmatched_dets:
                continue
            unmatched_tracks.remove(ti)
            unmatched_dets.remove(di)
            det_to_track[di] = ti

        for di in unmatched_dets:
            bbox = bboxes[di]
            self._tracks.append(
                Track(track_id=self._next_id, bbox=bbox.copy(), misses=0)
            )
            assigned.append((self._next_id, bbox))
            self._next_id += 1

        for di, ti in det_to_track.items():
            tr = self._tracks[ti]
            tr.bbox = bboxes[di].copy()
            tr.misses = 0
            assigned.append((tr.track_id, tr.bbox))

        for ti in unmatched_tracks:
            self._tracks[ti].misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self._max_misses]
        return assigned

    def get_track(self, track_id: int) -> Track | None:
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None
