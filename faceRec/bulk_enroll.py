"""Batch enrollment into the FAISS index."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from faceRec.faiss_gallery import FaissGallery

LOG = logging.getLogger(__name__)

ImageLoader = Callable[[Path], np.ndarray]
EmbedFn = Callable[[np.ndarray], np.ndarray]


def enroll_from_manifest(
    gallery: FaissGallery,
    manifest: Union[str, Path],
    embed_fn: EmbedFn,
    load_image: ImageLoader,
) -> Tuple[int, int]:
    """
    CSV manifest columns: identity, image_path
    Returns (ok_count, fail_count).
    """
    manifest = Path(manifest)
    ok, fail = 0, 0
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            identity = (row.get("identity") or row.get("name") or "").strip()
            rel = (row.get("image_path") or row.get("path") or "").strip()
            if not identity or not rel:
                fail += 1
                continue
            path = Path(rel)
            if not path.is_file():
                path = manifest.parent / rel
            try:
                emb = embed_fn(load_image(path))
                gallery.add(identity, emb)
                ok += 1
            except Exception as exc:
                LOG.warning("Skip %s: %s", path, exc)
                fail += 1
    return ok, fail


def enroll_from_folders(
    gallery: FaissGallery,
    root: Union[str, Path],
    embed_fn: EmbedFn,
    load_image: ImageLoader,
    extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Tuple[int, int]:
    """
    Layout: root/<identity>/*.jpg — one embedding per image (multiple per person allowed).
    """
    root = Path(root)
    exts = {e.lower() for e in extensions}
    ok, fail = 0, 0
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        identity = person_dir.name
        for img in sorted(person_dir.iterdir()):
            if img.suffix.lower() not in exts:
                continue
            try:
                emb = embed_fn(load_image(img))
                gallery.add(identity, emb)
                ok += 1
            except Exception as exc:
                LOG.warning("Skip %s: %s", img, exc)
                fail += 1
    return ok, fail


def migrate_numpy_gallery(
    gallery: FaissGallery,
    gallery_dir: Union[str, Path],
) -> int:
    """Import legacy data/gallery/*.npy into FAISS."""
    import json

    gallery_dir = Path(gallery_dir)
    meta_path = gallery_dir / "gallery.json"
    if not meta_path.is_file():
        return 0
    meta = json.loads(meta_path.read_text())
    count = 0
    for name in meta.get("identities", []):
        safe = name.replace("/", "_")
        path = gallery_dir / f"{safe}.npy"
        if path.is_file():
            gallery.add(name, np.load(path))
            count += 1
    return count
