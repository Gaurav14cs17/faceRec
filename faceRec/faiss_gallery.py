"""FAISS-backed gallery for ~100k+ identities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from config import (
    EMBED_DIM,
    FAISS_IDS_FILE,
    FAISS_INDEX_DIR,
    FAISS_INDEX_FILE,
    MATCH_THRESH,
    SEARCH_MARGIN,
)


@dataclass(frozen=True)
class IndexMatch:
    identity: Optional[str]
    score: float
    matched: bool
    rank: List[Tuple[str, float]]


class FaissGallery:
    """Inner-product index on L2-normalized embeddings (= cosine similarity)."""

    def __init__(
        self,
        match_thresh: float = MATCH_THRESH,
        margin: float = SEARCH_MARGIN,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        import faiss

        self._match_thresh = match_thresh
        self._margin = margin
        self._dim = embed_dim
        self._ids: List[str] = []
        self._index = faiss.IndexFlatIP(embed_dim)

    @property
    def size(self) -> int:
        return len(self._ids)

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        if v.shape[0] != self._dim:
            raise ValueError(f"Expected dim {self._dim}, got {v.shape[0]}")
        n = np.linalg.norm(v)
        if n > 1e-6:
            v = v / n
        return v

    def add(self, identity: str, embedding: np.ndarray) -> int:
        identity = identity.strip()
        if not identity:
            raise ValueError("identity must be non-empty")
        vec = self._normalize(embedding).reshape(1, -1)
        self._index.add(vec)
        self._ids.append(identity)
        return self._index.ntotal - 1

    def add_batch(self, identities: Sequence[str], embeddings: np.ndarray) -> int:
        if len(identities) != embeddings.shape[0]:
            raise ValueError("identities length must match embeddings rows")
        rows = []
        for i, identity in enumerate(identities):
            rows.append(self._normalize(embeddings[i]))
        mat = np.vstack(rows).astype(np.float32)
        self._index.add(mat)
        self._ids.extend([str(x).strip() for x in identities])
        return mat.shape[0]

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> IndexMatch:
        if self._index.ntotal == 0:
            return IndexMatch(identity=None, score=0.0, matched=False, rank=[])

        k = min(top_k, self._index.ntotal)
        vec = self._normalize(embedding).reshape(1, -1)
        scores, indices = self._index.search(vec, k)
        rank: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rank.append((self._ids[int(idx)], float(score)))

        if not rank:
            return IndexMatch(identity=None, score=0.0, matched=False, rank=[])

        best_id, best_score = rank[0]
        second = rank[1][1] if len(rank) > 1 else -1.0
        margin_ok = (best_score - second) >= self._margin
        matched = best_score >= self._match_thresh and margin_ok
        return IndexMatch(
            identity=best_id if matched else None,
            score=best_score,
            matched=matched,
            rank=rank,
        )

    def save(self, directory: Union[str, Path]) -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / FAISS_INDEX_FILE))
        (directory / FAISS_IDS_FILE).write_text(
            json.dumps({"ids": self._ids, "dim": self._dim}, indent=2)
        )

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        match_thresh: float = MATCH_THRESH,
        margin: float = SEARCH_MARGIN,
    ) -> "FaissGallery":
        import faiss

        directory = Path(directory)
        index_path = directory / FAISS_INDEX_FILE
        ids_path = directory / FAISS_IDS_FILE
        if not index_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(f"No FAISS index in {directory}")

        meta = json.loads(ids_path.read_text())
        gal = cls(
            match_thresh=match_thresh,
            margin=margin,
            embed_dim=int(meta.get("dim", EMBED_DIM)),
        )
        gal._index = faiss.read_index(str(index_path))
        gal._ids = list(meta.get("ids", []))
        if gal._index.ntotal != len(gal._ids):
            raise RuntimeError("Index vector count does not match ids.json length")
        return gal

    def build_ivf(self, nlist: int = 4096) -> None:
        """Optional: convert flat index to IVF for faster search at very large N."""
        import faiss

        n = self._index.ntotal
        if n < max(nlist * 10, 1000):
            return
        vectors = np.zeros((n, self._dim), dtype=np.float32)
        for i in range(n):
            self._index.reconstruct(int(i), vectors[i])
        quantizer = faiss.IndexFlatIP(self._dim)
        ivf = faiss.IndexIVFFlat(quantizer, self._dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(vectors)
        ivf.add(vectors)
        ivf.nprobe = min(64, nlist)
        self._index = ivf
