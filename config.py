"""Shared settings for detection, recognition, and matching."""

from pathlib import Path

# Project root (parent of this file)
ROOT = Path(__file__).resolve().parent

# InsightFace model cache (download source when weight/ is empty)
INSIGHTFACE_ROOT = ROOT / "data" / "insightface"

# Local ONNX weights (detection vs recognition)
WEIGHT_ROOT = ROOT / "weight"
WEIGHT_DET_DIR = WEIGHT_ROOT / "det"
WEIGHT_REC_DIR = WEIGHT_ROOT / "rec"

# InsightFace model pack: buffalo_l = best accuracy; buffalo_sc = faster / smaller
MODEL_NAME = "buffalo_l"

# Detection input size (larger helps small / distant faces; slower)
DET_SIZE = (640, 640)

# Minimum detection confidence (0–1)
DET_THRESH = 0.5

# Cosine similarity threshold for identity match (normed ArcFace embeddings)
# Typical range: 0.35–0.55 depending on lighting and enrollment quality
MATCH_THRESH = 0.45

# ONNX Runtime providers (CUDA first when available)
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# GPU device id for InsightFace ctx_id (-1 = CPU)
CTX_ID = 0


def resolve_providers(preferred: list[str] | None = None) -> list[str]:
    """ONNX Runtime providers that actually exist on this machine."""
    import onnxruntime as ort

    avail = set(ort.get_available_providers())
    prefs = preferred if preferred is not None else PROVIDERS
    chosen = [p for p in prefs if p in avail]
    return chosen or ["CPUExecutionProvider"]


def effective_ctx_id(ctx_id: int = CTX_ID) -> int:
    """Use CPU when CUDA provider is not installed."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return -1
    return ctx_id

# Small gallery (JSON + .npy files) for dev / few identities
GALLERY_DIR = ROOT / "data" / "gallery"

# Large-scale index (100k+ identities) — FAISS on disk
FAISS_INDEX_DIR = ROOT / "data" / "index"
FAISS_INDEX_FILE = "faces.index"
FAISS_IDS_FILE = "ids.json"
EMBED_DIM = 512

# Search: min cosine score and gap between 1st and 2nd hit (reduces false positives)
SEARCH_MARGIN = 0.08

# Multi-camera ingest
CAMERAS_CONFIG = ROOT / "data" / "cameras.yaml"
CAMERA_FRAME_SKIP = 4  # process every Nth frame per stream
CAMERA_EMBED_INTERVAL_SEC = 2.0  # min seconds between embeds per track
CAMERA_MAX_STREAMS = 32  # safety cap per worker process (scale out for 1000 cams)
