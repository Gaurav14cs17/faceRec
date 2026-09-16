"""Populate weight/det and weight/rec from the InsightFace model pack."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List, Optional, Sequence

from config import INSIGHTFACE_ROOT, MODEL_NAME, WEIGHT_DET_DIR, WEIGHT_REC_DIR

# Files used by this pipeline (buffalo_l layout)
DET_ONNX = ("det_10g.onnx", "2d106det.onnx")
REC_ONNX = ("w600k_r50.onnx",)


def _pack_dir() -> Path:
    return INSIGHTFACE_ROOT / "models" / MODEL_NAME


def weight_dirs_ready() -> bool:
    return all((WEIGHT_DET_DIR / f).is_file() for f in DET_ONNX) and all(
        (WEIGHT_REC_DIR / f).is_file() for f in REC_ONNX
    )


def sync_weights(force: bool = False) -> Path:
    """
    Copy ONNX files from the downloaded ``buffalo_*`` pack into ``weight/det`` and ``weight/rec``.
    Returns the pack directory used as source.
    """
    from insightface.utils import ensure_available

    src = Path(ensure_available("models", MODEL_NAME, root=str(INSIGHTFACE_ROOT)))
    WEIGHT_DET_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_REC_DIR.mkdir(parents=True, exist_ok=True)

    for name in DET_ONNX:
        _link_or_copy(src / name, WEIGHT_DET_DIR / name, force=force)
    for name in REC_ONNX:
        _link_or_copy(src / name, WEIGHT_REC_DIR / name, force=force)

    return src


def _link_or_copy(src: Path, dst: Path, force: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing model file: {src}")
    if dst.exists() and not force:
        return
    if dst.exists():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _onnx_candidates(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.onnx"))


def load_recognition_model(
    providers: Optional[Sequence[str]] = None,
    model_name: str = MODEL_NAME,
) -> Any:
    """Load ArcFace ONNX from ``weight/rec`` (InsightFace cannot use rec-only FaceAnalysis)."""
    from insightface.model_zoo import model_zoo
    from insightface.utils import ensure_available

    from config import resolve_providers

    provider_list = (
        list(providers) if providers is not None else resolve_providers()
    )

    if not weight_dirs_ready():
        sync_weights()

    search_dirs: List[Path] = []
    if weight_dirs_ready():
        search_dirs.append(WEIGHT_REC_DIR)
    search_dirs.append(
        Path(ensure_available("models", model_name, root=str(INSIGHTFACE_ROOT)))
    )

    for directory in search_dirs:
        for path in _onnx_candidates(directory):
            model = model_zoo.get_model(str(path), providers=provider_list)
            if model is not None and getattr(model, "taskname", None) == "recognition":
                return model

    raise RuntimeError("No recognition ONNX model found in weight/rec or model pack")
