"""Populate weight/det and weight/rec from the InsightFace model pack."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, List, Optional, Sequence

from config import INSIGHTFACE_ROOT, MODEL_NAME, WEIGHT_DET_DIR, WEIGHT_REC_DIR

LOG = logging.getLogger(__name__)

# Files used by this pipeline (buffalo_l layout)
DET_ONNX = ("det_10g.onnx", "2d106det.onnx")
REC_ONNX = ("w600k_r50.onnx",)

_MIN_ONNX_BYTES = 100_000


def _pack_dir() -> Path:
    return INSIGHTFACE_ROOT / "models" / MODEL_NAME


def _valid_onnx(path: Path) -> bool:
    """True if path is a real ONNX file (not a broken symlink)."""
    try:
        if path.is_symlink():
            path = path.resolve()
        return path.is_file() and path.stat().st_size >= _MIN_ONNX_BYTES
    except OSError:
        return False


def weight_dirs_ready() -> bool:
    return all(_valid_onnx(WEIGHT_DET_DIR / f) for f in DET_ONNX) and all(
        _valid_onnx(WEIGHT_REC_DIR / f) for f in REC_ONNX
    )


def sync_weights(
    force: bool = False,
    copy_files: bool = True,
) -> Path:
    """
    Install ONNX files into ``weight/det`` and ``weight/rec``.

    By default **copies** files so ``weight/`` works even if ``data/insightface`` is removed.
    Set ``copy_files=False`` to create symlinks instead (saves disk space).
    """
    from insightface.utils import ensure_available

    src = Path(ensure_available("models", MODEL_NAME, root=str(INSIGHTFACE_ROOT)))
    WEIGHT_DET_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_REC_DIR.mkdir(parents=True, exist_ok=True)

    for name in DET_ONNX:
        _install_file(src / name, WEIGHT_DET_DIR / name, force=force, copy_files=copy_files)
    for name in REC_ONNX:
        _install_file(src / name, WEIGHT_REC_DIR / name, force=force, copy_files=copy_files)

    if not weight_dirs_ready():
        raise RuntimeError(
            "weight/det or weight/rec still incomplete after sync. "
            f"Run: python run.py weights --force"
        )
    return src


def ensure_weights(force: bool = False) -> None:
    """Download (if needed) and install weights when missing or broken."""
    if weight_dirs_ready() and not force:
        return
    LOG.info("Installing ONNX weights into weight/det and weight/rec ...")
    sync_weights(force=True, copy_files=True)


def _install_file(
    src: Path,
    dst: Path,
    force: bool,
    copy_files: bool,
) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing model file in pack: {src}")

    if not force and _valid_onnx(dst):
        return

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy_files:
        shutil.copy2(src, dst)
        return

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

    ensure_weights()

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
