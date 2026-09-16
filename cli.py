"""Command-line interface for the face pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import (
    CAMERAS_CONFIG,
    FAISS_INDEX_DIR,
    GALLERY_DIR,
    MATCH_THRESH,
    ROOT,
    WEIGHT_DET_DIR,
    WEIGHT_REC_DIR,
)
from weight_sync import sync_weights, weight_dirs_ready
from pipeline import FacePipeline
from pipeline_search import SearchPipeline
from camera.worker import MultiCameraRunner, load_cameras_config

OUTPUT_DIR = ROOT / "data" / "output"
LOG = logging.getLogger("facerec.cli")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)


def _default_out_path(image: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{image.stem}_annotated.jpg"


def cmd_doctor(_: argparse.Namespace) -> int:
    import cv2
    import numpy as np
    import onnxruntime as ort

    print("Python:", sys.version.split()[0])
    print("OpenCV:", cv2.__version__)
    print("ONNX Runtime:", ort.__version__)
    print("Providers:", ort.get_available_providers())
    print("Gallery:", GALLERY_DIR)
    print("Weights det:", WEIGHT_DET_DIR)
    print("Weights rec:", WEIGHT_REC_DIR)
    if not weight_dirs_ready():
        print("Syncing weight/det and weight/rec from model pack...")
        sync_weights()
    print("Loading models...")
    pipe = FacePipeline()
    _ = pipe.detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    print("OK — pipeline ready.")
    return 0


def cmd_weights(args: argparse.Namespace) -> int:
    src = sync_weights(force=args.force)
    print(f"Synced from: {src}")
    print("det:", sorted(p.name for p in WEIGHT_DET_DIR.glob("*.onnx")))
    print("rec:", sorted(p.name for p in WEIGHT_REC_DIR.glob("*.onnx")))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    pipe = FacePipeline(gallery_dir=args.gallery)
    names = sorted(pipe.recognizer.gallery.keys())
    if not names:
        print(f"No identities in {args.gallery}")
        return 0
    for name in names:
        print(name)
    print(f"({len(names)} total)")
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    if not args.image.is_file():
        LOG.error("Image not found: %s", args.image)
        return 1
    pipe = FacePipeline(gallery_dir=args.gallery, match_thresh=args.thresh)
    pipe.enroll_file(args.name, args.image)
    pipe.save_gallery()
    print(f"Enrolled '{args.name}' -> {args.gallery}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    if not args.image.is_file():
        LOG.error("Image not found: %s", args.image)
        return 1
    pipe = FacePipeline(gallery_dir=args.gallery, match_thresh=args.thresh)
    img = pipe.read_image(args.image)
    results = pipe.process_image(img, max_faces=args.max_faces)
    if not results:
        print("No faces detected.")
        return 1
    for i, r in enumerate(results):
        if r.match.matched:
            print(f"Face {i}: {r.match.identity} (score={r.match.score:.3f})")
        else:
            print(f"Face {i}: unknown (best score={r.match.score:.3f})")
    out = args.out or _default_out_path(args.image)
    out.parent.mkdir(parents=True, exist_ok=True)
    annotated = pipe.draw_results(img, results)
    import cv2

    cv2.imwrite(str(out), annotated)
    print(f"Annotated image: {out}")
    return 0


def cmd_index_enroll(args: argparse.Namespace) -> int:
    if not args.image.is_file():
        LOG.error("Image not found: %s", args.image)
        return 1
    pipe = SearchPipeline(index_dir=args.index, match_thresh=args.thresh)
    pipe.enroll_index_file(args.name, args.image)
    pipe.save_index()
    print(f"Index size: {pipe.faiss.size} -> {args.index}")
    return 0


def cmd_index_build(args: argparse.Namespace) -> int:
    if not args.append and not (
        args.manifest or args.folder or args.from_gallery
    ):
        LOG.error(
            "Nothing to build. Use --manifest, --folder, or --from-gallery "
            "(or --append to add to the existing index)."
        )
        return 1

    from faceRec.bulk_enroll import migrate_numpy_gallery

    pipe = SearchPipeline(
        index_dir=args.index,
        match_thresh=args.thresh,
        load_existing_index=args.append,
        gallery_dir=args.gallery,
    )
    ok = fail = 0
    if args.from_gallery:
        migrated = migrate_numpy_gallery(pipe.faiss, args.gallery)
        ok += migrated
        print(f"Migrated {migrated} identities from {args.gallery}")
    if args.manifest:
        o, f = pipe.bulk_enroll_manifest(args.manifest)
        ok += o
        fail += f
    if args.folder:
        o, f = pipe.bulk_enroll_folders(args.folder)
        ok += o
        fail += f
    pipe.save_index()
    if args.ivf:
        pipe.faiss.build_ivf()
        pipe.save_index()
    print(f"Built index: {pipe.faiss.size} vectors ({ok} ok, {fail} failed) -> {args.index}")
    return 0


def cmd_index_verify(args: argparse.Namespace) -> int:
    if not args.image.is_file():
        LOG.error("Image not found: %s", args.image)
        return 1
    pipe = SearchPipeline(index_dir=args.index, match_thresh=args.thresh)
    if pipe.faiss.size == 0:
        LOG.error("Empty index at %s — run index-build or index-enroll first", args.index)
        return 1
    img = pipe.read_image(args.image)
    results = pipe.process_image(img, max_faces=args.max_faces)
    if not results:
        print("No faces detected.")
        return 1
    for i, r in enumerate(results):
        if r.match.matched:
            print(f"Face {i}: {r.match.identity} (score={r.match.score:.3f})")
        else:
            print(f"Face {i}: unknown (best score={r.match.score:.3f})")
    out = args.out or _default_out_path(args.image)
    out.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    cv2.imwrite(str(out), pipe.draw_results(img, results))
    print(f"Annotated image: {out}")
    return 0


def cmd_index_info(args: argparse.Namespace) -> int:
    try:
        from faceRec.faiss_gallery import FaissGallery

        gal = FaissGallery.load(args.index, match_thresh=args.thresh)
        print(f"Index: {args.index}")
        print(f"Vectors: {gal.size}")
    except FileNotFoundError:
        print(f"No index at {args.index}")
        return 1
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Smoke test: weights, models, FAISS index, sample verify."""
    sample = ROOT / "data" / "samples" / "t1.jpg"
    sample.parent.mkdir(parents=True, exist_ok=True)
    if not sample.is_file():
        import urllib.request

        url = (
            "https://raw.githubusercontent.com/deepinsight/insightface/master/"
            "python-package/insightface/data/images/t1.jpg"
        )
        print(f"Downloading sample image -> {sample}")
        try:
            urllib.request.urlretrieve(url, sample)
        except Exception as exc:
            LOG.error("Sample download failed: %s", exc)
            LOG.error("Place a face image at %s and re-run test", sample)
            return 1

    if not weight_dirs_ready():
        sync_weights()

    print("== doctor ==")
    if cmd_doctor(args) != 0:
        return 1

    print("== index-build --from-gallery ==")
    ns = argparse.Namespace(
        index=FAISS_INDEX_DIR,
        thresh=MATCH_THRESH,
        manifest=None,
        folder=None,
        from_gallery=True,
        gallery=GALLERY_DIR,
        append=False,
        ivf=False,
    )
    if cmd_index_build(ns) != 0:
        return 1

    print("== index-verify ==")
    ns = argparse.Namespace(
        index=FAISS_INDEX_DIR,
        thresh=MATCH_THRESH,
        image=sample,
        out=None,
        max_faces=0,
    )
    if cmd_index_verify(ns) != 0:
        return 1

    print("PASS — pipeline OK")
    return 0


def cmd_cameras(args: argparse.Namespace) -> int:
    specs = load_cameras_config(args.config)
    if not specs:
        LOG.error("No cameras in %s", args.config)
        return 1
    runner = MultiCameraRunner(
        specs,
        index_dir=args.index,
        match_thresh=args.thresh,
        show_window=args.show,
    )
    runner.run_until_interrupt()
    return 0


def cmd_webcam(args: argparse.Namespace) -> int:
    import cv2

    pipe = FacePipeline(gallery_dir=args.gallery, match_thresh=args.thresh)
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        LOG.error("Could not open camera device %s", args.device)
        return 1

    print("Webcam running — press Q or Esc to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = pipe.process_image(frame)
        display = pipe.draw_results(frame, results)
        cv2.imshow("faceRec", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Face detection + recognition (InsightFace buffalo_l)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Check deps and load models")
    p_doc.set_defaults(func=cmd_doctor)

    p_test = sub.add_parser("test", help="Smoke test (doctor + index + sample verify)")
    p_test.set_defaults(func=cmd_test)

    p_w = sub.add_parser(
        "weights",
        help="Copy/link ONNX files into weight/det and weight/rec",
    )
    p_w.add_argument(
        "--force",
        action="store_true",
        help="Replace existing files in weight/",
    )
    p_w.set_defaults(func=cmd_weights)

    p_list = sub.add_parser("list", help="List enrolled identities")
    p_list.add_argument("--gallery", type=Path, default=GALLERY_DIR)
    p_list.set_defaults(func=cmd_list)

    p_en = sub.add_parser("enroll", help="Enroll one person from an image")
    p_en.add_argument("--name", required=True)
    p_en.add_argument("--image", required=True, type=Path)
    p_en.add_argument("--gallery", type=Path, default=GALLERY_DIR)
    p_en.add_argument("--thresh", type=float, default=MATCH_THRESH)
    p_en.set_defaults(func=cmd_enroll)

    p_ver = sub.add_parser("verify", help="Detect and match faces in an image")
    p_ver.add_argument("--image", required=True, type=Path)
    p_ver.add_argument("--gallery", type=Path, default=GALLERY_DIR)
    p_ver.add_argument("--out", type=Path, help="Annotated output (default: data/output/)")
    p_ver.add_argument("--thresh", type=float, default=MATCH_THRESH)
    p_ver.add_argument("--max-faces", type=int, default=0, help="0 = all faces")
    p_ver.set_defaults(func=cmd_verify)

    p_cam = sub.add_parser("webcam", help="Live recognition from camera")
    p_cam.add_argument("--gallery", type=Path, default=GALLERY_DIR)
    p_cam.add_argument("--thresh", type=float, default=MATCH_THRESH)
    p_cam.add_argument("--device", type=int, default=0, help="Camera index")
    p_cam.set_defaults(func=cmd_webcam)

    idx = argparse.ArgumentParser(add_help=False)
    idx.add_argument("--index", type=Path, default=FAISS_INDEX_DIR)
    idx.add_argument("--thresh", type=float, default=MATCH_THRESH)

    p_ie = sub.add_parser("index-enroll", parents=[idx], help="Add one face to FAISS index")
    p_ie.add_argument("--name", required=True)
    p_ie.add_argument("--image", required=True, type=Path)
    p_ie.set_defaults(func=cmd_index_enroll)

    p_ib = sub.add_parser("index-build", parents=[idx], help="Bulk build FAISS index")
    p_ib.add_argument("--manifest", type=Path, help="CSV: identity,image_path")
    p_ib.add_argument("--folder", type=Path, help="Root with subfolders per identity")
    p_ib.add_argument("--from-gallery", action="store_true", help="Import data/gallery/*.npy")
    p_ib.add_argument("--gallery", type=Path, default=GALLERY_DIR)
    p_ib.add_argument(
        "--append",
        action="store_true",
        help="Add to existing index instead of rebuilding from scratch",
    )
    p_ib.add_argument("--ivf", action="store_true", help="Build IVF index (large N)")
    p_ib.set_defaults(func=cmd_index_build)

    p_iv = sub.add_parser("index-verify", parents=[idx], help="Match image against FAISS index")
    p_iv.add_argument("--image", required=True, type=Path)
    p_iv.add_argument("--out", type=Path)
    p_iv.add_argument("--max-faces", type=int, default=0)
    p_iv.set_defaults(func=cmd_index_verify)

    p_ii = sub.add_parser("index-info", parents=[idx], help="Show index vector count")
    p_ii.set_defaults(func=cmd_index_info)

    p_mcam = sub.add_parser("cameras", parents=[idx], help="Search faces on configured RTSP/USB streams")
    p_mcam.add_argument("--config", type=Path, default=CAMERAS_CONFIG)
    p_mcam.add_argument("--show", action="store_true", help="Open preview windows")
    p_mcam.set_defaults(func=cmd_cameras)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
