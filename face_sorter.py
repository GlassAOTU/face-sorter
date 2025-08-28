#!/usr/bin/env python3
"""
Minimal, practical face-sorting script for Windows/macOS/Linux.
- Uses InsightFace (buffalo_l) for detection+embeddings. Models auto-download on first run.
- CPU by default; GPU optional if onnxruntime-gpu is installed.
- Copies whole photos into output/<Person>/ (or Unknown/).

Usage (from project root):
  python face_sorter.py \
    --gallery ./gallery \
    --input ./input_photos \
    --output ./output \
    --threshold 0.35 \
    --min-face 90

Prereqs:
  pip install insightface onnxruntime opencv-python numpy pillow tqdm
  # If you have many HEICs:
  pip install pillow-heif

Notes:
  - Threshold 0.30–0.45 is typical. Start ~0.35 and tune.
  - Folder names under ./gallery are the person labels (e.g., ./gallery/Aba/...).
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

# OpenCV is used for BGR image read if Pillow path fails
import cv2


# Pillow for broader format support (PNG, JPEG)
from PIL import Image

# InsightFace
from insightface.app import FaceAnalysis

# ----------------------------
# Utils
# ----------------------------


def imread_any(path: Path):
    """Read image robustly. Try Pillow first, fallback to OpenCV. Returns RGB np.ndarray or None."""
    try:
        with Image.open(path) as im:
            im = im.convert('RGB')
            return np.array(im)
    except Exception:
        img = cv2.imread(str(path))
        if img is None:
            return None
        # cv2 loads BGR, convert to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are L2-normalized by InsightFace already, but be defensive
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


# ----------------------------
# Gallery building
# ----------------------------

def build_gallery(app: FaceAnalysis, gallery_dir: Path, min_face: int) -> Dict[str, List[np.ndarray]]:
    """Return mapping: name -> list of normed embeddings."""
    name_to_embs: Dict[str, List[np.ndarray]] = {}
    ppl = [d for d in gallery_dir.iterdir() if d.is_dir()]
    if not ppl:
        raise SystemExit(f"No person folders found in gallery: {gallery_dir}")

    for person_dir in ppl:
        person = person_dir.name
        embs: List[np.ndarray] = []
        imgs = [p for p in person_dir.rglob(
            '*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
        for img_path in imgs:
            img = imread_any(img_path)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            # choose the largest face (most likely the subject)
            faces.sort(key=lambda f: (
                f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            f0 = faces[0]
            w = f0.bbox[2]-f0.bbox[0]
            h = f0.bbox[3]-f0.bbox[1]
            if min(w, h) < min_face:
                continue
            emb = f0.normed_embedding
            if emb is None:
                continue
            embs.append(emb.astype(np.float32))
        if embs:
            name_to_embs[person] = embs
        else:
            print(f"[warn] No usable faces for {person}")
    if not name_to_embs:
        raise SystemExit(
            "No embeddings created from gallery. Check images/face sizes.")
    return name_to_embs


# ----------------------------
# Matching
# ----------------------------

def best_match(query_emb: np.ndarray, gallery: Dict[str, List[np.ndarray]]) -> Tuple[str, float]:
    best_name = "Unknown"
    best_score = -1.0
    for name, embs in gallery.items():
        # Take max similarity among that person's examples
        score = max(cosine_sim(query_emb, g) for g in embs)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score


# ----------------------------
# Main processing
# ----------------------------

def process(app: FaceAnalysis, gallery: Dict[str, List[np.ndarray]], input_dir: Path, output_dir: Path,
            threshold: float, min_face: int, copy_mode: str = 'copy'):
    ensure_dir(output_dir)
    ensure_dir(output_dir / 'Unknown')

    # Collect images
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    img_paths = [p for p in input_dir.rglob('*') if p.suffix.lower() in exts]
    if not img_paths:
        print(f"No images found under {input_dir}")
        return

    for p in tqdm(img_paths, desc="Scanning photos"):
        img = imread_any(p)
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            # No faces at all -> leave it alone or classify Unknown
            dst = output_dir / 'Unknown' / p.name
            ensure_dir(dst.parent)
            (shutil.copy2 if copy_mode == 'copy' else shutil.move)(p, dst)
            continue

        assigned_names = set()
        for f in faces:
            w = f.bbox[2]-f.bbox[0]
            h = f.bbox[3]-f.bbox[1]
            if min(w, h) < min_face:
                continue
            emb = f.normed_embedding
            if emb is None:
                continue
            name, score = best_match(emb.astype(np.float32), gallery)
            if score >= threshold:
                assigned_names.add(name)
        if assigned_names:
            for name in assigned_names:
                dst = output_dir / name / p.name
                ensure_dir(dst.parent)
                (shutil.copy2 if copy_mode == 'copy' else shutil.move)(p, dst)
        else:
            dst = output_dir / 'Unknown' / p.name
            ensure_dir(dst.parent)
            (shutil.copy2 if copy_mode == 'copy' else shutil.move)(p, dst)


# ----------------------------
# Entry
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sort photos into per-person folders using face recognition.")
    ap.add_argument('--gallery', type=Path, required=True,
                    help='Folder with subfolders per person (labeled examples).')
    ap.add_argument('--input', type=Path, required=True,
                    help='Folder of photos to scan (recurses).')
    ap.add_argument('--output', type=Path, required=True,
                    help='Where sorted photos will be copied/moved.')
    ap.add_argument('--threshold', type=float, default=0.35,
                    help='Cosine similarity threshold (0.30–0.45 typical).')
    ap.add_argument('--min-face', type=int, default=90,
                    help='Minimum face box side (pixels) to consider.')
    ap.add_argument('--move', action='store_true',
                    help='Move files instead of copying.')
    ap.add_argument('--gpu', action='store_true',
                    help='Use GPU provider if onnxruntime-gpu is installed.')
    args = ap.parse_args()

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.gpu else [
        'CPUExecutionProvider']

    # Init InsightFace (buffalo_l includes detector+recognizer by default)
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    # ctx_id is ignored when providers specified; size=640 balances speed/accuracy
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Build gallery embeddings
    gallery = build_gallery(app, args.gallery, args.min_face)

    # Process inputs
    copy_mode = 'move' if args.move else 'copy'
    process(app, gallery, args.input, args.output,
            args.threshold, args.min_face, copy_mode)


if __name__ == '__main__':
    main()
