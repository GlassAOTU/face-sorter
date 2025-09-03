# FaceSorter

A minimal, practical face-sorting script for Windows, macOS, or Linux. Uses InsightFace's `buffalo_l` model for face detection and recognition. Automatically downloads models on first run. Runs on CPU by default; GPU optional with `onnxruntime-gpu`. Sorts photos into `output/<Person>/` or `output/Unknown/`.

## Features
- Robust image reading with Pillow (supports PNG, JPEG, WebP, BMP) and OpenCV fallback.
- Recursive scanning of input photos.
- Cosine similarity-based matching with tunable threshold.
- Filters small faces to avoid noise.
- Option to copy or move files.
- Progress bar via `tqdm`.

## Prerequisites
Install dependencies:
```bash
pip install insightface onnxruntime opencv-python numpy pillow tqdm
```
For HEIC support (e.g., iPhone photos):
```bash
pip install pillow-heif
```
For GPU support (optional):
```bash
pip install onnxruntime-gpu
```

## Usage
Run from the project root:
```bash
python face_sorter.py \
  --gallery ./gallery \
  --input ./input_photos \
  --output ./output \
  --threshold 0.35 \
  --min-face 90
```
Optional flags:
- `--move`: Move files instead of copying.
- `--gpu`: Use GPU if `onnxruntime-gpu` is installed.

### Arguments
- `--gallery`: Folder with subfolders named by person (e.g., `./gallery/Alice/`, `./gallery/Bob/`), containing labeled example images.
- `--input`: Folder of photos to sort (recursively scanned).
- `--output`: Destination folder for sorted photos.
- `--threshold`: Cosine similarity threshold for matching (0.30–0.45; start with 0.35 and tune).
- `--min-face`: Minimum face size (pixels) to consider (default: 90).

## Folder Structure
```
project_root/
├── gallery/
│   ├── Person1/
│   │   ├── img1.jpg
│   │   ├── img2.png
│   ├── Person2/
│   │   ├── img3.jpg
├── input_photos/
│   ├── photo1.jpg
│   ├── photo2.png
├── output/
│   ├── Person1/
│   ├── Person2/
│   ├── Unknown/
```

## Notes
- **Threshold**: 0.30–0.45 is typical. Lower values increase matches but risk errors; higher values are stricter.
- **Face Size**: `--min-face` filters out small faces (e.g., in backgrounds). Adjust based on image resolution.
- **Performance**: CPU is sufficient for most cases. GPU can speed up processing for large datasets.
- **Image Formats**: Supports JPG, JPEG, PNG, WebP, BMP. HEIC requires `pillow-heif`.
- **Gallery Quality**: Use clear, well-lit face images in `gallery/` for best results.

## Troubleshooting
- **No faces detected**: Check image quality or lower `--min-face`.
- **Wrong matches**: Adjust `--threshold` (higher for stricter matching).
- **No embeddings**: Ensure `gallery/` has valid images and subfolders.
- **HEIC issues**: Install `pillow-heif` for iPhone photos.

## License
MIT License. Use freely, but no warranties provided.