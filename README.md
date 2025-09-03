# FaceSorter

A no-nonsense face-sorting script for Windows, macOS, or Linux. Uses InsightFace's `buffalo_l` model for face detection and recognition. Models download automatically on first run. Runs on CPU by default; GPU optional with `onnxruntime-gpu`. Sorts photos into `output/<Person>/` or `output/Unknown/`.

## Features
- Reads images (PNG, JPEG, WebP, BMP) with Pillow, falls back to OpenCV.
- Scans input folders recursively.
- Matches faces using cosine similarity with a tunable threshold.
- Filters out small faces to reduce noise.
- Option to copy or move files.
- Progress bar via `tqdm`.

## Setup
Use a virtual environment to avoid wrecking your system Python:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install insightface onnxruntime opencv-python numpy pillow tqdm
# Optional for HEIC (iPhone photos):
pip install pillow-heif
# Optional for GPU:
pip install onnxruntime-gpu
```
Deactivate when done:
```bash
deactivate
```

## Usage
Activate the virtual environment first:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
- `--gallery`: Folder with subfolders named by person (e.g., `./gallery/Alice/`, `./gallery/Bob/`) containing example images.
- `--input`: Folder of photos to sort (recursive).
- `--output`: Destination for sorted photos.
- `--threshold`: Cosine similarity threshold (0.30–0.45; start at 0.35, tweak as needed).
- `--min-face`: Minimum face size in pixels (default: 90).

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
├── .gitignore
├── .gitkeep
```
To include empty folders in Git, add `.gitkeep` files (e.g., `touch gallery/.gitkeep`). Add `venv/` to `.gitignore`.

## Notes
- **Threshold**: 0.30–0.45 works best. Lower risks false positives; higher is stricter.
- **Face Size**: `--min-face` skips tiny faces (e.g., background noise). Adjust for your image resolution.
- **Performance**: CPU is fine for most. GPU speeds things up for large datasets.
- **Formats**: Supports JPG, JPEG, PNG, WebP, BMP. HEIC needs `pillow-heif`.
- **Gallery**: Use clear, well-lit face images for best results.

## Troubleshooting
- **No faces detected**: Check image quality or lower `--min-face`.
- **Misclassified faces**: Tweak `--threshold` (higher for precision).
- **No embeddings**: Verify `gallery/` has valid images in named subfolders.
- **HEIC errors**: Install `pillow-heif`.

## License
MIT License. Use it, but don’t expect hand-holding.