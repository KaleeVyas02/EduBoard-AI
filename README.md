# EduBoard AI — Final beginner-friendly FOML project

## What it does
- Reads classroom photos (JPG/PNG)
- Attempts automatic four-corner board/screen detection and perspective correction (falls back to original when not found)
- Improves local contrast and sharpness using OpenCV
- Extracts text using pretrained Tesseract OCR
- Compares original/enhanced OCR character counts (NOT accuracy)
- Saves images, notes, and batch summary CSV
- Offers a local Django upload/download website

## Windows setup (VS Code terminal)
1. Install Python 3.11 or 3.12 (recommended) and Windows Tesseract OCR.
2. Open this folder in VS Code.
3. `py -3.11 -m venv venv` (or `python -m venv venv` with a compatible Python).
4. PowerShell: `Set-ExecutionPolicy -Scope Process Bypass` then `.\venv\Scripts\Activate.ps1`.
5. `python -m pip install -r requirements.txt`
6. If Tesseract is not on PATH, add near the top of `pipeline.py` after imports:
   `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`
7. `python analyze_dataset.py` (uses the included 3 sample images).
8. Extract your original 57 photos into `sample_images`, optionally remove sample images to avoid duplicates, and rerun `python analyze_dataset.py`.
9. In VS Code, open `EduBoard_AI_FOML.ipynb` and run each cell.
10. `cd web` then `python manage.py runserver` and open http://127.0.0.1:8000/.

## Evaluation
`results/summary.csv` contains blur, brightness, contrast, board-detection status, and OCR character counts. **These are not OCR accuracy scores.** For actual accuracy, manually transcribe 5–10 representative photos and calculate character error rate (CER) using the optional `evaluate_ocr.py` script.

## Limitations
Perspective correction may fail on faint edges, curved or obstructed boards. OCR struggles with small, distant, blurry text and mathematical notation. This is a local academic prototype, not a deployed production service. Images uploaded to Django are stored locally in `web/uploads`; avoid uploading private student data.


## Verified batch run (15 October)
All 57 photographs completed the fast batch run; 55 produced nonempty OCR output. See FINAL_TEST_RESULTS.txt and results/summary.csv. Accuracy has not been established without manual reference transcriptions. Django must still be run locally on Windows.
