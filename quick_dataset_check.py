"""Fast validation of ALL photos (quality metrics only, no OCR)."""
import csv
from pathlib import Path
import cv2
from pipeline import image_quality

root = Path(__file__).resolve().parent
photos = root / 'sample_images'
output = root / 'results'
output.mkdir(exist_ok=True)
rows = []
for photo in sorted(photos.iterdir()):
    if photo.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        continue
    image = cv2.imread(str(photo))
    if image is None:
        rows.append({'filename': photo.name, 'error': 'Cannot read image'})
        continue
    # Reduce resolution so the dataset scan is fast.
    h, w = image.shape[:2]
    if max(h, w) > 900:
        scale = 900 / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    row = {'filename': photo.name, 'error': ''}
    row.update(image_quality(image))
    rows.append(row)
with (output / 'quality_all_57.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'blur_score', 'brightness', 'contrast', 'error'])
    writer.writeheader()
    writer.writerows(rows)
print('Scanned:', len(rows), 'Unreadable:', sum(bool(r['error']) for r in rows))
