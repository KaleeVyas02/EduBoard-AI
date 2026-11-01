"""Batch processing for 57 classroom images. Run: python analyze_dataset.py"""
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pipeline import process_image

ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT / 'sample_images'
OUTPUT = ROOT / 'results'
OUTPUT.mkdir(exist_ok=True)
COLUMNS = ['filename', 'board_detected', 'blur_before', 'blur_after',
           'brightness_before', 'brightness_after', 'contrast_before',
           'contrast_after', 'ocr_characters', 'ocr_original_characters',
           'ocr_enhanced_characters', 'error']


def process_one(path):
    row = dict.fromkeys(COLUMNS, '')
    row['filename'] = path.name
    try:
        result = process_image(path, OUTPUT / (path.stem + '_' + path.suffix[1:].lower()), quick=True)
        row.update({
            'board_detected': result['board_detected'],
            'blur_before': result['before']['blur_score'],
            'blur_after': result['after']['blur_score'],
            'brightness_before': result['before']['brightness'],
            'brightness_after': result['after']['brightness'],
            'contrast_before': result['before']['contrast'],
            'contrast_after': result['after']['contrast'],
            'ocr_characters': result['characters'],
            'ocr_original_characters': result['ocr_original_characters'],
            'ocr_enhanced_characters': result['ocr_enhanced_characters']})
    except Exception as error:
        row['error'] = str(error)
    return row


def main():
    photos = sorted(p for p in PHOTOS.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    print('Found', len(photos), 'images. Processing...', flush=True)
    rows = []
    # Four images at a time; reduces waiting without changing beginner-friendly pipeline.py.
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(process_one, path): path for path in photos}
        for task in as_completed(tasks):
            row = task.result()
            rows.append(row)
            print(len(rows), '/', len(photos), row['filename'], 'OK' if not row['error'] else 'ERROR', flush=True)
    rows.sort(key=lambda row: row['filename'])
    with (OUTPUT / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print('Finished:', len(rows) - sum(bool(r['error']) for r in rows), 'successful images.')
    print('Summary:', OUTPUT / 'summary.csv')


if __name__ == '__main__':
    main()
