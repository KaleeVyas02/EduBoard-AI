"""EduBoard AI: beginner-friendly OpenCV + pretrained Tesseract OCR."""
from pathlib import Path
import cv2
import numpy as np
import pytesseract

import os

# On Windows, find the Tesseract program automatically if installed normally.
if os.name == "nt":
    program = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if program.exists():
        pytesseract.pytesseract.tesseract_cmd = str(program)


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('Please choose a valid JPG or PNG image.')
    return image


def image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    brightness = round(float(gray.mean()), 2)
    contrast = round(float(gray.std()), 2)
    return {'blur_score': blur, 'brightness': brightness, 'contrast': contrast}


def correct_perspective(image):
    # Find a large four-sided board/screen; otherwise preserve original image.
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:25]:
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(corners) != 4 or cv2.contourArea(corners) < gray.size * 0.12:
            continue
        points = corners.reshape(4, 2).astype('float32') / scale
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).reshape(-1)
        tl = points[np.argmin(sums)]
        tr = points[np.argmin(differences)]
        br = points[np.argmax(sums)]
        bl = points[np.argmax(differences)]
        ordered = np.array([tl, tr, br, bl], dtype='float32')
        new_width = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
        new_height = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
        if new_width < 120 or new_height < 100 or new_width * new_height > 12_000_000:
            continue
        target = np.array([[0,0], [new_width-1,0], [new_width-1,new_height-1], [0,new_height-1]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(ordered, target)
        return cv2.warpPerspective(image, matrix, (new_width, new_height)), True
    return image.copy(), False


def enhance_image(image):
    # Local contrast enhancement and mild sharpening.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    light = clahe.apply(light)
    improved = cv2.cvtColor(cv2.merge([light, a, b]), cv2.COLOR_LAB2BGR)
    soft = cv2.GaussianBlur(improved, (0, 0), 1.0)
    sharp = cv2.addWeighted(improved, 1.35, soft, -0.35, 0)
    return sharp


def extract_text(image, timeout=4):
    # Tesseract is a pretrained OCR engine. Results can be imperfect.
    height, width = image.shape[:2]
    if max(height, width) > 800:
        scale = 800 / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        return pytesseract.image_to_string(rgb, config='--psm 11', timeout=timeout).strip()
    except RuntimeError as error:
        # Some large or blurry images may take too long; continue the batch.
        if 'timeout' in str(error).lower():
            return ''
        raise


def process_image(input_path, output_folder, quick=False):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    image = read_image(input_path)
    # Limit huge phone photos to make processing practical on laptops.
    height, width = image.shape[:2]
    if max(height, width) > 1100:
        scale = 1100 / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    corrected, found_board = correct_perspective(image)
    enhanced = enhance_image(corrected)
    before = image_quality(image)
    after = image_quality(enhanced)
    # Compare OCR on corrected original and enhanced versions.
    # More characters are not necessarily more accurate: review the text manually.
    original_text = "" if quick else extract_text(corrected)
    enhanced_text = extract_text(enhanced, timeout=2 if quick else 4)
    text = enhanced_text if len(enhanced_text.strip()) >= len(original_text.strip()) else original_text
    cv2.imwrite(str(output_folder / 'original.jpg'), image)
    cv2.imwrite(str(output_folder / 'corrected.jpg'), corrected)
    cv2.imwrite(str(output_folder / 'enhanced.jpg'), enhanced)
    (output_folder / 'notes.txt').write_text(text, encoding='utf-8')
    return {'board_detected': found_board, 'before': before, 'after': after,
            'characters': len(text), 'text': text,
            'ocr_original_characters': len(original_text),
            'ocr_enhanced_characters': len(enhanced_text)}
