"""EduBoard AI: image enhancement and pretrained Tesseract OCR."""

import os
import shutil
from pathlib import Path

# Reduce excessive CPU threading on small hosting instances.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2
import numpy as np
import pytesseract

cv2.setNumThreads(1)

# Locate Tesseract on Windows or Linux.
if os.name == "nt":
    windows_tesseract = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    if windows_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(
            windows_tesseract
        )
else:
    pytesseract.pytesseract.tesseract_cmd = (
        shutil.which("tesseract") or "tesseract"
    )


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Please choose a valid JPG or PNG image.")
    return image


def image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        "blur_score": round(
            float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2
        ),
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
    }


def correct_perspective(image):
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / max(height, width))

    small = (
        cv2.resize(image, None, fx=scale, fy=scale)
        if scale < 1
        else image.copy()
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in sorted(
        contours, key=cv2.contourArea, reverse=True
    )[:25]:
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if (
            len(corners) != 4
            or not cv2.isContourConvex(corners)
            or cv2.contourArea(corners) < gray.size * 0.12
        ):
            continue

        points = corners.reshape(4, 2).astype("float32") / scale
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).reshape(-1)

        tl = points[np.argmin(sums)]
        tr = points[np.argmin(differences)]
        br = points[np.argmax(sums)]
        bl = points[np.argmax(differences)]

        ordered = np.array([tl, tr, br, bl], dtype="float32")

        if len(np.unique(ordered, axis=0)) != 4:
            continue

        new_width = int(
            max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        )
        new_height = int(
            max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        )

        if (
            new_width < 120
            or new_height < 100
            or new_width * new_height > 12_000_000
        ):
            continue

        target = np.array(
            [
                [0, 0],
                [new_width - 1, 0],
                [new_width - 1, new_height - 1],
                [0, new_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(ordered, target)

        corrected = cv2.warpPerspective(
            image, matrix, (new_width, new_height)
        )

        return corrected, True

    return image.copy(), False


def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    )
    light = clahe.apply(light)

    improved = cv2.cvtColor(
        cv2.merge([light, a, b]), cv2.COLOR_LAB2BGR
    )

    soft = cv2.GaussianBlur(improved, (0, 0), 1.0)
    return cv2.addWeighted(improved, 1.35, soft, -0.35, 0)


def prepare_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Keep useful detail while limiting memory and processing time.
    longest = max(height, width)
    target = 1600
    scale = min(2.0, target / longest)

    if abs(scale - 1.0) > 0.05:
        interpolation = (
            cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        )
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=interpolation,
        )

    gray = cv2.normalize(
        gray, None, 0, 255, cv2.NORM_MINMAX
    )

    # Convert dark boards to a light background.
    if np.median(gray) < 127:
        gray = cv2.bitwise_not(gray)

    return cv2.copyMakeBorder(
        gray,
        20, 20, 20, 20,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def extract_text(image, timeout=25, psm=11, threshold=False):
    prepared = prepare_for_ocr(image)

    if threshold:
        _, prepared = cv2.threshold(
            prepared,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    try:
        text = pytesseract.image_to_string(
            prepared,
            lang="eng",
            config=f"--oem 1 --psm {psm}",
            timeout=timeout,
        ).strip()

        print(
            f"OCR: psm={psm}, threshold={threshold}, "
            f"characters={len(text)}",
            flush=True,
        )
        return text

    except RuntimeError as error:
        if "timeout" in str(error).lower():
            print(
                f"OCR timed out after {timeout} seconds",
                flush=True,
            )
            return ""

        raise


def useful_text(text):
    return sum(character.isalnum() for character in text) >= 3


def process_image(input_path, output_folder, quick=False):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image = read_image(input_path)

    # Preserve more detail than the original 1100-pixel limit.
    height, width = image.shape[:2]
    if max(height, width) > 2000:
        scale = 2000 / max(height, width)
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    corrected, found_board = correct_perspective(image)
    enhanced = enhance_image(corrected)

    before = image_quality(image)
    after = image_quality(enhanced)

    # Save images even if OCR later reports an error.
    cv2.imwrite(str(output_folder / "original.jpg"), image)
    cv2.imwrite(str(output_folder / "corrected.jpg"), corrected)
    cv2.imwrite(str(output_folder / "enhanced.jpg"), enhanced)

    timeout = 10 if quick else 25

    # Read the enhanced image without aggressive thresholding first.
    enhanced_text = extract_text(
        enhanced, timeout=timeout, psm=11
    )

    original_text = ""
    fallback_text = ""

    if not quick:
        original_text = extract_text(
            corrected, timeout=timeout, psm=11
        )

        # Retry using a text-block layout and binary image.
        if not useful_text(enhanced_text) and not useful_text(
            original_text
        ):
            fallback_text = extract_text(
                image,
                timeout=timeout,
                psm=6,
                threshold=True,
            )

    # More characters do not establish OCR accuracy.
    candidates = [enhanced_text, original_text, fallback_text]
    text = max(
        candidates,
        key=lambda value: sum(c.isalnum() for c in value),
    )

    if not useful_text(text):
        text = ""

    (output_folder / "notes.txt").write_text(
        text, encoding="utf-8"
    )

    return {
        "board_detected": found_board,
        "before": before,
        "after": after,
        "characters": len(text),
        "text": text,
        "ocr_original_characters": len(original_text),
        "ocr_enhanced_characters": len(enhanced_text),
    }
