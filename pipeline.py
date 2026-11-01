"""EduBoard AI: image enhancement and Tesseract OCR."""

import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2
import numpy as np
import pytesseract

cv2.setNumThreads(1)

# Find Tesseract on Windows or Linux.
if os.name == "nt":
    program = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    if program.exists():
        pytesseract.pytesseract.tesseract_cmd = str(program)
else:
    pytesseract.pytesseract.tesseract_cmd = (
        shutil.which("tesseract") or "tesseract"
    )


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Please choose a valid JPG or PNG image.")
    return image


def resize_limit(image, maximum=2200):
    height, width = image.shape[:2]
    if max(height, width) <= maximum:
        return image

    scale = maximum / max(height, width)
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


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
    """Try detecting a large rectangular board or screen."""
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / max(height, width))

    small = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0), 50, 150
    )

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in sorted(
        contours, key=cv2.contourArea, reverse=True
    )[:30]:
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if (
            len(corners) != 4
            or not cv2.isContourConvex(corners)
            or cv2.contourArea(corners) < gray.size * 0.15
        ):
            continue

        points = corners.reshape(4, 2).astype("float32") / scale
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).reshape(-1)

        ordered = np.array(
            [
                points[np.argmin(sums)],
                points[np.argmin(differences)],
                points[np.argmax(sums)],
                points[np.argmax(differences)],
            ],
            dtype="float32",
        )

        if len(np.unique(ordered, axis=0)) != 4:
            continue

        tl, tr, br, bl = ordered

        new_width = int(
            max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        )
        new_height = int(
            max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        )

        if (
            new_width < 200
            or new_height < 150
            or new_width * new_height > 10_000_000
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

        transform = cv2.getPerspectiveTransform(ordered, target)

        return cv2.warpPerspective(
            image, transform, (new_width, new_height)
        ), True

    return image.copy(), False


def enhance_image(image):
    """Improve local contrast and apply mild sharpening."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)

    light = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(light)

    improved = cv2.cvtColor(
        cv2.merge([light, a, b]), cv2.COLOR_LAB2BGR
    )

    blurred = cv2.GaussianBlur(improved, (0, 0), 1.0)
    return cv2.addWeighted(improved, 1.35, blurred, -0.35, 0)


def prepare_for_ocr(image, threshold=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Preserve detail without creating excessively large images.
    scale = min(2.0, 2200 / max(height, width))

    if abs(scale - 1.0) > 0.05:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=(
                cv2.INTER_CUBIC
                if scale > 1
                else cv2.INTER_AREA
            ),
        )

    # Support dark boards with light writing.
    if np.median(gray) < 100:
        gray = cv2.bitwise_not(gray)

    if threshold:
        gray = cv2.normalize(
            gray, None, 0, 255, cv2.NORM_MINMAX
        )
        _, gray = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    return cv2.copyMakeBorder(
        gray, 20, 20, 20, 20,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def read_ocr_candidate(image, timeout=25, psm=11, threshold=False):
    """Return text and a confidence-based selection score."""
    prepared = prepare_for_ocr(image, threshold=threshold)

    try:
        data = pytesseract.image_to_data(
            prepared,
            lang="eng",
            config=f"--oem 1 --psm {psm}",
            timeout=timeout,
            output_type=pytesseract.Output.DICT,
        )
    except RuntimeError as error:
        if "timeout" in str(error).lower():
            print(
                f"OCR timed out: psm={psm}, "
                f"threshold={threshold}",
                flush=True,
            )
            return {"text": "", "score": 0.0}
        raise

    lines = {}

    for index, raw_word in enumerate(data["text"]):
        word = raw_word.strip()
        confidence = float(data["conf"][index])

        if not word or confidence < 45:
            continue

        key = (
            data["block_num"][index],
            data["par_num"][index],
            data["line_num"][index],
        )

        lines.setdefault(key, []).append((word, confidence))

    retained_lines = []
    retained_words = []

    for words in lines.values():
        average = sum(
            confidence for _, confidence in words
        ) / len(words)

        # Reject isolated symbols and many diagram-only labels.
        contains_word = any(
            sum(character.isalpha() for character in word) >= 3
            for word, _ in words
        )

        if not contains_word or average < 55:
            continue

        retained_lines.append(
            " ".join(word for word, _ in words)
        )
        retained_words.extend(words)

    text = "\n".join(retained_lines)

    if not retained_words:
        return {"text": "", "score": 0.0}

    weighted_confidence = sum(
        confidence * len(word)
        for word, confidence in retained_words
    ) / sum(len(word) for word, _ in retained_words)

    # Confidence is a heuristic, not a measured accuracy percentage.
    # Penalize candidates containing very little useful text.
    word_count = sum(
        sum(character.isalpha() for character in word) >= 3
        for word, _ in retained_words
    )
    score = weighted_confidence * min(1.0, word_count / 8)

    print(
        f"OCR: psm={psm}, threshold={threshold}, "
        f"words={word_count}, selection_score={score:.1f}",
        flush=True,
    )

    return {"text": text, "score": score}


def extract_text(image, timeout=25):
    """Compatibility helper for other project scripts."""
    return read_ocr_candidate(
        image, timeout=timeout
    )["text"]


def process_image(input_path, output_folder, quick=False):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image = resize_limit(read_image(input_path))
    corrected, found_board = correct_perspective(image)
    enhanced = enhance_image(corrected)

    before = image_quality(image)
    after = image_quality(enhanced)

    cv2.imwrite(str(output_folder / "original.jpg"), image)
    cv2.imwrite(str(output_folder / "corrected.jpg"), corrected)
    cv2.imwrite(str(output_folder / "enhanced.jpg"), enhanced)

    timeout = 10 if quick else 25

    # Always read the original image so an incorrect board crop
    # cannot become the only OCR input.
    original_result = read_ocr_candidate(
        image, timeout=timeout, psm=11
    )

    if quick:
        enhanced_result = {"text": "", "score": 0.0}
        candidates = [original_result]
    else:
        enhanced_result = read_ocr_candidate(
            enhanced, timeout=timeout, psm=11
        )
        candidates = [original_result, enhanced_result]

        # One additional attempt keeps the OCR time bounded.
        if max(item["score"] for item in candidates) < 80:
            fallback_result = read_ocr_candidate(
                corrected,
                timeout=timeout,
                psm=6,
                threshold=True,
            )
            candidates.append(fallback_result)

    best = max(candidates, key=lambda item: item["score"])
    text = best["text"]

    (output_folder / "notes.txt").write_text(
        text, encoding="utf-8"
    )

    # Preserve the keys expected by the existing Django template.
    return {
        "board_detected": found_board,
        "before": before,
        "after": after,
        "characters": len(text),
        "text": text,
        "ocr_original_characters": len(original_result["text"]),
        "ocr_enhanced_characters": len(enhanced_result["text"]),
    }
