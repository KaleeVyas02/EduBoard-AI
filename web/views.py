
import sys
import uuid
import traceback
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render
from django.views.decorators.http import require_GET

from PIL import Image, UnidentifiedImageError, ImageOps

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline import process_image

OUTPUT = BASE_DIR / "web" / "uploads"
OUTPUT.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_SIZE = 12 * 1024 * 1024  # 12 MB


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def valid_job(job):
    """Validate a generated job ID."""
    return (
        isinstance(job, str)
        and len(job) == 32
        and all(c in "0123456789abcdef" for c in job)
    )


def save_uploaded_image(photo, folder):
    """
    Validate and save an uploaded image.
    Convert it to RGB JPEG so PNG uploads are handled safely.
    """
    original_path = folder / "original_upload"

    # Save the uploaded bytes temporarily.
    with original_path.open("wb") as file:
        for chunk in photo.chunks():
            file.write(chunk)

    # Verify that the uploaded file is a real image.
    with Image.open(original_path) as image:
        image.verify()

    # Reopen after verify(), apply orientation, and normalize.
    with Image.open(original_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        input_path = folder / "input.jpg"
        image.save(input_path, format="JPEG", quality=95)

    # Remove the temporary upload.
    original_path.unlink(missing_ok=True)

    return input_path


def get_error_message(error):
    """Return a readable message for common processing errors."""
    message = str(error).strip()

    if isinstance(error, UnidentifiedImageError):
        return "The uploaded file is not a readable image. Please try another JPG or PNG."

    if isinstance(error, FileNotFoundError):
        return (
            "A required file or OCR program could not be found. "
            "Check the VS Code terminal for details."
        )

    if isinstance(error, ImportError):
        return (
            "A required Python library is missing. "
            "Install the project requirements in your active virtual environment."
        )

    if not message:
        message = "No additional error details were provided."

    return f"Processing failed ({type(error).__name__}): {message}"


# --------------------------------------------------
# HOME / IMAGE UPLOAD
# --------------------------------------------------

def home(request):
    context = {}

    if request.method != "POST":
        return render(request, "index.html", context)

    photo = request.FILES.get("photo")

    if not photo:
        context["error"] = "Please choose a classroom image first."
        return render(request, "index.html", context)

    if not photo.name.lower().endswith((".jpg", ".jpeg", ".png")):
        context["error"] = "Please upload a JPG, JPEG, or PNG image."
        return render(request, "index.html", context)

    if photo.size == 0:
        context["error"] = "The uploaded file is empty. Please choose another image."
        return render(request, "index.html", context)

    if photo.size > MAX_IMAGE_SIZE:
        context["error"] = "Image must be smaller than 12 MB."
        return render(request, "index.html", context)

    job = uuid.uuid4().hex
    folder = OUTPUT / job
    folder.mkdir(parents=True, exist_ok=False)

    try:
        # Step 1: Save and normalize the image.
        input_path = save_uploaded_image(photo, folder)

        # Step 2: Run the existing image-processing and OCR pipeline.
        result = process_image(input_path, folder)

        # Step 3: Make the results available to the template.
        context["result"] = result
        context["job"] = job

        # Show a message if the pipeline returned no extracted text.
        if isinstance(result, dict):
            extracted_text = (
                result.get("text")
                or result.get("extracted_text")
                or result.get("notes")
                or ""
            )

            if not str(extracted_text).strip():
                context["warning"] = (
                    "The image was processed, but no readable text was returned. "
                    "Try a clearer image with larger writing."
                )

    except Exception as error:
        print("\n" + "=" * 60)
        print("EDUBOARD AI: IMAGE PROCESSING ERROR")
        print(f"Job ID: {job}")
        traceback.print_exc()
        print("=" * 60 + "\n")

        context["error"] = get_error_message(error)

    return render(request, "index.html", context)


# --------------------------------------------------
# DOWNLOAD EXTRACTED NOTES
# --------------------------------------------------

@require_GET
def download_notes(request, job):
    if not valid_job(job):
        raise Http404("Invalid job ID.")

    notes_file = OUTPUT / job / "notes.txt"

    if not notes_file.is_file():
        raise Http404("Notes file not found. Process an image first.")

    return FileResponse(
        notes_file.open("rb"),
        as_attachment=True,
        filename="eduboard_notes.txt",
        content_type="text/plain; charset=utf-8",
    )


# --------------------------------------------------
# VIEW PROCESSED IMAGES
# --------------------------------------------------

@require_GET
def result_image(request, job, kind):
    if not valid_job(job):
        raise Http404("Invalid job ID.")

    if kind not in ("original", "corrected", "enhanced"):
        raise Http404("Invalid image type.")

    image_file = OUTPUT / job / f"{kind}.jpg"

    if not image_file.is_file():
        raise Http404("Processed image not found.")

    return FileResponse(
        image_file.open("rb"),
        content_type="image/jpeg",
    )
