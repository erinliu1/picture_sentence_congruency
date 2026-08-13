from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.environ["QUALTRICS_API_TOKEN"]
DATACENTER = os.environ["QUALTRICS_DATACENTER"]

PICTURES_DIR = Path("/Intern/Erin/picture_sentence_congruency/stimuli/pictures")
OUTPUT_PATH = Path("/Intern/Erin/picture_sentence_congruency/survey/qualtrics_image_mapping.json")

# Paste your Qualtrics library ID here after running list_libraries().
LIBRARY_ID = "UR_6KACh5YeBYAQwaq"

# Optional folder name inside your Qualtrics Graphics Library.
QUALTRICS_FOLDER = "Picture-Sentence Stimuli"

REQUEST_DELAY_SECONDS = 0.25
MAX_ATTEMPTS = 4

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif"}


def api_base_url() -> str:
    datacenter = DATACENTER.strip().removeprefix("https://").rstrip("/")

    # Supports either:
    #   iad1
    #   iad1.qualtrics.com
    if "." not in datacenter:
        datacenter = f"{datacenter}.qualtrics.com"

    return f"https://{datacenter}/API/v3"


def headers() -> dict[str, str]:
    return {
        "X-API-TOKEN": API_TOKEN,
        "Accept": "application/json",
    }


def request_with_retry(
    method: str,
    url: str,
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.request(
                method,
                url,
                headers=headers(),
                timeout=60,
                **kwargs,
            )

            if response.ok:
                return response

            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text}"
            )

        except (requests.RequestException, RuntimeError) as error:
            last_error = error

            if attempt == MAX_ATTEMPTS:
                break

            print(
                f"    Attempt {attempt}/{MAX_ATTEMPTS} failed: {error}"
            )
            time.sleep(2)

    raise RuntimeError(
        f"All {MAX_ATTEMPTS} attempts failed. Last error: {last_error}"
    )


def list_libraries() -> None:
    """Print libraries available to the current Qualtrics account."""
    url = f"{api_base_url()}/libraries"
    response = request_with_retry("GET", url)
    data = response.json()

    libraries = data.get("result", {}).get("elements", [])

    if not libraries:
        print(json.dumps(data, indent=2))
        raise RuntimeError("No libraries were returned.")

    print("\nAvailable Qualtrics libraries:\n")

    for library in libraries:
        library_id = library.get("libraryId") or library.get("id")
        name = library.get("name", "(unnamed)")
        print(f"{name}: {library_id}")


def load_existing_mapping() -> dict[str, dict[str, str]]:
    if not OUTPUT_PATH.exists():
        return {}

    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_mapping(mapping: dict[str, dict[str, str]]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2, ensure_ascii=False)
        file.write("\n")


def get_image_paths() -> list[Path]:
    if not PICTURES_DIR.exists():
        raise FileNotFoundError(
            f"Pictures directory not found: {PICTURES_DIR.resolve()}"
        )

    paths = sorted(
        path
        for path in PICTURES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not paths:
        raise RuntimeError(
            f"No supported images found in {PICTURES_DIR.resolve()}"
        )

    return paths


def upload_image(image_path: Path) -> dict[str, str]:
    url = f"{api_base_url()}/libraries/{LIBRARY_ID}/graphics"

    mime_type = (
        mimetypes.guess_type(image_path.name)[0]
        or "application/octet-stream"
    )

    with image_path.open("rb") as image_file:
        files = {
            "file": (
                image_path.name,
                image_file,
                mime_type,
            ),
            # The (None, value) form makes this an ordinary
            # multipart form field rather than another file.
            "folder": (
                None,
                QUALTRICS_FOLDER,
            ),
        }

        response = request_with_retry(
            "POST",
            url,
            files=files,
        )

    data = response.json()
    result = data.get("result", {})

    graphic_id = (
        result.get("id")
        or result.get("graphicId")
        or result.get("imageId")
    )

    graphic_url = (
        result.get("url")
        or result.get("graphicUrl")
        or result.get("imageUrl")
    )

    if not graphic_id:
        raise RuntimeError(
            "Upload succeeded, but no graphic ID was found in the response:\n"
            + json.dumps(data, indent=2)
        )

    return {
        "graphic_id": graphic_id,
        "url": graphic_url or "",
    }


def upload_all_images() -> None:
    if not LIBRARY_ID:
        raise RuntimeError(
            "LIBRARY_ID is empty. Run list_libraries() first, then paste "
            "your personal library ID into LIBRARY_ID."
        )
    image_paths = get_image_paths()
    mapping = load_existing_mapping()

    print(f"Found {len(image_paths)} images.")

    for index, image_path in enumerate(image_paths, start=1):
        filename = image_path.name

        if filename in mapping:
            print(
                f"[{index}/{len(image_paths)}] Skipping {filename}: "
                "already in mapping."
            )
            continue

        print(f"[{index}/{len(image_paths)}] Uploading {filename}...")

        try:
            uploaded = upload_image(image_path)
        except Exception as error:
            print(f"    Failed: {error}")
            continue

        mapping[filename] = uploaded
        save_mapping(mapping)

        print(f"    Uploaded as {uploaded['graphic_id']}")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nSaved mapping to: {OUTPUT_PATH.resolve()}")


def main() -> None:
    if not LIBRARY_ID:
        list_libraries()
        print(
            "\nCopy the ID for your personal library into LIBRARY_ID, "
            "then run the script again."
        )
        return

    upload_all_images()


if __name__ == "__main__":
    main()