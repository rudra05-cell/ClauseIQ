"""Downloads and extracts the CUAD dataset."""

import urllib.request
import zipfile
import os

URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "cuad.zip")
EXTRACT_DIR = os.path.join(RAW_DIR, "cuad_extracted")

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    urllib.request.urlretrieve(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print(f"Dataset ready at {EXTRACT_DIR}/CUADv1.json")
