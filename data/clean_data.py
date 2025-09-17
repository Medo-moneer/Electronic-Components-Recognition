import os
import shutil
from pathlib import Path
from PIL import Image

# ---------------- Settings ----------------
RAW_DATA_DIR = "dataset/raw"          # where your raw images are stored
CLEAN_DATA_DIR = "dataset/clean"      # where cleaned data will be saved
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]  # allowed formats
MIN_SIZE = (50, 50)   # minimum width, height
# ------------------------------------------

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

def clean_and_copy():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Raw data folder not found: {RAW_DATA_DIR}")
        return

    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

    total, kept, removed = 0, 0, 0

    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            total += 1
            if not is_image_file(file):
                print(f"⚠️ Skipped non-image: {file}")
                removed += 1
                continue

            src_path = os.path.join(root, file)
            try:
                img = Image.open(src_path)
                img.verify()  # check corruption
                img = Image.open(src_path)  # reopen for size check

                if img.size[0] < MIN_SIZE[0] or img.size[1] < MIN_SIZE[1]:
                    print(f"⚠️ Removed small image: {file} ({img.size})")
                    removed += 1
                    continue

                # create class subfolder inside clean dataset
                rel_path = Path(root).relative_to(RAW_DATA_DIR)
                dst_dir = Path(CLEAN_DATA_DIR) / rel_path
                dst_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy2(src_path, dst_dir / file)
                kept += 1

            except Exception as e:
                print(f"❌ Removed corrupted file: {file}, error: {e}")
                removed += 1

    print(f"\n📊 Cleaning Summary")
    print(f"Total files scanned: {total}")
    print(f"✅ Kept: {kept}")
    print(f"🗑️ Removed: {removed}")

if __name__ == "__main__":
    clean_and_copy()
