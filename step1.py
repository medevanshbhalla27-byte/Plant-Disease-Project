"""Step 1: Explore the PlantVillage folder structure and sample images."""
from collections import Counter
from PIL import Image
from config import DATASET_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}\n"
            "Edit DATASET_DIR in config.py or set PLANTVILLAGE_DIR."
        )

    class_folders = sorted(
        p for p in DATASET_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not class_folders:
        raise RuntimeError(f"No class folders found inside {DATASET_DIR}")

    print(f"\nDataset: {DATASET_DIR}")
    print(f"Number of classes: {len(class_folders)}\n")

    total_images = 0
    for folder in class_folders:
        images = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        total_images += len(images)
        print(f"{folder.name}: {len(images)} images")

    print(f"\nTotal images: {total_images}")

    first_images = [
        p for p in class_folders[0].iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if first_images:
        sample_path = first_images[0]
        with Image.open(sample_path) as image:
            print(f"\nSample image: {sample_path.name}")
            print(f"Original image size: {image.size}")
            print(f"Image mode: {image.mode}")

if __name__ == "__main__":
    main()
