from pathlib import Path
from collections import Counter

import cv2
import yaml

# Get the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset location
DATASET_DIR = PROJECT_ROOT / "road_damage_dataset"

# Load dataset configuration 
DATA_YAML = DATASET_DIR / "data.yaml"

with open(DATA_YAML, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

CLASS_NAMES = config["names"]

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Overall statistics
overall_class_counts = Counter()

total_images = 0
total_labels = 0
total_missing_labels = 0
total_orphan_labels = 0
total_empty_labels = 0
total_invalid_labels = 0
total_unreadable_images = 0

print("\n========= DATASET INFORMATION ==========")
print("Dataset:", DATASET_DIR.name)
print("Number of classes: ", len(CLASS_NAMES))

for class_id, class_name in enumerate(CLASS_NAMES):
    print(f"{class_id}: {class_name}")

# Process each dataset split

for split in SPLITS:
    print(f"\n========= {split.upper()} DATASET =========")

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print("Image or label directory not found.")
        continue

    images = [
        file for file in image_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS 
    ]

    labels = list(label_dir.glob("*.txt"))

    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    class_counts = Counter()
    empty_labels = 0
    invalid_labels = 0
    unreadable_images = 0
    image_dimensions = Counter()

    # Check image dimensions
    for image in images:
        img = cv2.imread(str(image))

        if img is None:
            unreadable_images += 1
            continue

        height, width = img.shape[:2]
        image_dimensions[(width, height)] += 1

    # Analyse annotation files
    for label_file in labels:
        with open(label_file, "r", encoding = "utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            values = line.split()

            if len(values) != 5:
                invalid_labels += 1
                continue

            try:
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]

                if not 0 <= class_id < len(CLASS_NAMES):
                    invalid_labels += 1
                    continue

                if not all(0 <= value <=1 for value in coordinates):
                    invalid_labels += 1
                    continue

                x, y, width, height = coordinates

                if width <= 0 or height <= 0:
                    invalid_labels += 1
                    continue
                # Check bounding box boundaries
                if(
                    x - width / 2 < 0
                    or y - height / 2 < 0
                    or x + width / 2 > 1
                    or y + height / 2 > 1
                ):
                    invalid_labels += 1
                    continue

                class_counts[class_id] += 1

            except ValueError:
                invalid_labels += 1

    print("Images: ", len(images))
    print("Label files: ", len(labels))
    print("Images without label files: ", len(missing_labels))
    print("Labels without images: ", len(orphan_labels))
    print("Empty label files: ", empty_labels)
    print("Invalid annotations: ", invalid_labels)
    print("Unreadable images: ", unreadable_images)
    print("\n Image dimensions:")
    for dimensions, count in image_dimensions.most_common(5):
        print(f"{dimensions[0]} x {dimensions[1]}: {count} images")

    print("\nClass distribution:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {class_counts[class_id]}")

    total_images += len(images)
    total_labels += len(labels)
    total_missing_labels += len(missing_labels)
    total_orphan_labels += len(orphan_labels)
    total_empty_labels += empty_labels
    total_invalid_labels += invalid_labels
    total_unreadable_images += unreadable_images

    overall_class_counts.update(class_counts)

print("\n========= FINAL SUMMARY =========")
print("Total images: ", total_images)
print("Total label files: ", total_labels)
print("Images without labels:", total_missing_labels)
print("Labels without images:", total_orphan_labels)
print("Empty label files:", total_empty_labels)
print("Invalid annotations:", total_invalid_labels)
print("Unreadable images:", total_unreadable_images)

print("\nOverall class distribution: ")
for class_id, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name}: {overall_class_counts[class_id]}")

print("\nTotal annotations: ", sum(overall_class_counts.values()))
print("\nDataset analysis completed!")
