from pathlib import Path
import random

import cv2
import yaml
import matplotlib.pyplot as plt

# Project paths 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "road_damage_dataset"

# Load class names 
with open(DATASET_DIR / "data.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

CLASS_NAMES = config["names"]

# Select training images and labels
IMAGE_DIR = DATASET_DIR / "train" / "images"
LABEL_DIR = DATASET_DIR / "train" / "labels"

# Select the first image that has a matching label
image_extensions = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}

image_files = sorted(
    file for file in IMAGE_DIR.iterdir()
    if file.suffix.lower() in image_extensions
    and (LABEL_DIR / f"{file.stem}.txt").exists()
)

# Select 10 random images
random.seed(42)
selected_images = random.sample(image_files, min(10, len(image_files)))

fig, axes = plt.subplots(2, 5, figsize=(22, 10))
axes = axes.flatten()

for index, image_path in enumerate(selected_images):
    image = cv2.imread(str(image_path))

    if image is None:
        axes[index].set_visible(False)
        continue

    image_height, image_width = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    with open(label_path, "r", encoding="utf-8") as file:
        annotations = file.readlines()

    for annotation in annotations:
        values = annotation.strip().split()

        if len(values) != 5:
            continue

        class_id = int(values[0])
        x_center, y_center, box_width, box_height = map(float, values[1:])

        # Convert normalized coordinates to pixel coordinates
        x_center *= image_width
        y_center *= image_height
        box_width *= image_width
        box_height *= image_height

        # Calculate bounding box corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 

        # Add class name
        class_name = CLASS_NAMES[class_id]

        cv2.putText(
            image,
            class_name,
            (x1, max(y1-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Convert BGR image to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axes[index].imshow(image_rgb)
    axes[index].set_title(image_path.name, fontsize = 8)
    axes[index].axis("off")

# Hide the unused sixth subplot
for index in range(len(selected_images), len(axes)):
    axes[index].set_visible(False)
plt.suptitle("Road Damage Dataset - Annotation Samples", fontsize = 16)
plt.tight_layout()
plt.show()

print("Successfully visualized", len(selected_images), "images.")