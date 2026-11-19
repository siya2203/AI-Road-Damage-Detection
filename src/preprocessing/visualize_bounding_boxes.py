
from pathlib import Path
from collections import Counter
import random
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "road_damage_dataset"

IMAGE_PATH = DATASET_PATH / "train" / "images"
LABEL_PATH = DATASET_PATH / "train" / "labels"

# 2. Load class names
with open(DATASET_PATH / "data.yaml", "r") as file:
    data = yaml.safe_load(file)

class_names = data["names"]

# 3. Find images with corresponding labels
image_files = []

for image_file in IMAGE_PATH.iterdir():
    if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        label_file = LABEL_PATH / (image_file.stem + ".txt")

        if label_file.exists():
            image_files.append(image_file)

# 4. Select 12 random images
random.seed(42)
samples = random.sample(image_files, min(12, len(image_files)))

# 5. Define colors
colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow"]

# 6. Create a 3 x 4 visualization
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
axes = axes.flatten()

print("\nANNOTATION INSPECTION REPORT")
print("=" * 60)

for index, (ax, image_file) in enumerate(zip(axes, samples), start=1):

    # Load image
    image = Image.open(image_file).convert("RGB")
    img_width, img_height = image.size

    ax.imshow(image)

    # Load corresponding label
    label_file = LABEL_PATH / (image_file.stem + ".txt")

    with open(label_file, "r") as file:
        lines = file.readlines()

    image_class_counts = Counter()
    box_count = 0

    # 7. Read and draw bounding boxes
    for line in lines:

        values = line.strip().split()

        if len(values) != 5:
            continue

        class_id = int(values[0])

        if class_id < 0 or class_id >= len(class_names):
            continue

        x_center, y_center, width, height = map(float, values[1:])

        # Convert normalized coordinates to pixels
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Calculate top-left corner
        x_min = x_center - width / 2
        y_min = y_center - height / 2

        color = colors[class_id]

        rectangle = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )

        ax.add_patch(rectangle)

        # Add class name
        ax.text(
            x_min,
            max(0, y_min - 5),
            class_names[class_id],
            color="white",
            fontsize=7,
            bbox=dict(facecolor=color, alpha=0.8)
        )

        image_class_counts[class_names[class_id]] += 1
        box_count += 1

    # 8. Short image title
    ax.set_title(image_file.stem[:28], fontsize=8)
    ax.axis("off")

    # 9. Print annotation report
    print(f"\nImage {index}: {image_file.name}")
    print(f"Total bounding boxes: {box_count}")

    for class_name, count in image_class_counts.items():
        print(f"  {class_name}: {count}")

# 10. Hide unused axes
for ax in axes[len(samples):]:
    ax.axis("off")

plt.tight_layout()

# 11. Save and display
output_path = PROJECT_ROOT / "bounding_box_samples_12.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("\n" + "=" * 60)
print(f"Visualized {len(samples)} images.")
print(f"Chart saved at: {output_path}")

plt.show()
