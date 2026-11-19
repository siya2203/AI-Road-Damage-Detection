
from pathlib import Path
from collections import Counter
import yaml
import matplotlib.pyplot as plt
import numpy as np

# 1. Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "road_damage_dataset"

# 2. Load class names
with open(DATASET_PATH / "data.yaml", "r") as file:
    data = yaml.safe_load(file)

class_names = data["names"]

splits = ["train", "valid", "test"]

# 3. Store annotation counts for every split
split_counts = {}

for split in splits:

    labels_path = DATASET_PATH / split / "labels"

    class_counter = Counter()

    for label_file in labels_path.glob("*.txt"):

        with open(label_file, "r") as file:

            for line in file:

                values = line.strip().split()

                if not values:
                    continue

                class_id = int(values[0])

                if 0 <= class_id < len(class_names):
                    class_counter[class_id] += 1

    split_counts[split] = [
        class_counter.get(i, 0)
        for i in range(len(class_names))
    ]

# 4. Print split-wise statistics
print("\nSPLIT-WISE CLASS DISTRIBUTION")
print("=" * 65)

for split in splits:

    total = sum(split_counts[split])

    print(f"\n{split.upper()} - Total annotations: {total}")

    for i, class_name in enumerate(class_names):

        count = split_counts[split][i]

        percentage = (count / total * 100) if total else 0

        print(f"{class_name}: {count} ({percentage:.2f}%)")

# 5. Prepare grouped bar chart
x = np.arange(len(class_names))
bar_width = 0.25

plt.figure(figsize=(14, 7))

for index, split in enumerate(splits):

    positions = x + (index - 1) * bar_width

    plt.bar(
        positions,
        split_counts[split],
        bar_width,
        label=split.capitalize()
    )

# 6. Format chart
plt.title("Road Damage Class Distribution Across Dataset Splits")
plt.xlabel("Road Damage Classes")
plt.ylabel("Number of Annotations")

plt.xticks(x, class_names, rotation=30, ha="right")

plt.legend()
plt.tight_layout()

# 7. Save chart
output_path = PROJECT_ROOT / "split_class_distribution.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"\nChart saved at: {output_path}")

plt.show()
