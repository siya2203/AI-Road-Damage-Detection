from pathlib import Path 
from collections import Counter
import yaml
import matplotlib.pyplot as plt

# 1. Define project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "road_damage_dataset"

# 2. Load class names from data.yaml
yaml_path = DATASET_PATH / "data.yaml"

with open(yaml_path, "r") as file:
    data = yaml.safe_load(file)

class_names = data["names"]

# 3. Create a counter for all classes 
class_counts = Counter()

# 4. Read lables from all dataset splits
for split in ["train", "valid", "test"]:

    labels_path = DATASET_PATH / split / "labels"

    for label_file in labels_path.glob("*.txt"):

        with open(label_file, "r") as file:
            for line in file:

                values = line.strip().split()

                if not values:
                    continue

                class_id = int(values[0])
                class_counts[class_id]+=1

# 5. Prepare chart data
counts = [class_counts.get(i, 0) for i in range(len(class_names))]

# 6. Print class-wise counts
print("\nClass Distribution:")
for i, name in enumerate(class_names):
    print(f"{name}: {counts[i]}")

# 7. Create bar chart
plt.figure(figsize=(12, 6))

bars = plt.bar(class_names, counts)

plt.title("Road Damage Dataset - Class Distribution")
plt.xlabel("Road Damage Classes")
plt.ylabel("Number of Annotations")
plt.xticks(rotation = 30, ha = "right")

# Display values above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height, 
        str(int(height)),
        ha = "center",
        va = "bottom"
    )

plt.tight_layout()

# 8. Save and display chart
output_path = PROJECT_ROOT / "class_distribution.png"
plt.savefig(output_path, dpi = 300)
print(f"\nChart saved at: {output_path}")

plt.show()