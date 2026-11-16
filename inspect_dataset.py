from datasets import load_dataset
from collections import Counter

DATASET_NAME = "geraldmc/plantvillage-full"

print("Loading the downloaded PlantVillage dataset...")

dataset = load_dataset(DATASET_NAME)
data = dataset["train"]

print("\n" + "=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

print(f"Total images: {len(data):,}")
print(f"Available columns: {data.column_names}")

# Inspect original split distribution
print("\n" + "=" * 60)
print("2. ORIGINAL SPLIT DISTRIBUTION")
print("=" * 60)

split_counts = Counter(str(value) for value in data["split"])

for split_name, count in sorted(split_counts.items()):
    print(f"{split_name}: {count:,} images")

# Inspect all class names
print("\n" + "=" * 60)
print("3. AVAILABLE TOMATO CLASSES")
print("=" * 60)

class_counts = Counter()

for sample in data:
    label = sample["class_label"]

    if sample["host"] == "Tomato":
        class_counts[label] += 1

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count:,} images")

print(f"\nTotal tomato images: {sum(class_counts.values()):,}")
print(f"Number of tomato classes: {len(class_counts)}")

# Check split distribution for tomato classes
print("\n" + "=" * 60)
print("4. TOMATO CLASS DISTRIBUTION BY ORIGINAL SPLIT")
print("=" * 60)

split_class_counts = Counter()

for sample in data:
    if sample["host"] == "Tomato":
        key = (sample["class_label"], str(sample["split"]))
        split_class_counts[key] += 1

for (class_name, split_name), count in sorted(split_class_counts.items()):
    print(f"{class_name} | {split_name}: {count:,}")

# Inspect a sample image
print("\n" + "=" * 60)
print("5. SAMPLE IMAGE DETAILS")
print("=" * 60)

for sample in data:
    if sample["host"] == "Tomato":
        image = sample["image"]

        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        print(f"Class label: {sample['class_label']}")
        print(f"Original split: {sample['split']}")
        print(f"Leaf ID: {sample['leaf_id']}")
        break

print("\nDataset inspection completed successfully.")
