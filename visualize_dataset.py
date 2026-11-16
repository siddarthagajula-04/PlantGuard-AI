from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_DIR = Path("data/processed/train")

print("Checking training dataset...")

class_counts = Counter()

for class_dir in sorted(TRAIN_DIR.iterdir()):

    if class_dir.is_dir():

        images = [
            path for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        class_counts[class_dir.name] = len(images)

print("\nImages per class:")

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

print(f"\nTotal training images: {sum(class_counts.values()):,}")

# -----------------------------------------
# Plot class distribution
# -----------------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    [name.replace("Tomato___", "") for name in class_counts],
    class_counts.values()
)

plt.title("Tomato Leaf Disease Dataset: Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()

Path("outputs").mkdir(exist_ok=True)

plt.savefig(
    "outputs/class_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# -----------------------------------------
# Display sample images
# -----------------------------------------

available_classes = [
    name for name, count in class_counts.items()
    if count > 0
]

number_to_show = min(6, len(available_classes))

fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 8)
)

axes = axes.flatten()

for i, class_name in enumerate(available_classes[:number_to_show]):

    class_dir = TRAIN_DIR / class_name

    image_paths = list(class_dir.glob("*.jpg"))

    if not image_paths:
        continue

    image = Image.open(image_paths[0]).convert("RGB")

    axes[i].imshow(image)
    axes[i].set_title(class_name.replace("Tomato___", ""))
    axes[i].axis("off")

for i in range(number_to_show, len(axes)):
    axes[i].axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/sample_images.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nVisualization completed.")
print("Saved charts to the outputs folder.")