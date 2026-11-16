from datasets import load_dataset
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import hashlib

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_NAME = "geraldmc/plantvillage-full"

OUTPUT_DIR = Path("data/processed")
TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"

# --------------------------------------------------
# 2. Create output directories
# --------------------------------------------------

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

print("Loading PlantVillage dataset...")

dataset = load_dataset(DATASET_NAME)
data = dataset["train"]

print(f"Total source images: {len(data):,}")

# --------------------------------------------------
# 3. Filter tomato images
# --------------------------------------------------

tomato_indices = [
    i
    for i, sample in enumerate(data)
    if sample["host"] == "Tomato"
]

print(f"Tomato images found: {len(tomato_indices):,}")

# --------------------------------------------------
# 4. Export images
# --------------------------------------------------

counts = Counter()
failed = []

# Track image hashes to avoid writing duplicate images
seen_hashes = set()

for index in tqdm(tomato_indices, desc="Exporting tomato images"):

    sample = data[index]

    class_name = sample["class_label"]
    original_split = str(sample["split"]).lower()

    # Preserve the dataset's original train/test assignment
    if original_split == "train":
        destination_root = TRAIN_DIR
    elif original_split == "test":
        destination_root = TEST_DIR
    else:
        failed.append((index, f"Unknown split: {original_split}"))
        continue

    image = sample["image"]

    if image is None:
        failed.append((index, "Missing image"))
        continue

    image = image.convert("RGB")

    # Hash the image to detect exact duplicate images
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()

    if image_hash in seen_hashes:
        continue

    seen_hashes.add(image_hash)

    class_dir = destination_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Use the original dataset index for a unique filename
    image_path = class_dir / f"image_{index}.jpg"

    try:
        image.save(image_path, format="JPEG", quality=95)
        counts[(original_split, class_name)] += 1

    except Exception as error:
        failed.append((index, str(error)))

# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET EXPORT SUMMARY")
print("=" * 60)

for split_name in ("train", "test"):

    total = sum(
        count
        for (split, _), count in counts.items()
        if split == split_name
    )

    print(f"\n{split_name.upper()}: {total:,} images")

    for (split, class_name), count in sorted(counts.items()):
        if split == split_name:
            print(f"  {class_name}: {count:,}")

print(f"\nFailed records: {len(failed)}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")

if failed:
    print("\nFirst five failures:")
    for record in failed[:5]:
        print(record)

print("\nDataset preparation completed.")