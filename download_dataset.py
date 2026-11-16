from datasets import load_dataset

print("Loading PlantVillage dataset...")

dataset = load_dataset("geraldmc/plantvillage-full")

print("\nDataset loaded successfully!")
print(dataset)

for split_name, split_data in dataset.items():
    print(f"\nSplit: {split_name}")
    print(f"Number of images: {len(split_data)}")
    print(f"Columns: {split_data.column_names}")

    if len(split_data) > 0:
        sample = split_data[0]

        print("\nSample record:")
        for key, value in sample.items():
            if key == "image":
                print(f"{key}: {value.size if value else None}")
            else:
                print(f"{key}: {value}")

print("\nDataset inspection completed.")
