import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT_DIR / "data" / "processed" / "test"
MODEL_PATH = ROOT_DIR / "models" / "plantguard_model.keras"
CLASS_NAMES_PATH = ROOT_DIR / "models" / "class_names.json"
OUTPUT_DIR = ROOT_DIR / "outputs"

IMG_SIZE = 160
BATCH_SIZE = 16


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    print("Loading test dataset...")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    # Confirm that dataset class ordering matches the trained model.
    if test_ds.class_names != class_names:
        raise ValueError(
            "Test dataset class order does not match the model's class order."
        )

    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    print("\nEvaluating model on unseen test images...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    true_labels = []
    predicted_labels = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        true_labels.extend(labels.numpy())
        predicted_labels.extend(predictions)

    print("\nClassification Report:")
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    with open(
        OUTPUT_DIR / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
    )

    print("\nTop Classification Errors:")

    error_pairs = []

    for actual_idx in range(len(class_names)):
        for predicted_idx in range(len(class_names)):
            if actual_idx != predicted_idx:
                error_count = int(cm[actual_idx, predicted_idx])

                if error_count > 0:
                    error_pairs.append(
                        (
                            error_count,
                            class_names[actual_idx],
                            class_names[predicted_idx],
                        )
                    )

    error_pairs.sort(reverse=True)

    for count, actual, predicted in error_pairs[:15]:
        actual_name = actual.replace("Tomato___", "").replace("_", " ")
        predicted_name = predicted.replace("Tomato___", "").replace("_", " ")

        print(
            f"Actual: {actual_name} | "
            f"Predicted: {predicted_name} | "
            f"Errors: {count}"
        )

    display_names = [
        name.replace("Tomato___", "").replace("_", " ")
        for name in class_names
    ]

    fig, ax = plt.subplots(figsize=(14, 12))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_names,
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False,
    )

    plt.title("PlantGuard AI - Confusion Matrix")
    plt.tight_layout()

    confusion_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "number_of_test_images": len(true_labels),
        "number_of_classes": len(class_names),
    }

    with open(
        OUTPUT_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nEvaluation completed successfully.")
    print(f"Confusion matrix: {confusion_path}")
    print(f"Classification report: {OUTPUT_DIR / 'classification_report.json'}")
    print(f"Evaluation summary: {OUTPUT_DIR / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
