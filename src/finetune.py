import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

# --------------------------------------------------
# Project paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT_DIR / "data" / "processed" / "train"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

ORIGINAL_MODEL_PATH = MODEL_DIR / "plantguard_model.keras"
FINETUNED_MODEL_PATH = MODEL_DIR / "plantguard_model_finetuned.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = 160
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.125
SEED = 42
EPOCHS = 5
LEARNING_RATE = 1e-5
UNFREEZE_LAST_LAYERS = 30


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load existing trained model
    # --------------------------------------------------

    print("Loading original trained model...")

    model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)

    print("Original model loaded.")

    # --------------------------------------------------
    # Load training and validation datasets
    # --------------------------------------------------

    print("\nLoading training dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        saved_class_names = json.load(file)

    if class_names != saved_class_names:
        raise ValueError(
            "Dataset class order does not match the saved class names."
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # --------------------------------------------------
    # Find the pretrained MobileNetV2 backbone
    # --------------------------------------------------

    base_model = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        ),
        None,
    )

    if base_model is None:
        raise ValueError(
            "Could not find MobileNetV2 inside the saved model."
        )

    # --------------------------------------------------
    # Unfreeze only the last selected layers
    # --------------------------------------------------

    base_model.trainable = True

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-UNFREEZE_LAST_LAYERS:]:
        if not isinstance(
            layer, tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True

    trainable_layers = sum(
        1 for layer in base_model.layers if layer.trainable
    )

    print(f"\nTotal MobileNetV2 layers: {len(base_model.layers)}")
    print(f"Trainable MobileNetV2 layers: {trainable_layers}")

    # --------------------------------------------------
    # Recompile using a low learning rate
    # --------------------------------------------------

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nFine-tuning configuration:")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    callbacks = [
        ModelCheckpoint(
            filepath=str(FINETUNED_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # --------------------------------------------------
    # Fine-tune model
    # --------------------------------------------------

    print("\nStarting fine-tuning...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # --------------------------------------------------
    # Save fine-tuning history
    # --------------------------------------------------

    history_path = OUTPUT_DIR / "finetuning_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            file,
            indent=4,
        )

    print("\nFine-tuning completed.")
    print(f"Fine-tuned model: {FINETUNED_MODEL_PATH}")
    print(f"Training history: {history_path}")
    print("\nYour original model has not been overwritten.")


if __name__ == "__main__":
    main()
