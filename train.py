"""
Bước 3: Train model nhận diện biểu cảm với Transfer Learning (MobileNetV2)
Dataset: archive/train  |  Validation: archive/test

Chạy nhanh (test pipeline):  python train.py --quick
Train đầy đủ:               python train.py
"""

import argparse
import sys
from pathlib import Path

import bootstrap

bootstrap.ensure_venv()

try:
    import tensorflow as tf
    import keras
    from keras import layers
    from keras.applications import MobileNetV2
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except ImportError as exc:
    print("=" * 50)
    print("LỖI: Không import được TensorFlow!")
    print("=" * 50)
    print(f"Chi tiết: {exc}\n")
    print("Nguyên nhân: Bạn đang dùng Python global (bị lỗi), không phải .venv")
    print("\nCách sửa — chạy trong PowerShell:")
    print('  cd "D:\\nhan dien bieu cam mat qua cam"')
    print("  .\\.venv\\Scripts\\Activate.ps1")
    print("  python train.py")
    print("\nHoặc chạy trực tiếp:")
    print("  .\\.venv\\Scripts\\python.exe train.py")
    sys.exit(1)

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EPOCHS,
    IMG_SIZE,
    MODEL_DIR,
    MODEL_PATH,
    TEST_DIR,
    TRAIN_DIR,
)


def build_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def make_datasets(quick=False):
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy {TRAIN_DIR}")

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42,
        subset="training" if quick else None,
        validation_split=0.1 if quick else None,
    )

    if quick:
        val_ds = keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            labels="inferred",
            label_mode="int",
            class_names=CLASS_NAMES,
            color_mode="rgb",
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            shuffle=False,
            seed=42,
            subset="validation",
            validation_split=0.1,
        )
    else:
        val_ds = keras.utils.image_dataset_from_directory(
            TEST_DIR,
            labels="inferred",
            label_mode="int",
            class_names=CLASS_NAMES,
            color_mode="rgb",
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            shuffle=False,
        )

    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomContrast(0.1),
        ]
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=autotune,
    )
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds


def compute_class_weights(train_dir: Path):
    counts = []
    for name in CLASS_NAMES:
        folder = train_dir / name
        counts.append(len(list(folder.glob("*"))))

    total = sum(counts)
    weights = {i: total / (len(CLASS_NAMES) * c) for i, c in enumerate(counts)}
    return weights, dict(zip(CLASS_NAMES, counts))


def fine_tune(model, base, train_ds, val_ds, class_weight):
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Train emotion model on FER2013")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Train nhanh trên 10%% data (kiểm tra pipeline)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("BƯỚC 3: TRAIN MODEL (MobileNetV2 + FER2013)")
    print("=" * 50)

    class_weight, counts = compute_class_weights(TRAIN_DIR)
    print("Số ảnh mỗi lớp (train):", counts)

    train_ds, val_ds = make_datasets(quick=args.quick)
    model, base = build_model()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    print(f"\nPhase 1: Train top layers ({args.epochs} epochs max)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    if not args.quick:
        print("\nPhase 2: Fine-tune (8 epochs max)...")
        fine_tune(model, base, train_ds, val_ds, class_weight)

    model.save(MODEL_PATH)
    best_acc = max(history.history["val_accuracy"])
    print(f"\nHoàn tất! Model lưu tại: {MODEL_PATH}")
    print(f"Val accuracy (phase 1 best): {best_acc:.2%}")
    print("Chạy tiếp: python evaluate.py  →  python main.py")


if __name__ == "__main__":
    main()
