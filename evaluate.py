"""
Bước 5: Đánh giá model trên tập test + vẽ confusion matrix

Chạy: python evaluate.py  hoặc  run_evaluate.bat
"""

import bootstrap

bootstrap.ensure_venv()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES, EMOTION_VI, MODEL_PATH, RESULTS_DIR, TEST_DIR


def load_test_data():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=64,
        image_size=(96, 96),
        shuffle=False,
    )
    return test_ds


def main():
    if not MODEL_PATH.exists():
        print(f"Chưa có model! Chạy trước: python train.py")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("BƯỚC 5: ĐÁNH GIÁ MODEL")
    print("=" * 50)

    model = tf.keras.models.load_model(MODEL_PATH)
    test_ds = load_test_data()

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred)

    labels_vi = [EMOTION_VI[c] for c in CLASS_NAMES]
    report = classification_report(y_true, y_pred, target_names=labels_vi)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy trên tập test: {accuracy:.2%}\n")
    print(report)

    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text(
        f"Accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels_vi)),
        yticks=range(len(labels_vi)),
        xticklabels=labels_vi,
        yticklabels=labels_vi,
        ylabel="Nhãn thật",
        xlabel="Dự đoán",
        title=f"Confusion Matrix (Accuracy: {accuracy:.1%})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    chart_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nĐã lưu báo cáo: {report_path}")
    print(f"Đã lưu biểu đồ: {chart_path}")


if __name__ == "__main__":
    main()
