# Nhận diện biểu cảm khuôn mặt qua Webcam

Dự án AI nhận diện 7 biểu cảm (vui, buồn, giận, sợ, ghê tởm, ngạc nhiên, bình thường) realtime qua webcam, sử dụng **Transfer Learning (MobileNetV2)** trên dataset **FER2013**.

## Cấu trúc dự án

```
├── archive/          # Dataset FER2013 (train/ + test/) — tải riêng
├── config.py         # Cấu hình chung
├── train.py          # Bước 3: Train model
├── evaluate.py       # Bước 5: Đánh giá + confusion matrix
├── main.py           # Bước 4: Webcam realtime
├── models/           # Model sau khi train
├── results/          # Báo cáo đánh giá
└── requirements.txt
```

## Yêu cầu

- Python 3.10 – 3.12
- Webcam
- Windows (font tiếng Việt dùng Segoe UI)

## Cài đặt

```powershell
cd "nhan-dien-bieu-cam-mat-qua-cam"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset FER2013

1. Tải từ [Kaggle – FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Giải nén vào thư mục `archive/` với cấu trúc:

```
archive/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── (cùng 7 thư mục)
```

## Chạy dự án

```powershell
# Luôn bật .venv trước
.\.venv\Scripts\Activate.ps1

# 1. Train model (~1–3 giờ trên CPU)
python train.py

# 2. Đánh giá (accuracy + confusion matrix)
python evaluate.py

# 3. Nhận diện qua webcam (nhấn 'q' để thoát)
python main.py
```

Hoặc double-click: `run_train.bat`, `run_evaluate.bat`, `run_main.bat`

### Train nhanh (kiểm tra pipeline)

```powershell
python train.py --quick --epochs 2
```

## Quy trình 5 bước

| Bước | Mô tả | File |
|------|--------|------|
| 1 | Chuẩn bị dataset FER2013 | `archive/` |
| 2 | Phát hiện khuôn mặt (Haar Cascade) | `main.py` |
| 3 | Train model MobileNetV2 | `train.py` |
| 4 | Ứng dụng realtime webcam | `main.py` |
| 5 | Đánh giá & tối ưu | `evaluate.py` |

## Kết quả đánh giá

Sau `python evaluate.py`:

- `results/classification_report.txt` — precision, recall, F1
- `results/confusion_matrix.png` — biểu đồ nhầm lẫn (dùng trong báo cáo)

## Lưu ý

- **Luôn chạy trong `.venv`**, không dùng Python global.
- Dataset và model **không** nằm trong Git (quá nặng) — xem `.gitignore`.
- Sau khi clone repo: cài đặt → tải dataset → `train.py` → `main.py`.

## Công nghệ

- OpenCV — webcam, phát hiện mặt
- TensorFlow / Keras — MobileNetV2 transfer learning
- Pillow — hiển thị tiếng Việt có dấu
- scikit-learn — confusion matrix
