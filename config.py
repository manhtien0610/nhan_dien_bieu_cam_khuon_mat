from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "archive"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = MODEL_DIR / "emotion_mobilenet.keras"

IMG_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 20

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

EMOTION_VI = {
    "angry": "Giận",
    "disgust": "Ghê tởm",
    "fear": "Sợ",
    "happy": "Vui",
    "sad": "Buồn",
    "surprise": "Ngạc nhiên",
    "neutral": "Bình thường",
}


FONT_PATH = "C:/Windows/Fonts/segoeui.ttf"
SMOOTH_FRAMES = 15
SUBTLE_EMOTIONS = ("sad", "fear", "disgust", "angry")
