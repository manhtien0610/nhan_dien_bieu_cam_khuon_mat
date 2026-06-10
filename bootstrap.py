"""Tự chuyển sang Python trong .venv nếu đang dùng Python global."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def ensure_venv():
    if not VENV_PYTHON.exists():
        return

    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return

    print("Phát hiện Python global (TensorFlow lỗi). Chuyển sang .venv...")
    cmd = [str(VENV_PYTHON), *sys.argv]
    raise SystemExit(subprocess.call(cmd))
