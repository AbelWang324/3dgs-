"""Use the installed Windows runtime to start the reconstruction worker."""

import argparse
import os
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parent


def worker_launch(attempt):
    runtime = PROJECT / ".runtime"
    python_dir = runtime / "native-venv" / "Scripts"
    colmap = runtime / "colmap" / "COLMAP-3.9.1-windows-cuda"
    ffmpeg = next((runtime / "ffmpeg").rglob("ffmpeg.exe"), None)
    python = python_dir / "python.exe"
    if not python.is_file() or not (colmap / "bin" / "colmap.exe").is_file() or ffmpeg is None:
        raise RuntimeError("本地重建环境不完整，需要已安装的 Python、COLMAP 和 FFmpeg。")
    if not (ffmpeg.parent / "ffprobe.exe").is_file():
        raise RuntimeError("本地重建环境缺少 ffprobe。")

    env = os.environ.copy()
    search = (python_dir, ffmpeg.parent, colmap / "bin", colmap / "lib")
    env["PATH"] = os.pathsep.join([*map(str, search), env.get("PATH", "")])
    env.update(
        QT_PLUGIN_PATH=str(colmap / "lib" / "plugins"),
        QT_QPA_PLATFORM="offscreen",
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        MPLCONFIGDIR=str(runtime / "matplotlib"),
        TORCH_EXTENSIONS_DIR=str(runtime / "torch-extensions"),
        TORCH_HOME=str(runtime / "torch-cache"),
    )
    command = [
        str(python), "-m", "recon_worker",
        "--request", str(attempt / "request.json"),
        "--work-dir", str(attempt),
    ]
    return command, {"cwd": PROJECT, "env": env}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    args = parser.parse_args()
    command, options = worker_launch(args.attempt.resolve(strict=True))
    return subprocess.run(command, **options, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
