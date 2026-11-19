"""Run one static 3DGS reconstruction in an isolated attempt directory."""

import argparse
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic


RULES_VERSION = "static3dgs-v1"
WORKER_VERSION = "1.0.0"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}
MAX_INPUT_BYTES = 2 * 1024**3
MAX_PHOTO_BYTES = 40 * 1024**2
MAX_PHOTO_PIXELS = 32_000_000
MIN_PHOTOS = 12
MAX_PHOTOS = 300
MAX_LONG_EDGE = 1600
VIDEO_FRAME_TARGET = 120
TRAINING_STEPS = {"standard": 30_000, "smoke": 3_000}
ERROR_CODES = {
    "INPUT_INVALID": 2,
    "WORK_DIR_IN_USE": 3,
    "TOOL_UNAVAILABLE": 4,
    "SFM_FAILED": 5,
    "SFM_INSUFFICIENT_REGISTRATION": 5,
    "TRAINING_FAILED": 6,
    "GPU_OOM": 6,
    "EXPORT_INVALID": 7,
    "INTERNAL_ERROR": 1,
}


class WorkerError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def publish_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_request(path, root):
    try:
        request = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object.")
        for key in ("job_id", "attempt_id", "input_kind", "input_path", "rules_version"):
            if not isinstance(request.get(key), str) or not request[key].strip():
                raise ValueError(f"{key} must be a nonempty string.")
        if request["rules_version"] != RULES_VERSION:
            raise ValueError(f"rules_version must be {RULES_VERSION}.")
        if request["input_kind"] not in ("images", "video"):
            raise ValueError("input_kind must be images or video.")
        profile = request.get("quality_profile", "standard")
        if not isinstance(profile, str) or profile not in TRAINING_STEPS:
            raise ValueError("quality_profile must be standard or smoke.")
        source = Path(request["input_path"])
        if source.is_absolute() or source.drive or ".." in source.parts:
            raise ValueError("input_path must be relative to the attempt input directory.")
        input_root = (root / "input").resolve(strict=True)
        source = (root / source).resolve(strict=True)
        if not input_root.is_relative_to(root) or not source.is_relative_to(input_root):
            raise ValueError("input_path must resolve within the attempt input directory.")
    except OSError as exc:
        raise WorkerError("INPUT_INVALID", "Could not read the request or its input path.") from exc
    except (ValueError, RuntimeError) as exc:
        raise WorkerError("INPUT_INVALID", str(exc)) from exc
    return request, source, profile


def event(status, stage, ids, **details):
    print(json.dumps({"status": status, "stage": stage, **ids, **details}), flush=True)


def run_tool(command, log_path, error_code, message, cwd, capture=False):
    executable = shutil.which(command[0])
    if executable is None:
        raise WorkerError("TOOL_UNAVAILABLE", f"Required command not found: {command[0]}.")
    with log_path.open("wb") as log:
        try:
            result = subprocess.run(
                [executable, *command[1:]],
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE if capture else log,
                stderr=log if capture else subprocess.STDOUT,
                check=False,
            )
        except OSError as exc:
            raise WorkerError("TOOL_UNAVAILABLE", f"Could not launch {command[0]}.") from exc
    if result.returncode:
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 65536))
            tail = log.read().lower()
        if error_code == "TRAINING_FAILED" and any(
            word in tail for word in (b"cuda out of memory", b"cuda error: out of memory")
        ):
            raise WorkerError("GPU_OOM", "The training job exceeded available GPU memory.")
        raise WorkerError(error_code, message)
    return result.stdout


def prepare_photos(source, frames, input_root):
    from PIL import Image, ImageOps

    if not source.is_dir():
        raise WorkerError("INPUT_INVALID", "The images input must be a directory.")
    photos = sorted(
        path for path in source.iterdir() if path.suffix.lower() in PHOTO_EXTENSIONS
    )
    if not MIN_PHOTOS <= len(photos) <= MAX_PHOTOS:
        raise WorkerError("INPUT_INVALID", f"Provide {MIN_PHOTOS}–{MAX_PHOTOS} JPG or PNG images.")
    total_bytes = 0
    dimensions = None
    for index, path in enumerate(photos, 1):
        if not path.is_file() or not path.resolve().is_relative_to(input_root):
            raise WorkerError("INPUT_INVALID", f"Invalid image entry at index {index}.")
        size = path.stat().st_size
        total_bytes += size
        if not 0 < size <= MAX_PHOTO_BYTES or total_bytes > MAX_INPUT_BYTES:
            raise WorkerError("INPUT_INVALID", "Image file size or total upload size exceeds the limit.")
        try:
            with Image.open(path) as image:
                if image.format not in ("JPEG", "PNG") or image.width * image.height > MAX_PHOTO_PIXELS:
                    raise WorkerError("INPUT_INVALID", f"Invalid format or dimensions at image {index}.")
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                normalized.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.Resampling.LANCZOS)
                if dimensions is None:
                    dimensions = normalized.size
                if normalized.size != dimensions:
                    raise WorkerError("INPUT_INVALID", "Photos must have consistent dimensions after EXIF rotation and resizing.")
                normalized.save(frames / f"{index:06d}.jpg", quality=95)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise WorkerError("INPUT_INVALID", f"Could not decode image {index}.") from exc
    return len(photos)


def prepare_video(source, frames, logs, root):
    if not source.is_file() or source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise WorkerError("INPUT_INVALID", "Video must be an MP4 or MOV file.")
    if not 0 < source.stat().st_size <= MAX_INPUT_BYTES:
        raise WorkerError("INPUT_INVALID", "Video file size exceeds the limit or the file is empty.")
    probe = run_tool(
        ["ffprobe", "-v", "error", "-select_streams", "V:0",
         "-show_entries", "stream=duration,width,height:format=duration,format_name",
         "-of", "json", str(source)],
        logs / "ffprobe.log", "INPUT_INVALID", "Could not read video metadata.", root, capture=True,
    )
    try:
        metadata = json.loads(probe)
        stream = metadata["streams"][0]
        container = metadata["format"]
        duration = float(stream.get("duration") or container["duration"])
        width, height = stream["width"], stream["height"]
        if not {"mov", "mp4"}.intersection(container["format_name"].split(",")):
            raise ValueError("Unexpected container format.")
        if not 3 <= duration <= 180 or min(width, height) < 2 or max(width, height) > 4096:
            raise ValueError("Invalid duration or resolution.")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise WorkerError("INPUT_INVALID", "Video must be 3–180 seconds and at most 4096 pixels per side.") from exc
    frame_rate = min(6.0, VIDEO_FRAME_TARGET / duration)
    scale = (
        f"scale=w='min(iw,{MAX_LONG_EDGE})':h='min(ih,{MAX_LONG_EDGE})'"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    run_tool(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map", "0:V:0",
         "-vf", f"fps={frame_rate:.8f},{scale},setsar=1", "-frames:v", str(VIDEO_FRAME_TARGET),
         "-q:v", "2", str(frames / "%06d.jpg")],
        logs / "ffmpeg.log", "INPUT_INVALID", "Could not decode video frames.", root,
    )
    return 1


def run_sfm(dataset, image_dir, matching, logs, root):
    colmap = dataset / "colmap"
    sparse = colmap / "sparse"
    sparse.mkdir(parents=True)
    database = str(colmap / "database.db")
    steps = (
        ("features", ["feature_extractor", "--database_path", database,
                      "--image_path", str(image_dir), "--ImageReader.single_camera", "1",
                      "--ImageReader.camera_model", "OPENCV"]),
        ("matching", [f"{matching}_matcher", "--database_path", database]),
        ("mapping", ["mapper", "--database_path", database, "--image_path", str(image_dir),
                     "--output_path", str(sparse)]),
    )
    for name, options in steps:
        run_tool(["colmap", *options], logs / f"sfm-{name}.log", "SFM_FAILED",
                 f"COLMAP {name} failed.", root)

    models = []
    for model in sorted(sparse.iterdir()):
        if all((model / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
            with (model / "images.bin").open("rb") as file:
                models.append((int.from_bytes(file.read(8), "little"), model))
    if not models:
        raise WorkerError("SFM_INSUFFICIENT_REGISTRATION", "COLMAP did not produce a camera model.")
    _, model = max(models, key=lambda item: item[0])
    conversion = (
        "import sys; from pathlib import Path; "
        "from nerfstudio.process_data.colmap_utils import colmap_to_json; "
        "colmap_to_json(Path(sys.argv[1]), Path(sys.argv[2]))"
    )
    run_tool(
        [sys.executable, "-c", conversion, str(model), str(dataset)],
        logs / "sfm-conversion.log", "SFM_FAILED", "Could not convert the COLMAP camera model.", root,
    )
    return model


def registered_views(dataset, model, submitted_views):
    try:
        transforms = json.loads((dataset / "transforms.json").read_text(encoding="utf-8"))
        count = len(transforms["frames"])
        points = dataset / transforms["ply_file_path"]
        # A COLMAP points3D.bin starts with an unsigned 64-bit point count.
        with (model / "points3D.bin").open("rb") as handle:
            point_count = int.from_bytes(handle.read(8), "little")
        if not points.is_file() or point_count < 4:
            raise ValueError("No usable SfM initialization.")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise WorkerError("SFM_INSUFFICIENT_REGISTRATION", "COLMAP did not produce usable camera poses and points.") from exc
    if count < max(8, math.ceil(submitted_views * 0.60)):
        raise WorkerError(
            "SFM_INSUFFICIENT_REGISTRATION",
            f"Registered {count} of {submitted_views} views. Capture sharper frames with more viewpoint overlap.",
        )
    return count


def verify_ply(path):
    import numpy as np

    required = (
        {"x", "y", "z", "opacity"}
        | {f"scale_{i}" for i in range(3)}
        | {f"rot_{i}" for i in range(4)}
        | {f"f_dc_{i}" for i in range(3)}
        | {f"f_rest_{i}" for i in range(45)}
    )
    try:
        with path.open("rb") as file:
            header = []
            while file.tell() < 32768:
                raw = file.readline(1024)
                if not raw.endswith(b"\n"):
                    raise ValueError("Incomplete PLY header.")
                line = raw.decode("ascii").strip()
                if line == "end_header":
                    break
                header.append(line)
            else:
                raise ValueError("PLY header exceeds the limit.")
            if header[:2] != ["ply", "format binary_little_endian 1.0"]:
                raise ValueError("Expected a binary little-endian PLY.")
            elements = [line.split() for line in header if line.startswith("element ")]
            if len(elements) != 1 or len(elements[0]) != 3 or elements[0][1] != "vertex":
                raise ValueError("Expected one vertex element.")
            count = int(elements[0][2])
            properties = [line.split() for line in header if line.startswith("property ")]
            if any(len(item) != 3 or item[1] not in ("float", "float32") for item in properties):
                raise ValueError("Expected float32 Gaussian properties.")
            names = [item[2] for item in properties]
            if len(set(names)) != len(names) or not required.issubset(names):
                raise ValueError("Missing or duplicate Gaussian properties.")
            stride = len(names) * 4
            size = path.stat().st_size
            if count <= 0 or size != file.tell() + count * stride:
                raise ValueError("Empty or truncated PLY vertex data.")
            for offset in range(0, count, 65536):
                values = min(65536, count - offset) * len(names)
                block = np.fromfile(file, dtype="<f4", count=values)
                if block.size != values or not np.isfinite(block).all():
                    raise ValueError("PLY contains non-finite attributes.")
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, ValueError) as exc:
        raise WorkerError("EXPORT_INVALID", f"Invalid Gaussian PLY: {exc}") from exc
    return {
        "path": "out/scene.ply",
        "format": "inria_gaussian_ply",
        "encoding": "binary_little_endian",
        "sh_degree": 3,
        "coordinate_system": "nerfstudio_normalized_z_up",
        "size_bytes": size,
        "gaussian_count": count,
        "sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    private = root / ".worker"
    result_path = root / "worker_result.json"
    out = root / "out"
    ids = {}
    started = monotonic()
    stage = "initializing"
    owns_attempt = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        try:
            private.mkdir()
        except FileExistsError as exc:
            raise WorkerError("WORK_DIR_IN_USE", "Use a new directory for each execution attempt.") from exc
        if result_path.exists() or result_path.is_symlink() or out.exists() or out.is_symlink():
            private.rmdir()
            raise WorkerError("WORK_DIR_IN_USE", "The attempt directory already contains output.")
        owns_attempt = True
        request, source, profile = load_request(args.request, root)
        ids = {key: request[key] for key in ("job_id", "attempt_id")}
        logs = private / "logs"
        dataset = private / "dataset"
        frames = dataset / "images"
        logs.mkdir()
        frames.mkdir(parents=True)

        stage = "preprocessing"
        event("running", stage, ids)
        if request["input_kind"] == "images":
            input_count = prepare_photos(source, frames, (root / "input").resolve())
            matching = "exhaustive"
        else:
            input_count = prepare_video(source, frames, logs, root)
            matching = "sequential"
        frame_count = len(list(frames.glob("*.jpg")))
        if frame_count < MIN_PHOTOS:
            raise WorkerError("INPUT_INVALID", f"Only {frame_count} usable images or video frames.")

        stage = "sfm"
        event("running", stage, ids, frame_count=frame_count)
        model = run_sfm(dataset, frames, matching, logs, root)
        registered = registered_views(dataset, model, frame_count)

        stage = "training"
        event("running", stage, ids, registered_views=registered, total_steps=TRAINING_STEPS[profile])
        train = private / "train"
        run_tool(
            ["ns-train", "splatfacto", "--data", str(dataset), "--output-dir", str(train),
             "--experiment-name", "static3dgs", "--timestamp", "run", "--vis", "tensorboard",
             "--max-num-iterations", str(TRAINING_STEPS[profile]),
             "--steps-per-eval-image", "0", "--steps-per-eval-batch", "0",
             "--steps-per-eval-all-images", "0",
             "--pipeline.model.sh-degree", "3", "--pipeline.model.rasterize-mode", "classic"],
            logs / "train.log", "TRAINING_FAILED", "Splatfacto training failed.", root,
        )

        stage = "exporting"
        event("packaging", stage, ids)
        temporary_ply = private / "scene.ply"
        run_tool(
            ["ns-export", "gaussian-splat", "--load-config",
             str(train / "static3dgs/splatfacto/run/config.yml"), "--output-dir", str(private),
             "--output-filename", temporary_ply.name, "--ply-color-mode", "sh_coeffs"],
            logs / "export.log", "EXPORT_INVALID", "Splatfacto PLY export failed.", root,
        )
        artifact = verify_ply(temporary_ply)
        algorithm_version = f"splatfacto/{version('nerfstudio')}+gsplat/{version('gsplat')}"
        out.mkdir()
        os.replace(temporary_ply, out / "scene.ply")
        publish_json(result_path, {
            "schema_version": 1, "status": "succeeded", **ids,
            "provenance": "captured_reconstruction", "algorithm_version": algorithm_version,
            "worker_version": WORKER_VERSION, "rules_version": RULES_VERSION, "quality_profile": profile,
            "duration_seconds": round(monotonic() - started, 3),
            "input": {"kind": request["input_kind"], "file_count": input_count,
                      "frame_count": frame_count, "registered_views": registered},
            "artifacts": [artifact],
        })
        event("succeeded", "finished", ids, result="worker_result.json")
        return 0
    except Exception as exc:
        if isinstance(exc, WorkerError):
            code, message = exc.code, str(exc)
        else:
            code, message = "INTERNAL_ERROR", "The reconstruction worker encountered an internal error."
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        error = {"code": code, "message": message}
        if owns_attempt:
            try:
                publish_json(result_path, {
                    "schema_version": 1, "status": "failed", **ids, "stage": stage, "error": error,
                    "duration_seconds": round(monotonic() - started, 3),
                })
            except OSError as write_error:
                print(f"Could not write the result JSON: {write_error}", file=sys.stderr)
        event("failed", stage, ids, error=error)
        return ERROR_CODES[code]


if __name__ == "__main__":
    sys.exit(main())
