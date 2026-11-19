"""A local Windows interface for the static 3DGS worker."""

from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import shutil
import subprocess
from threading import Thread
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk
from uuid import uuid4

from recon_local import PROJECT, worker_launch
from recon_worker import (
    MAX_INPUT_BYTES, MAX_PHOTO_BYTES, MAX_PHOTOS, MIN_PHOTOS,
    PHOTO_EXTENSIONS, RULES_VERSION, TRAINING_STEPS, VIDEO_EXTENSIONS,
)


PROFILES = ("standard", "smoke")
PROFILE_LABELS = (f"标准 · {TRAINING_STEPS['standard']:,} 步",
                  f"快速试跑 · {TRAINING_STEPS['smoke']:,} 步")
STAGES = {
    "preprocessing": "正在检查素材并准备训练图像…",
    "sfm": "正在估计相机位姿…",
    "training": "正在训练高斯模型…",
    "exporting": "正在导出并校验 Gaussian PLY…",
}
ERRORS = {
    "INPUT_INVALID": "素材未通过检查",
    "TOOL_UNAVAILABLE": "重建工具无法启动",
    "SFM_FAILED": "相机配准失败",
    "SFM_INSUFFICIENT_REGISTRATION": "成功配准的视角太少，请检查照片重叠或视频清晰度",
    "TRAINING_FAILED": "高斯模型训练失败",
    "GPU_OOM": "显存不足",
    "EXPORT_INVALID": "PLY 导出或校验失败",
    "INTERNAL_ERROR": "重建程序遇到内部错误",
}


def input_files(source, kind):
    if kind == "images":
        if not source.is_dir():
            raise ValueError("请选择一个照片文件夹。")
        files = sorted(
            file for file in source.iterdir()
            if file.is_file() and file.suffix.lower() in PHOTO_EXTENSIONS
        )
        if not MIN_PHOTOS <= len(files) <= MAX_PHOTOS:
            raise ValueError(f"找到 {len(files)} 张照片，请提供 {MIN_PHOTOS}–{MAX_PHOTOS} 张 JPG 或 PNG。")
    else:
        if not source.is_file() or source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("请选择一个 MP4 或 MOV 视频。")
        files = [source]

    sizes = [file.stat().st_size for file in files]
    if sum(sizes) > MAX_INPUT_BYTES:
        raise ValueError("素材总大小超过 2 GiB。")
    if kind == "images" and max(sizes) > MAX_PHOTO_BYTES:
        raise ValueError("单张照片大小不能超过 40 MiB。")
    return files


def reconstruct(source, kind, profile, events):
    try:
        files = input_files(source, kind)
        identifier = f"local-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        attempt = PROJECT / "runs" / identifier
        command, process_options = worker_launch(attempt)

        target = attempt / "input"
        if kind == "images":
            target /= "photos"
        target.mkdir(parents=True, exist_ok=False)
        events.put(("directory", attempt))
        for file in files:
            filename = file.name if kind == "images" else f"capture{file.suffix.lower()}"
            shutil.copyfile(file, target / filename)
        input_path = target if kind == "images" else target / filename
        request = {
            "job_id": identifier,
            "attempt_id": "attempt-001",
            "input_kind": kind,
            "input_path": input_path.relative_to(attempt).as_posix(),
            "rules_version": RULES_VERSION,
            "quality_profile": profile,
        }
        (attempt / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")

        with (attempt / "launcher.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                command, **process_options, creationflags=subprocess.CREATE_NO_WINDOW,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            ) as worker:
                for line in worker.stdout:
                    log.write(line)
                    try:
                        state = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    stage = state.get("stage") if isinstance(state, dict) else None
                    if stage in STAGES:
                        status = STAGES[stage]
                        if stage == "training":
                            status += f" 注册 {state['registered_views']} 个视角，训练 {TRAINING_STEPS[profile]:,} 步。"
                        events.put(("status", status))
                exit_code = worker.wait()

        result_path = attempt / "worker_result.json"
        if not result_path.is_file():
            raise RuntimeError(f"重建进程退出（{exit_code}），请查看任务目录中的 launcher.log。")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if exit_code != 0 or result.get("status") != "succeeded":
            error = result.get("error", {})
            code = error.get("code", "UNKNOWN")
            summary = ERRORS.get(code, "重建失败")
            detail = error.get("message")
            raise RuntimeError(f"{summary}（{code}）" + (f"\n{detail}" if detail else ""))

        artifact = next((item for item in result["artifacts"] if item["path"] == "out/scene.ply"), None)
        output = attempt / "out"
        if artifact is None or not (output / "scene.ply").is_file():
            raise RuntimeError("任务没有生成可用的 scene.ply。")
        events.put(("done", (output, artifact)))
    except Exception as exc:
        events.put(("failed", str(exc)))


class ReconstructionWindow:
    def __init__(self, root):
        self.root = root
        self.events = SimpleQueue()
        self.busy = False
        self.target_directory = None
        self.active_kind = None
        self.kind = tk.StringVar(value="images")
        self.source = tk.StringVar()
        self.hint = tk.StringVar()
        self.status = tk.StringVar(value="请选择要重建的素材。")

        root.title("3DGS 重建")
        root.geometry("680x435")
        root.minsize(560, 415)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.configure(background="#f1f4fa")
        for name in ("TkDefaultFont", "TkTextFont"):
            font.nametofont(name).configure(family="Microsoft YaHei UI", size=10)
        root.option_add("*TCombobox*Listbox.font", font.nametofont("TkDefaultFont"))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Page.TFrame", background="#f1f4fa")
        style.configure("Title.TLabel", background="#f1f4fa", foreground="#172640",
                        font=("Microsoft YaHei UI", 23, "bold"))
        style.configure("Card.TFrame", background="white")
        style.configure("TLabel", background="white", foreground="#243754")
        style.configure("Help.TLabel", background="white", foreground="#586987")
        style.configure("TRadiobutton", background="white", foreground="#243754", padding=(0, 4))
        style.map("TRadiobutton", background=[("active", "white")])
        style.configure("TButton", padding=(12, 7), foreground="#243754")
        style.configure("Primary.TButton", background="#2856c8", bordercolor="#2856c8",
                        foreground="white", font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#7990c5"), ("active", "#2048ad")],
                  foreground=[("disabled", "white")])

        page = ttk.Frame(root, style="Page.TFrame", padding=(24, 18))
        page.pack(fill="both", expand=True)
        page.columnconfigure(0, weight=1)
        ttk.Label(page, text="3DGS 重建", style="Title.TLabel").grid(sticky="w", pady=(0, 12))

        card = ttk.Frame(page, style="Card.TFrame", padding=(20, 14))
        card.grid(sticky="nsew")
        card.columnconfigure(0, weight=1)

        types = ttk.Frame(card, style="Card.TFrame")
        types.grid(sticky="w")
        self.type_buttons = []
        for kind, label in (("images", "照片文件夹"), ("video", "视频文件")):
            button = ttk.Radiobutton(types, text=label, variable=self.kind, value=kind, command=self.change_kind)
            button.pack(side="left", padx=(0, 22))
            self.type_buttons.append(button)

        source_row = ttk.Frame(card, style="Card.TFrame")
        source_row.grid(sticky="ew", pady=(8, 4))
        self.source_entry = ttk.Entry(source_row, textvariable=self.source, state="readonly")
        self.source_entry.pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 8))
        self.browse_button = ttk.Button(source_row, command=self.choose_source)
        self.browse_button.pack(side="right")
        ttk.Label(card, textvariable=self.hint, style="Help.TLabel").grid(sticky="w", pady=(0, 12))

        options = ttk.Frame(card, style="Card.TFrame")
        options.grid(sticky="ew")
        ttk.Label(options, text="质量").pack(side="left", padx=(0, 12))
        self.quality = ttk.Combobox(options, values=PROFILE_LABELS, state="readonly", width=25)
        self.quality.current(0)
        self.quality.pack(side="left", ipady=4)
        self.start_button = ttk.Button(options, text="开始重建", style="Primary.TButton", command=self.start)
        self.start_button.pack(side="right")

        feedback = ttk.Frame(page, style="Card.TFrame", padding=(20, 12))
        feedback.grid(sticky="ew", pady=(10, 0))
        feedback.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(feedback, textvariable=self.status, wraplength=470, justify="left")
        self.status_label.grid(row=0, column=0, sticky="w")
        self.directory_button = ttk.Button(feedback, command=self.open_directory)
        self.directory_button.grid(row=0, column=1, sticky="e", padx=(10, 0))
        self.directory_button.grid_remove()
        self.progress = ttk.Progressbar(feedback, mode="indeterminate")
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.progress.grid_remove()
        feedback.bind("<Configure>", self.resize_feedback)
        self.change_kind()

    def resize_feedback(self, event):
        self.status_label.configure(wraplength=max(180, event.width - 180))

    def change_kind(self):
        kind = self.kind.get()
        if kind == self.active_kind:
            return
        self.active_kind = kind
        self.source.set("")
        is_video = kind == "video"
        self.browse_button.configure(text="选择视频…" if is_video else "选择文件夹…")
        self.hint.set("MP4 或 MOV，3–180 秒，拍摄静态场景。"
                      if is_video else "12–300 张 JPG 或 PNG，照片直接放在同一文件夹。")
        self.start_button.state(["disabled"])
        if self.target_directory is None:
            self.status.set("请选择要重建的素材。")

    def choose_source(self):
        if self.kind.get() == "images":
            selected = filedialog.askdirectory(title="选择照片文件夹", parent=self.root, mustexist=True)
        else:
            selected = filedialog.askopenfilename(
                title="选择绕拍视频", parent=self.root, filetypes=[("MP4 或 MOV 视频", "*.mp4 *.mov")],
            )
        if selected:
            self.source.set(selected)
            self.status.set("素材已选择，可以开始重建。")
            self.target_directory = None
            self.directory_button.grid_remove()
            self.start_button.state(["!disabled"])

    def start(self):
        if self.busy or not self.source.get():
            return
        self.busy = True
        self.target_directory = None
        self.directory_button.grid_remove()
        for control in (*self.type_buttons, self.quality, self.browse_button, self.start_button):
            control.state(["disabled"])
        self.status.set("正在复制输入素材…")
        self.progress.grid()
        self.progress.start(15)
        args = (Path(self.source.get()), self.kind.get(), PROFILES[self.quality.current()], self.events)
        Thread(target=reconstruct, args=args, daemon=True).start()
        self.root.after(100, self.poll_events)

    def poll_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "status":
                    self.status.set(data)
                elif kind == "directory":
                    self.target_directory = data
                else:
                    self.finish(kind == "done", data)
        except Empty:
            if self.busy:
                self.root.after(100, self.poll_events)

    def finish(self, success, data):
        self.busy = False
        self.progress.stop()
        self.progress.grid_remove()
        for control in (*self.type_buttons, self.quality, self.browse_button, self.start_button):
            control.state(["!disabled"])
        if success:
            self.target_directory, artifact = data
            size_mb = artifact["size_bytes"] / 1_000_000
            self.status.set(f"已生成 scene.ply · {artifact['gaussian_count']:,} 个高斯 · {size_mb:.2f} MB")
            self.directory_button.configure(text="打开输出目录")
            self.directory_button.grid()
            self.open_directory()
        else:
            self.status.set(data)
            if self.target_directory is not None:
                self.directory_button.configure(text="打开任务目录")
                self.directory_button.grid()
            messagebox.showerror("重建失败", data, parent=self.root)

    def open_directory(self):
        if self.target_directory is not None:
            try:
                os.startfile(self.target_directory)
            except OSError:
                messagebox.showinfo("目录位置", str(self.target_directory), parent=self.root)

    def close(self):
        if self.busy:
            messagebox.showinfo("正在重建", "请保持窗口开启，等当前任务完成后再关闭。", parent=self.root)
        else:
            self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    ReconstructionWindow(window)
    window.mainloop()
