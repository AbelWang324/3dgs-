# Static 3DGS Worker

一个进程处理一次重建尝试。照片经 EXIF 方向校正后进入 COLMAP。视频由 FFmpeg 抽帧，再用 COLMAP 求位姿和稀疏点云。Worker 选择注册视角最多的 COLMAP 模型，并转换为 Nerfstudio 数据集。Nerfstudio Splatfacto 训练并导出 INRIA 兼容 Gaussian PLY。程序在校验完整文件后才写成功结果。

## Windows 本地界面

这台机器已配置本地依赖。双击工作区根目录的 [启动3DGS重建.cmd](../启动3DGS重建.cmd)，选择照片文件夹或者 MP4 / MOV 视频，然后点击「开始重建」。界面默认使用标准档，也可以选择快速试跑。窗口会显示当前阶段，成功后自动打开包含 `scene.ply` 的目录。每次运行会将素材复制到新的 `runs/local-*` 目录，失败时可从界面打开任务目录查看结果和日志。

## Docker 调用

在有 NVIDIA GPU 的 Linux 节点安装 Docker 和 NVIDIA Container Toolkit。从本目录构建镜像。Dockerfile 固定了上游 Nerfstudio 镜像摘要，并在构建时下载 Splatfacto 评估用的 AlexNet 权重，任务执行时可复用镜像内的缓存。

```bash
docker build -t static-3dgs-worker:1.0.0 .
```

每次执行创建一个新的任务目录。照片放在 `input/photos/`，视频放在 `input/capture.mp4`。将对应的 [照片请求](examples/request.images.json) 或 [视频请求](examples/request.video.json) 复制为该目录下的 `request.json`，填入本次任务 ID 和尝试 ID。以下 Bash 命令假设任务目录是当前目录下的 `runs/attempt-001`。

```bash
attempt_dir="$PWD/runs/attempt-001"
docker run --rm --init --gpus device=0 --shm-size=4g \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$attempt_dir,dst=/job" \
  static-3dgs-worker:1.0.0 \
  --request /job/request.json --work-dir /job
```

本地也可用装有 CUDA 版 PyTorch、gsplat 及 Nerfstudio 的 Python 环境运行。将 `colmap`、`ns-train` 和 `ns-export` 加入当前进程 PATH。处理视频时还需要 `ffmpeg` 和 `ffprobe`，再从本目录执行。

```bash
python -m recon_worker --request ./runs/attempt-001/request.json --work-dir ./runs/attempt-001
```

## 输入规则 `static3dgs-v1`

这些是首版服务规则。后端用同一版本发布 Service Catalog 配置。目录中的照片按文件名排序，仅处理支持的图片扩展名。照片应使用同一焦段和拍摄方向，校正方向和缩放后尺寸相同，并覆盖主体的多侧视角。视频请使用静态场景的连续绕拍。

| 类型 | 接受范围 | 处理 |
| --- | --- | --- |
| `images` | 12–300 张 JPG、JPEG 或 PNG。单张至多 40 MiB 和 3200 万像素，总共至多 2 GiB。 | 校正方向后转 RGB JPEG，限制长边为 1600 像素，进行穷举匹配。 |
| `video` | 一段 MP4 或 MOV，3–180 秒，单边至多 4096 像素，至多 2 GiB。 | 均匀抽取至多 120 帧，再按时间顺序匹配。解码后至少需要 12 帧。 |

`input_path` 相对于任务目录，且必须指向该目录下的 `input/`。面向用户的质量档为 `standard`，训练 30,000 步。`smoke` 训练 3,000 步，用于内部联调。在线任务跳过周期性评估渲染，独立校验最终产物。训练前至少需要注册 8 个视角，且达到送入 COLMAP 帧数的 60%，同时须产出可用稀疏点云。

## 后端任务与结果

- 后端完成权限、额度和幂等校验后入队，分配尝试目录并下载输入文件。每次重试使用新的 `attempt_id` 和目录。Worker 通过独占创建 `.worker/` 拒绝重复使用目录。
- Worker 在 stdout 逐行输出 JSON 状态事件，阶段为 `preprocessing`、`sfm`、`training` 和 `exporting`。工具日志位于 `.worker/logs/`。后端负责执行超时、终止容器和任务重试，按当前尝试更新平台任务。
- 退出码 0 且 [成功结果](examples/worker_result.success.json) 的 `status` 为 `succeeded` 时，上传 `artifacts[].path` 到对象存储，检查长度与 SHA-256，登记 `resource_id`，再生成平台的 `result_manifest`。PLY 路径相对于任务目录。签名 URL 由平台生成。样例结果的大小、耗时和哈希是示例值。
- 非零退出码时读取 [失败结果](examples/worker_result.failed.json) 或 stdout 最后的失败事件。若进程被终止或任务目录不可写，调度器按执行错误处理。下载能力由已登记的产物决定，和 Web Viewer 预览状态独立。

| 退出码 | 错误码 |
| --- | --- |
| 1 | `INTERNAL_ERROR` |
| 2 | `INPUT_INVALID` |
| 3 | `WORK_DIR_IN_USE` |
| 4 | `TOOL_UNAVAILABLE` |
| 5 | `SFM_FAILED`、`SFM_INSUFFICIENT_REGISTRATION` |
| 6 | `TRAINING_FAILED`、`GPU_OOM` |
| 7 | `EXPORT_INVALID` |

## PLY 约定

`out/scene.ply` 使用未压缩的 `binary_little_endian 1.0`、float32 顶点字段、三阶球谐颜色、对数尺度和 logit 不透明度。坐标沿用 Nerfstudio 的归一化 Z-up 约定，单位由重建决定。文件不包含相机和时间轨迹。第一次接入 Web Viewer 时用样例核对朝向、颜色与透明度。

## 本机冒烟验证

在本工作区的 Windows 原生环境和 RTX 3060 12 GB 上，使用 `smoke` 档各训练 3,000 步。素材为同一静态合成场景的 32 个视角和由这些视角编码的一段 MP4。两份产物均通过独立 PLY 解析，字段完整，数值有限，文件大小与 SHA-256 均和结果 JSON 相符。

| 输入 | 注册视角 | 高斯数 | PLY | 结果 |
| --- | --- | --- | --- | --- |
| 32 张照片 | 32/32 | 200,609 | [49.75 MB](runs/rtx3060-photo-demo/out/scene.ply) | [结果 JSON](runs/rtx3060-photo-demo/worker_result.json) |
| 一段 MP4 | 32/32 | 181,413 | [44.99 MB](runs/rtx3060-video/out/scene.ply) | [结果 JSON](runs/rtx3060-video/worker_result.json) |

这台机器已保留本地依赖。要在同一机器复跑，从本目录执行 `& ./.runtime/run_attempt.ps1 -Attempt ./runs/new-attempt`。先在新目录放好 `input/` 与 `request.json`。以上用例验证了 Windows 原生执行和文件格式，Linux 容器与项目 Web Viewer 仍需联调。
