from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import numpy as np
from PIL import Image


APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR = APP_ROOT / "runs"
DEFAULT_MUSETALK_DIR = Path(os.environ.get("MUSETALK_DIR", APP_ROOT / "MuseTalk"))
DEFAULT_OPENVINO_UNET = DEFAULT_MUSETALK_DIR / "models" / "openvino" / "musetalkV15_unet.xml"
DEFAULT_TORCH_DEVICE = "xpu" if (APP_ROOT / ".xpu-probe" / "Scripts" / "python.exe").exists() else "auto"
APP_CSS = """
#main-layout {
    align-items: flex-start;
}

#side-panel {
    max-width: 320px;
}

#face-image {
    max-width: 300px;
}

#face-image .image-container,
#face-image [data-testid="image"] {
    min-height: 150px !important;
    max-height: 190px !important;
}

#preview-output .image-container,
#preview-output [data-testid="image"],
#preview-output video,
#preview-output .video-container {
    min-height: 62vh !important;
    max-height: 70vh !important;
    object-fit: contain !important;
}

#preview-status textarea {
    min-height: 46px !important;
}
"""


def _default_musetalk_python() -> str:
    configured = os.environ.get("MUSETALK_PYTHON")
    if configured:
        return configured

    xpu_python = APP_ROOT / ".xpu-probe" / "Scripts" / "python.exe"
    if xpu_python.exists():
        return str(xpu_python)

    venv_python = DEFAULT_MUSETALK_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)

    return sys.executable


def _quote_yaml(value: str | Path) -> str:
    text = Path(value).resolve().as_posix()
    return '"' + text.replace("\\", "/").replace('"', '\\"') + '"'


def _copy_upload(source: str | None, target_dir: Path, fallback_name: str) -> Path:
    if not source:
        raise gr.Error(f"{fallback_name} をアップロードしてください。")

    source_path = Path(source)
    suffix = source_path.suffix or Path(fallback_name).suffix
    target = target_dir / f"{Path(fallback_name).stem}{suffix}"
    shutil.copy2(source_path, target)
    return target


def _save_audio_array(audio: tuple[int, np.ndarray], target: Path) -> Path:
    sample_rate, data = audio
    samples = np.asarray(data)
    if samples.size == 0:
        raise gr.Error("音声が空です。別の音声ファイルを選び直してください。")

    if samples.ndim == 1:
        samples = samples[:, None]

    if np.issubdtype(samples.dtype, np.floating):
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
    elif samples.dtype != np.int16:
        max_value = np.iinfo(samples.dtype).max if np.issubdtype(samples.dtype, np.integer) else 32767
        samples = (samples.astype(np.float32) / max_value * 32767).astype(np.int16)

    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(samples.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(samples.tobytes())

    return target


def _materialize_audio(source: Any, target_dir: Path, fallback_name: str) -> Path:
    if source is None:
        raise gr.Error("音声を入力してください。")

    target = target_dir / fallback_name
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        suffix = source_path.suffix or Path(fallback_name).suffix
        target = target_dir / f"{Path(fallback_name).stem}{suffix}"
        shutil.copy2(source_path, target)
        return target

    if isinstance(source, tuple) and len(source) == 2:
        return _save_audio_array(source, target)

    raise gr.Error(f"未対応の音声形式です: {type(source).__name__}")


def _musetalk_paths(musetalk_dir: Path) -> dict[str, Path | str]:
    return {
        "version": "v15",
        "openvino_unet_path": musetalk_dir / "models" / "openvino" / "musetalkV15_unet.xml",
    }


def _validate_musetalk(musetalk_dir: Path, paths: dict[str, Path | str]) -> None:
    missing: list[str] = []
    realtime_module = musetalk_dir / "scripts" / "realtime_inference.py"
    whisper_dir = musetalk_dir / "models" / "whisper"

    for path in [realtime_module, whisper_dir, paths["openvino_unet_path"]]:
        if isinstance(path, Path) and not path.exists():
            missing.append(str(path))

    if missing:
        message = "\n".join(missing)
        raise gr.Error(
            "MuseTalk 本体またはモデルが見つかりません。\n"
            "先に `scripts\\setup_musetalk.ps1` を実行するか、MUSETALK_DIR を設定してください。\n\n"
            f"不足:\n{message}"
        )


def _latest_image(result_dir: Path, newer_than: float) -> Path | None:
    candidates = [
        path
        for path in result_dir.rglob("*.png")
        if path.is_file() and path.stat().st_mtime >= newer_than
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_command_with_live_frames(
    command: list[str],
    cwd: Path,
    frame_dir: Path,
    stable_frame: Path,
    final_video: Path,
    started_at: float,
    logs: list[str],
) -> Iterable[tuple[str | None, str | None, str]]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    output_queue: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line.rstrip())

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    last_frame: Path | None = None
    while process.poll() is None:
        while True:
            try:
                logs.append(output_queue.get_nowait())
            except queue.Empty:
                break

        frame = _latest_image(frame_dir, started_at)
        if frame is not None:
            try:
                shutil.copy2(frame, stable_frame)
                last_frame = stable_frame
            except OSError:
                last_frame = frame
        video = final_video if final_video.exists() else None
        yield (
            str(last_frame) if last_frame else None,
            str(video) if video else None,
            "\n".join(logs[-180:]),
        )
        time.sleep(0.4)

    reader.join(timeout=1)
    while True:
        try:
            logs.append(output_queue.get_nowait())
        except queue.Empty:
            break

    return_code = process.wait()
    frame = _latest_image(frame_dir, started_at)
    if frame is not None:
        try:
            shutil.copy2(frame, stable_frame)
            last_frame = stable_frame
        except OSError:
            last_frame = frame
    video = final_video if final_video.exists() else None
    yield (
        str(last_frame) if last_frame else None,
        str(video) if video else None,
        "\n".join(logs[-180:]),
    )

    if return_code != 0:
        raise gr.Error(f"MuseTalk が終了コード {return_code} で停止しました。ログを確認してください。")


def generate_realtime_lipsync(
    image_path: str | None,
    audio_path: Any,
    musetalk_dir_text: str,
    python_executable: str,
    runtime_device: str,
    openvino_unet_path: str,
    openvino_device: str,
    fps: int,
    bbox_shift: int,
    keep_coord: bool,
    ffmpeg_path: str,
) -> Iterable[tuple[None | str, None | str, str]]:
    started_at = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    avatar_id = f"app_{run_id}"
    run_dir = RUNS_DIR / run_id
    input_dir = run_dir / "inputs"
    frame_source_dir = input_dir / "avatar_frames"
    result_dir = run_dir / "realtime_results"
    frame_source_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not image_path:
        yield None, None, "顔画像をアップロードしてください。"
        return

    if audio_path is None:
        yield image_path, None, "マイク録音を停止すると MuseTalk のライブ生成を開始します。"
        return

    musetalk_dir = Path(musetalk_dir_text or DEFAULT_MUSETALK_DIR).expanduser().resolve()
    python_bin = python_executable.strip() or sys.executable
    paths = _musetalk_paths(musetalk_dir)
    if openvino_unet_path.strip():
        paths["openvino_unet_path"] = Path(openvino_unet_path.strip()).expanduser().resolve()
    _validate_musetalk(musetalk_dir, paths)

    image = _copy_upload(image_path, input_dir, "avatar.png")
    Image.open(image).convert("RGB").save(frame_source_dir / "00000000.png")
    audio = _materialize_audio(audio_path, input_dir, "voice.wav")

    config_path = run_dir / "realtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"{avatar_id}:",
                "  preparation: True",
                f"  video_path: {_quote_yaml(frame_source_dir)}",
                f"  bbox_shift: {bbox_shift}",
                "  audio_clips:",
                f"    preview: {_quote_yaml(audio)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    avatar_root = result_dir / "v15" / "avatars" / avatar_id
    frame_dir = avatar_root / "tmp"
    stable_frame = run_dir / "latest_frame.png"
    final_video = avatar_root / "vid_output" / "preview.mp4"

    command = [
        python_bin,
        "-m",
        "scripts.realtime_inference",
        "--inference_config",
        str(config_path),
        "--result_dir",
        str(result_dir),
        "--version",
        "v15",
        "--device",
        runtime_device,
        "--openvino_unet_path",
        str(paths["openvino_unet_path"]),
        "--openvino_device",
        openvino_device,
        "--fps",
        str(fps),
        "--batch_size",
        "1",
    ]

    if ffmpeg_path.strip():
        command.extend(["--ffmpeg_path", ffmpeg_path.strip()])
    if keep_coord:
        command.append("--saved_coord")

    logs = [
        f"Run: {run_id}",
        f"MuseTalk realtime: {musetalk_dir}",
        "マイク録音を保存しました。MuseTalk のリアルタイム推論を開始します。",
        "生成中のフレームをライブ表示します。",
        "",
        "Command:",
        " ".join(command),
        "",
    ]
    yield str(image), None, "\n".join(logs)

    try:
        for frame, video, log in _run_command_with_live_frames(command, musetalk_dir, frame_dir, stable_frame, final_video, started_at, logs):
            yield frame or str(image), video, log
    except Exception as exc:
        logs.append("")
        logs.append(f"ERROR: {exc}")
        frame = _latest_image(frame_dir, started_at)
        yield str(frame) if frame else str(image), str(final_video) if final_video.exists() else None, "\n".join(logs[-180:])
        return

    if final_video.exists():
        logs.append("")
        logs.append(f"完了: {final_video}")
        yield str(stable_frame if stable_frame.exists() else image), str(final_video), "\n".join(logs[-180:])
        return

    logs.append("完成動画が見つかりませんでした。MuseTalk のログを確認してください。")
    yield str(stable_frame if stable_frame.exists() else image), None, "\n".join(logs[-180:])


with gr.Blocks(title="MuseTalk 口パク生成", css=APP_CSS) as demo:
    gr.Markdown("# MuseTalk 口パク生成")

    with gr.Row(elem_id="main-layout"):
        with gr.Column(scale=1, min_width=260, elem_id="side-panel"):
            image_input = gr.Image(label="顔画像", type="filepath", sources=["upload"], height=170, elem_id="face-image")

            with gr.Accordion("MuseTalk 設定", open=False):
                musetalk_dir = gr.Textbox(
                    label="MuseTalk ディレクトリ",
                    value=str(DEFAULT_MUSETALK_DIR),
                    placeholder=r"C:\path\to\MuseTalk",
                )
                python_executable = gr.Textbox(
                    label="Python 実行ファイル",
                    value=_default_musetalk_python(),
                    placeholder=r"C:\path\to\python.exe",
                )
                runtime_device = gr.Radio(["auto", "cpu", "xpu"], value=DEFAULT_TORCH_DEVICE, label="補助デバイス")
                openvino_unet_path = gr.Textbox(
                    label="OpenVINO UNet",
                    value=str(DEFAULT_OPENVINO_UNET),
                )
                openvino_device = gr.Radio(["AUTO", "GPU", "CPU"], value="AUTO", label="OpenVINO デバイス")
                fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
                bbox_shift = gr.Slider(-20, 20, value=0, step=1, label="BBox shift")
                keep_coord = gr.Checkbox(value=True, label="座標キャッシュを保存")
                ffmpeg_path = gr.Textbox(
                    label="FFmpeg bin パス",
                    value="",
                    placeholder=r"C:\path\to\ffmpeg\bin",
                )

        with gr.Column(scale=4, min_width=520):
            with gr.Tabs():
                with gr.Tab("OpenVINO リアルタイム推論"):
                    preview_live_output = gr.Image(label="MuseTalk ライブ表示", type="filepath", height=640, elem_id="preview-output")
                    with gr.Row():
                        preview_mic = gr.Audio(
                            label="マイク",
                            type="filepath",
                            sources=["microphone"],
                            format="wav",
                            max_length=30,
                        )
                        preview_generate_button = gr.Button("録音停止後にライブ生成", variant="primary")
                    preview_video_output = gr.Video(label="完成動画", autoplay=True)
                    preview_status = gr.Textbox(label="ログ", lines=12, max_lines=18, elem_id="preview-status")

                    preview_mic.stop_recording(
                        generate_realtime_lipsync,
                        inputs=[
                            image_input,
                            preview_mic,
                            musetalk_dir,
                            python_executable,
                            runtime_device,
                            openvino_unet_path,
                            openvino_device,
                            fps,
                            bbox_shift,
                            keep_coord,
                            ffmpeg_path,
                        ],
                        outputs=[preview_live_output, preview_video_output, preview_status],
                    )
                    preview_generate_button.click(
                        generate_realtime_lipsync,
                        inputs=[
                            image_input,
                            preview_mic,
                            musetalk_dir,
                            python_executable,
                            runtime_device,
                            openvino_unet_path,
                            openvino_device,
                            fps,
                            bbox_shift,
                            keep_coord,
                            ffmpeg_path,
                        ],
                        outputs=[preview_live_output, preview_video_output, preview_status],
                    )


if __name__ == "__main__":
    demo.queue().launch()
