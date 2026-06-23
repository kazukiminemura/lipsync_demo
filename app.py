from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import numpy as np


APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR = APP_ROOT / "runs"
DEFAULT_MUSETALK_DIR = Path(os.environ.get("MUSETALK_DIR", APP_ROOT / "MuseTalk"))


def _default_musetalk_python() -> str:
    configured = os.environ.get("MUSETALK_PYTHON")
    if configured:
        return configured

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
        raise gr.Error("マイク音声が空です。録音し直してください。")

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


def _musetalk_paths(musetalk_dir: Path, version: str) -> dict[str, Path | str]:
    if version == "v15":
        return {
            "version": "v15",
            "unet_model_path": musetalk_dir / "models" / "musetalkV15" / "unet.pth",
            "unet_config": musetalk_dir / "models" / "musetalkV15" / "musetalk.json",
        }

    return {
        "version": "v1",
        "unet_model_path": musetalk_dir / "models" / "musetalk" / "pytorch_model.bin",
        "unet_config": musetalk_dir / "models" / "musetalk" / "musetalk.json",
    }


def _validate_musetalk(musetalk_dir: Path, paths: dict[str, Path | str]) -> None:
    missing: list[str] = []
    inference_module = musetalk_dir / "scripts" / "inference.py"
    whisper_dir = musetalk_dir / "models" / "whisper"

    for path in [inference_module, whisper_dir, paths["unet_model_path"], paths["unet_config"]]:
        if isinstance(path, Path) and not path.exists():
            missing.append(str(path))

    if missing:
        message = "\n".join(missing)
        raise gr.Error(
            "MuseTalk 本体またはモデルが見つかりません。\n"
            "先に `scripts\\setup_musetalk.ps1` を実行するか、MUSETALK_DIR を設定してください。\n\n"
            f"不足:\n{message}"
        )


def _latest_video(result_dir: Path, newer_than: float) -> Path | None:
    candidates = [
        path
        for path in result_dir.rglob("*.mp4")
        if path.is_file() and path.stat().st_mtime >= newer_than
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_command(command: list[str], cwd: Path) -> Iterable[str]:
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

    assert process.stdout is not None
    for line in process.stdout:
        yield line.rstrip()

    return_code = process.wait()
    if return_code != 0:
        raise gr.Error(f"MuseTalk が終了コード {return_code} で停止しました。ログを確認してください。")


def generate_lipsync(
    image_path: str | None,
    audio_path: Any,
    musetalk_dir_text: str,
    python_executable: str,
    version: str,
    fps: int,
    bbox_shift: int,
    batch_size: int,
    use_float16: bool,
    keep_coord: bool,
    ffmpeg_path: str,
) -> Iterable[tuple[None | str, str]]:
    started_at = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    input_dir = run_dir / "inputs"
    result_dir = run_dir / "results"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    musetalk_dir = Path(musetalk_dir_text or DEFAULT_MUSETALK_DIR).expanduser().resolve()
    python_bin = python_executable.strip() or sys.executable
    paths = _musetalk_paths(musetalk_dir, version)
    _validate_musetalk(musetalk_dir, paths)

    image = _copy_upload(image_path, input_dir, "avatar.png")
    audio = _materialize_audio(audio_path, input_dir, "voice.wav")
    config_path = run_dir / "inference.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task_0:",
                f"  video_path: {_quote_yaml(image)}",
                f"  audio_path: {_quote_yaml(audio)}",
                '  result_name: "lipsync.mp4"',
                f"  bbox_shift: {bbox_shift}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    command = [
        python_bin,
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config_path),
        "--result_dir",
        str(result_dir),
        "--unet_model_path",
        str(paths["unet_model_path"]),
        "--unet_config",
        str(paths["unet_config"]),
        "--version",
        str(paths["version"]),
        "--fps",
        str(fps),
        "--batch_size",
        str(batch_size),
        "--bbox_shift",
        str(bbox_shift),
        "--output_vid_name",
        "lipsync.mp4",
    ]

    if ffmpeg_path.strip():
        command.extend(["--ffmpeg_path", ffmpeg_path.strip()])
    if use_float16:
        command.append("--use_float16")
    if keep_coord:
        command.append("--saved_coord")

    logs = [
        f"Run: {run_id}",
        f"MuseTalk: {musetalk_dir}",
        "画像と音声を保存しました。推論を開始します。",
        "",
        "Command:",
        " ".join(command),
        "",
    ]
    yield None, "\n".join(logs)

    try:
        for line in _run_command(command, musetalk_dir):
            logs.append(line)
            video = _latest_video(result_dir, started_at)
            yield str(video) if video else None, "\n".join(logs[-160:])
    except Exception as exc:
        logs.append("")
        logs.append(f"ERROR: {exc}")
        video = _latest_video(result_dir, started_at)
        yield str(video) if video else None, "\n".join(logs[-180:])
        return

    video = _latest_video(result_dir, started_at)
    if video is None:
        logs.append("出力動画が見つかりませんでした。MuseTalk のログを確認してください。")
        yield None, "\n".join(logs[-180:])
        return

    logs.append("")
    logs.append(f"完了: {video}")
    yield str(video), "\n".join(logs[-180:])


def mic_level(audio: tuple[int, np.ndarray] | None) -> str:
    if audio is None:
        return "マイク待機中"

    sample_rate, data = audio
    samples = np.asarray(data)
    if samples.size == 0:
        return "マイク入力: 0.0 秒 / 音量 0%"

    duration = samples.shape[0] / float(sample_rate)
    if np.issubdtype(samples.dtype, np.integer):
        peak = np.iinfo(samples.dtype).max
        level = float(np.max(np.abs(samples))) / max(float(peak), 1.0)
    else:
        level = float(np.max(np.abs(samples)))
    return f"マイク入力: {duration:.1f} 秒 / 音量 {min(level * 100, 100):.0f}%"


with gr.Blocks(title="MuseTalk 口パク生成") as demo:
    gr.Markdown("# MuseTalk 口パク生成")

    image_input = gr.Image(label="顔画像", type="filepath", sources=["upload"])

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
        with gr.Row():
            version = gr.Radio(["v15", "v1"], value="v15", label="モデル")
            fps = gr.Slider(1, 60, value=25, step=1, label="FPS")
            batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch size")
        with gr.Row():
            bbox_shift = gr.Slider(-20, 20, value=0, step=1, label="BBox shift")
            use_float16 = gr.Checkbox(value=True, label="float16")
            keep_coord = gr.Checkbox(value=True, label="座標キャッシュを保存")
        ffmpeg_path = gr.Textbox(
            label="FFmpeg bin パス",
            value="",
            placeholder=r"C:\path\to\ffmpeg\bin",
        )

    with gr.Tabs():
        with gr.Tab("音声ファイル"):
            audio_input = gr.Audio(label="音声", type="filepath", sources=["upload"])
            generate_button = gr.Button("生成", variant="primary")
            file_video_output = gr.Video(label="出力動画", autoplay=True)
            file_log_output = gr.Textbox(label="ログ", lines=18, max_lines=24)

            generate_button.click(
                generate_lipsync,
                inputs=[
                    image_input,
                    audio_input,
                    musetalk_dir,
                    python_executable,
                    version,
                    fps,
                    bbox_shift,
                    batch_size,
                    use_float16,
                    keep_coord,
                    ffmpeg_path,
                ],
                outputs=[file_video_output, file_log_output],
            )

        with gr.Tab("マイク"):
            mic_input = gr.Audio(
                label="マイク",
                type="numpy",
                sources=["microphone"],
                streaming=True,
                format="wav",
                max_length=30,
            )
            mic_generate_button = gr.Button("録音から再生成", variant="secondary")
            mic_video_output = gr.Video(label="ライブ表示", autoplay=True)
            mic_log_output = gr.Textbox(label="ログ", lines=18, max_lines=24)

            mic_input.stream(mic_level, inputs=[mic_input], outputs=[mic_log_output])
            mic_input.stop_recording(
                generate_lipsync,
                inputs=[
                    image_input,
                    mic_input,
                    musetalk_dir,
                    python_executable,
                    version,
                    fps,
                    bbox_shift,
                    batch_size,
                    use_float16,
                    keep_coord,
                    ffmpeg_path,
                ],
                outputs=[mic_video_output, mic_log_output],
            )
            mic_generate_button.click(
                generate_lipsync,
                inputs=[
                    image_input,
                    mic_input,
                    musetalk_dir,
                    python_executable,
                    version,
                    fps,
                    bbox_shift,
                    batch_size,
                    use_float16,
                    keep_coord,
                    ffmpeg_path,
                ],
                outputs=[mic_video_output, mic_log_output],
            )


if __name__ == "__main__":
    demo.queue().launch()
