from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_MODEL = Path("models/openvino/musetalkV15_unet.xml")


def import_runtime_packages():
    missing: list[str] = []
    try:
        import cv2
    except ModuleNotFoundError:
        cv2 = None
        missing.append("opencv-python")

    try:
        import sounddevice as sd
    except ModuleNotFoundError:
        sd = None
        missing.append("sounddevice")

    try:
        import openvino as ov
    except ModuleNotFoundError:
        ov = None
        missing.append("openvino")

    if missing:
        packages = " ".join(missing + ["pillow", "numpy"])
        raise SystemExit(
            "Missing runtime package(s): "
            + ", ".join(missing)
            + "\nInstall them with:\n"
            + f"  python -m pip install {packages}"
        )

    return cv2, sd, ov


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenVINO MuseTalk UNet with a still image and live microphone input. "
            "This is a UNet-only realtime preview: it visualizes the output tensor over "
            "the image, but it does not reconstruct full lipsync frames without MuseTalk's "
            "VAE, Whisper, and face-processing models."
        )
    )
    parser.add_argument("image", type=Path, help="Input face image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="OpenVINO IR .xml path.")
    parser.add_argument("--device", default="AUTO", help="OpenVINO device, e.g. AUTO, GPU, CPU.")
    parser.add_argument("--latent-size", type=int, default=32, help="UNet latent H/W size.")
    parser.add_argument("--audio-tokens", type=int, default=10, help="Number of synthetic audio tokens.")
    parser.add_argument("--audio-dim", type=int, default=384, help="Cross-attention audio feature width.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate.")
    parser.add_argument("--block-ms", type=int, default=200, help="Microphone block size in milliseconds.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay strength.")
    return parser.parse_args()


def load_image(path: Path, latent_size: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    display = np.asarray(image, dtype=np.uint8)
    small = image.resize((latent_size, latent_size), Image.Resampling.BICUBIC)
    rgb = np.asarray(small, dtype=np.float32) / 255.0

    gray = rgb.mean(axis=2, keepdims=True)
    grad_y, grad_x = np.gradient(gray[..., 0])
    channels = [
        rgb[..., 0],
        rgb[..., 1],
        rgb[..., 2],
        gray[..., 0],
        grad_x,
        grad_y,
        np.sin(gray[..., 0] * np.pi),
        np.ones((latent_size, latent_size), dtype=np.float32),
    ]
    latent = np.stack(channels, axis=0)[None, ...].astype(np.float32)
    latent = latent * 2.0 - 1.0
    return display, latent


def audio_to_hidden(audio: np.ndarray, token_count: int, dim: int) -> tuple[np.ndarray, float]:
    if audio.size == 0:
        return np.zeros((1, token_count, dim), dtype=np.float32), 0.0

    audio = audio.astype(np.float32).reshape(-1)
    level = float(np.sqrt(np.mean(np.square(audio))) + 1e-8)
    windowed = audio * np.hanning(audio.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed))
    if spectrum.size == 0 or float(spectrum.max()) <= 0.0:
        features = np.zeros(token_count * dim, dtype=np.float32)
    else:
        xp = np.linspace(0, spectrum.size - 1, token_count * dim)
        features = np.interp(xp, np.arange(spectrum.size), spectrum).astype(np.float32)
        features = np.log1p(features)
        features = (features - features.mean()) / (features.std() + 1e-6)

    hidden = features.reshape(1, token_count, dim).astype(np.float32)
    hidden *= min(level * 12.0, 1.5)
    return hidden, level


def normalize_map(tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor)
    while tensor.ndim > 2:
        tensor = tensor.mean(axis=0)
    tensor = tensor.astype(np.float32)
    tensor -= float(tensor.min())
    peak = float(tensor.max())
    if peak > 1e-6:
        tensor /= peak
    return tensor


def render_preview(cv2, base_rgb: np.ndarray, output: np.ndarray, level: float, fps: float, alpha: float) -> np.ndarray:
    heat = normalize_map(output[0] if output.ndim >= 3 else output)
    heat = cv2.resize(heat, (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    frame = cv2.addWeighted(base_bgr, 1.0 - alpha, heat_bgr, alpha, 0)

    meter_width = int(min(max(level * 1200.0, 4.0), 260.0))
    cv2.rectangle(frame, (16, 16), (276, 44), (20, 20, 20), -1)
    cv2.rectangle(frame, (18, 18), (18 + meter_width, 42), (80, 220, 120), -1)
    cv2.putText(frame, f"mic {level:.4f}  fps {fps:.1f}", (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, "UNet tensor preview - press q to quit", (16, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main() -> int:
    args = parse_args()
    cv2, sd, ov = import_runtime_packages()

    if not args.model.exists():
        raise SystemExit(f"OpenVINO model not found: {args.model}")

    base_rgb, latent = load_image(args.image, args.latent_size)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
    block_size = max(1, int(args.sample_rate * args.block_ms / 1000))

    def audio_callback(indata, frames, callback_time, status) -> None:
        if status:
            print(status, file=sys.stderr)
        block = np.asarray(indata[:, 0], dtype=np.float32).copy()
        try:
            audio_queue.put_nowait(block)
        except queue.Full:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            audio_queue.put_nowait(block)

    core = ov.Core()
    model = core.read_model(str(args.model))
    model.reshape(
        {
            "latent_batch": [1, 8, args.latent_size, args.latent_size],
            "timesteps": [1],
            "encoder_hidden_states": [1, args.audio_tokens, args.audio_dim],
        }
    )
    compiled = core.compile_model(model, args.device)
    output_key = compiled.outputs[0]

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print("Starting microphone. Press q or Esc in the preview window to stop.")

    timestep = np.array([0], dtype=np.int64)
    last_audio = np.zeros(block_size, dtype=np.float32)
    last_time = time.perf_counter()
    fps = 0.0

    with sd.InputStream(channels=1, samplerate=args.sample_rate, blocksize=block_size, callback=audio_callback):
        while True:
            try:
                last_audio = audio_queue.get_nowait()
            except queue.Empty:
                pass

            hidden, level = audio_to_hidden(last_audio, args.audio_tokens, args.audio_dim)
            result = compiled(
                {
                    "latent_batch": latent,
                    "timesteps": timestep,
                    "encoder_hidden_states": hidden,
                }
            )[output_key]

            now = time.perf_counter()
            dt = max(now - last_time, 1e-6)
            fps = (fps * 0.85) + ((1.0 / dt) * 0.15)
            last_time = now

            frame = render_preview(cv2, base_rgb, result, level, fps, args.alpha)
            cv2.imshow("OpenVINO MuseTalk UNet realtime preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
