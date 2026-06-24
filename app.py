from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import warnings
import wave
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import openvino as ov
import requests
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm


warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Pass sr=16000, n_fft=800 as keyword args.*", category=FutureWarning)

APP_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = APP_ROOT / "vendor"
WAV2LIP_DIR = VENDOR_DIR / "Wav2Lip"
CHECKPOINT_DIR = APP_ROOT / "checkpoints"
MODEL_DIR = APP_ROOT / "models" / "wav2lip"
RUNS_DIR = APP_ROOT / "runs"

FACE_DETECTION_MODEL = MODEL_DIR / "face_detection.xml"
WAV2LIP_MODEL = MODEL_DIR / "wav2lip.xml"
IMG_SIZE = 96
MEL_STEP_SIZE = 16


def run_command(command: list[str], cwd: Path | None = None) -> None:
    process = subprocess.run(command, cwd=cwd, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(command)}")


def ensure_wav2lip_repo() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    if (WAV2LIP_DIR / "models" / "wav2lip.py").exists():
        return
    if WAV2LIP_DIR.exists():
        shutil.rmtree(WAV2LIP_DIR)
    print("Cloning Wav2Lip...")
    run_command(["git", "clone", "--depth", "1", "https://github.com/Rudrabha/Wav2Lip.git", str(WAV2LIP_DIR)])


def add_wav2lip_to_path() -> None:
    ensure_wav2lip_repo()
    for path in (VENDOR_DIR, WAV2LIP_DIR):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def download_file(url: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {target.name}...")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return target


def load_wav2lip_checkpoint(checkpoint_path: Path):
    add_wav2lip_to_path()
    from Wav2Lip.models import Wav2Lip

    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state)
    return model.eval()


def ensure_openvino_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    add_wav2lip_to_path()

    if FACE_DETECTION_MODEL.exists() and WAV2LIP_MODEL.exists():
        print("OpenVINO models are ready.")
        return

    from Wav2Lip.face_detection.detection.sfd.net_s3fd import s3fd

    face_checkpoint = CHECKPOINT_DIR / "face_detection.pth"
    download_file("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth", face_checkpoint)

    if not FACE_DETECTION_MODEL.exists():
        print("Converting face detector to OpenVINO IR...")
        face_detector = s3fd()
        face_detector.load_state_dict(torch.load(face_checkpoint, map_location="cpu"))
        face_detector.eval()
        dummy_input = torch.FloatTensor(np.random.rand(1, 3, 768, 576))
        ov_model = ov.convert_model(face_detector, example_input=dummy_input)
        ov.save_model(ov_model, FACE_DETECTION_MODEL)
        print(f"Saved {FACE_DETECTION_MODEL}")

    wav2lip_checkpoint = Path(
        hf_hub_download(
            repo_id="numz/wav2lip_studio",
            filename="Wav2lip/wav2lip.pth",
            local_dir=str(CHECKPOINT_DIR),
        )
    )

    if not WAV2LIP_MODEL.exists():
        print("Converting Wav2Lip to OpenVINO IR...")
        wav2lip = load_wav2lip_checkpoint(wav2lip_checkpoint)
        image_batch = torch.FloatTensor(np.random.rand(16, 6, IMG_SIZE, IMG_SIZE))
        mel_batch = torch.FloatTensor(np.random.rand(16, 1, 80, MEL_STEP_SIZE))
        ov_model = ov.convert_model(wav2lip, example_input={"audio_sequences": mel_batch, "face_sequences": image_batch})
        ov.save_model(ov_model, WAV2LIP_MODEL)
        print(f"Saved {WAV2LIP_MODEL}")


def nms(dets: np.ndarray, threshold: float) -> list[int]:
    if len(dets) == 0:
        return []

    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = width * height / (areas[i] + areas[order[1:]] - width * height)
        order = order[np.where(overlap <= threshold)[0] + 1]

    return keep


def decode_boxes(loc: torch.Tensor, priors: torch.Tensor, variances: list[float]) -> torch.Tensor:
    boxes = torch.cat(
        (
            priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
        ),
        dim=2,
    )
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def batch_detect(face_detector: ov.CompiledModel, images: np.ndarray) -> list[list[np.ndarray]]:
    input_batch = images - np.array([104, 117, 123])
    input_batch = input_batch.transpose(0, 3, 1, 2).astype(np.float32)
    raw_outputs = face_detector({"x": input_batch})
    outputs = [torch.Tensor(raw_outputs[output]) for output in face_detector.outputs]

    bboxlist = []
    for i in range(len(outputs) // 2):
        cls_output = F.softmax(outputs[i * 2], dim=1).data.cpu()
        reg_output = outputs[i * 2 + 1].data.cpu()
        batch, _, _, _ = cls_output.size()
        stride = 2 ** (i + 2)

        for _, h_idx, w_idx in zip(*np.where(cls_output[:, 1, :, :] > 0.05)):
            axc = stride / 2 + w_idx * stride
            ayc = stride / 2 + h_idx * stride
            score = cls_output[:, 1, h_idx, w_idx]
            loc = reg_output[:, :, h_idx, w_idx].contiguous().view(batch, 1, 4)
            priors = torch.Tensor([[axc, ayc, stride * 4, stride * 4]]).view(1, 1, 4)
            boxes = decode_boxes(loc, priors, [0.1, 0.2])[:, 0]
            bboxlist.append(torch.cat([boxes, score.unsqueeze(1)], 1).cpu().numpy())

    bbox_array = np.array(bboxlist) if bboxlist else np.zeros((1, len(images), 5))
    detections: list[list[np.ndarray]] = []
    for image_index in range(bbox_array.shape[1]):
        keep = nms(bbox_array[:, image_index, :], 0.3)
        detections.append([box for box in bbox_array[keep, image_index, :] if box[-1] > 0.5])
    return detections


def smooth_boxes(boxes: np.ndarray, window: int = 5) -> np.ndarray:
    smoothed = boxes.copy()
    for i in range(len(boxes)):
        smoothed[i] = np.mean(boxes[max(0, i - window + 1) : i + 1], axis=0)
    return smoothed


def face_detect(
    face_detector: ov.CompiledModel,
    frames: list[np.ndarray],
    batch_size: int,
    pads: tuple[int, int, int, int],
    smooth: bool,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    predictions: list[tuple[int, int, int, int] | None] = []
    for i in tqdm(range(0, len(frames), batch_size), desc="Detecting faces"):
        detections = batch_detect(face_detector, np.array(frames[i : i + batch_size]))
        for detection in detections:
            if not detection:
                predictions.append(None)
                continue
            x1, y1, x2, y2 = map(int, np.clip(detection[0][:-1], 0, None))
            predictions.append((x1, y1, x2, y2))

    pady1, pady2, padx1, padx2 = pads
    boxes: list[list[int]] = []
    for rect, frame in zip(predictions, frames):
        if rect is None:
            raise RuntimeError("Face was not detected. Try a clearer frontal face or a smaller crop.")
        x1, y1, x2, y2 = rect
        boxes.append(
            [
                max(0, y1 - pady1),
                min(frame.shape[0], y2 + pady2),
                max(0, x1 - padx1),
                min(frame.shape[1], x2 + padx2),
            ]
        )

    box_array = np.array(boxes)
    if smooth:
        box_array = smooth_boxes(box_array)

    results = []
    for frame, (y1, y2, x1, x2) in zip(frames, box_array):
        y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
        results.append((frame[y1:y2, x1:x2], (y1, y2, x1, x2)))
    return results


def read_face_media(
    face_video: Path | None,
    face_image: Path | None,
    image_fps: float,
    resize_factor: int,
    rotate: bool,
) -> tuple[list[np.ndarray], float, bool]:
    if face_video:
        stream = cv2.VideoCapture(str(face_video))
        fps = stream.get(cv2.CAP_PROP_FPS) or image_fps
        frames = []
        while True:
            ok, frame = stream.read()
            if not ok:
                stream.release()
                break
            frames.append(frame)
        static = False
    elif face_image:
        frame = cv2.imread(str(face_image))
        if frame is None:
            raise RuntimeError(f"Could not read face image: {face_image}")
        fps = image_fps
        frames = [frame]
        static = True
    else:
        raise RuntimeError("Specify --face-video or --face-image.")

    processed = []
    for frame in frames:
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        processed.append(frame)

    if not processed:
        raise RuntimeError("No frames were read.")
    return processed, fps, static


def extract_wav(audio_path: Path, run_dir: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    wav_path = run_dir / "audio.wav"
    run_command(["ffmpeg", "-y", "-i", str(audio_path), "-strict", "-2", str(wav_path)])
    return wav_path


def record_microphone(duration: float, sample_rate: int, device: str | None, output_path: Path) -> Path:
    if duration <= 0:
        raise RuntimeError("--mic-duration must be greater than 0.")

    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError("sounddevice is not installed. Run `uv sync` first.") from exc

    input_device: int | str | None = int(device) if device and device.isdigit() else device
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration * sample_rate)
    print(f"Recording microphone for {duration:.1f}s at {sample_rate} Hz...")
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32", device=input_device)
    sd.wait()

    samples = np.clip(recording.reshape(-1), -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    print(f"Recorded audio: {output_path}")
    return output_path


def build_mel_chunks(audio_path: Path, fps: float, run_dir: Path) -> list[np.ndarray]:
    add_wav2lip_to_path()
    from Wav2Lip import audio

    wav_path = extract_wav(audio_path, run_dir)
    wav = audio.load_wav(str(wav_path), 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise RuntimeError("Mel spectrogram contains NaN values. Try a different audio file.")

    chunks = []
    multiplier = 80.0 / fps
    i = 0
    while True:
        start_idx = int(i * multiplier)
        if start_idx + MEL_STEP_SIZE > len(mel[0]):
            chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE :])
            break
        chunks.append(mel[:, start_idx : start_idx + MEL_STEP_SIZE])
        i += 1
    return chunks


def prepare_batch(img_batch, mel_batch, frame_batch, coords_batch):
    images = np.asarray(img_batch)
    mels = np.asarray(mel_batch)
    masked = images.copy()
    masked[:, IMG_SIZE // 2 :] = 0
    images = np.concatenate((masked, images), axis=3) / 255.0
    mels = np.reshape(mels, [len(mels), mels.shape[1], mels.shape[2], 1])
    return images, mels, frame_batch, coords_batch


def make_batches(
    frames: list[np.ndarray],
    mels: list[np.ndarray],
    face_results: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    batch_size: int,
    static: bool,
) -> Iterable[tuple[np.ndarray, np.ndarray, list[np.ndarray], list[tuple[int, int, int, int]]]]:
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    for i, mel in enumerate(mels):
        idx = 0 if static else i % len(frames)
        face, coords = face_results[idx]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        img_batch.append(face)
        mel_batch.append(mel)
        frame_batch.append(frames[idx].copy())
        coords_batch.append(coords)

        if len(img_batch) >= batch_size:
            yield prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if img_batch:
        yield prepare_batch(img_batch, mel_batch, frame_batch, coords_batch)


def run_wav2lip(
    face_video: Path | None,
    face_image: Path | None,
    audio_path: Path,
    output_path: Path,
    device: str,
    batch_size: int,
    face_det_batch_size: int,
    image_fps: float,
    resize_factor: int,
    rotate: bool,
    pads: tuple[int, int, int, int],
    smooth: bool,
) -> Path:
    started_at = time.time()
    run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_openvino_models()

    frames, fps, static = read_face_media(face_video, face_image, image_fps, resize_factor, rotate)
    print(f"Frames: {len(frames)} at {fps:.2f} FPS")

    mel_chunks = build_mel_chunks(audio_path, fps, run_dir)
    print(f"Mel chunks: {len(mel_chunks)}")

    if static:
        frames = frames * len(mel_chunks)
    else:
        frames = frames[: len(mel_chunks)]

    core = ov.Core()
    print(f"Compiling face detector on CPU: {FACE_DETECTION_MODEL}")
    face_detector = core.compile_model(str(FACE_DETECTION_MODEL), "CPU")
    face_source = [frames[0]] if static else frames
    face_results = face_detect(face_detector, face_source, face_det_batch_size, pads, smooth)
    if static:
        face_results = face_results * len(mel_chunks)

    print(f"Compiling Wav2Lip on {device}: {WAV2LIP_MODEL}")
    wav2lip_model = core.compile_model(str(WAV2LIP_MODEL), device)

    frame_h, frame_w = frames[0].shape[:-1]
    temp_video = run_dir / "result.avi"
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"DIVX"), fps, (frame_w, frame_h))

    batches = make_batches(frames, mel_chunks, face_results, batch_size, static)
    total_batches = int(np.ceil(float(len(mel_chunks)) / batch_size))
    for image_batch, mel_batch, frame_batch, coords_batch in tqdm(batches, total=total_batches, desc="Generating"):
        image_batch = np.transpose(image_batch, (0, 3, 1, 2)).astype(np.float32)
        mel_batch = np.transpose(mel_batch, (0, 3, 1, 2)).astype(np.float32)
        predictions = wav2lip_model({"audio_sequences": mel_batch, "face_sequences": image_batch})[wav2lip_model.outputs[0]]
        predictions = predictions.transpose(0, 2, 3, 1) * 255.0

        for pred, frame, coords in zip(predictions, frame_batch, coords_batch):
            y1, y2, x1, x2 = coords
            pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1))
            frame[y1:y2, x1:x2] = pred
            writer.write(frame)

    writer.release()
    wav_path = extract_wav(audio_path, run_dir)
    run_command(["ffmpeg", "-y", "-i", str(wav_path), "-i", str(temp_video), "-strict", "-2", "-q:v", "1", str(output_path)])

    print(f"Done in {time.time() - started_at:.1f}s")
    print(f"Output: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVINO Wav2Lip from the command line.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--face-video", type=Path, help="Input face video.")
    input_group.add_argument("--face-image", type=Path, help="Input face image. The image is treated as a still video.")
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--audio", type=Path, help="Input audio file.")
    audio_group.add_argument("--mic", action="store_true", help="Record audio directly from the microphone.")
    parser.add_argument("--output", type=Path, default=APP_ROOT / "runs" / "wav2lip_result.mp4", help="Output mp4 path.")
    parser.add_argument("--device", default="AUTO", help="OpenVINO device for Wav2Lip: AUTO, CPU, GPU.")
    parser.add_argument("--batch-size", type=int, default=16, help="Wav2Lip batch size.")
    parser.add_argument("--face-det-batch-size", type=int, default=4, help="Face detection batch size.")
    parser.add_argument("--image-fps", type=float, default=25.0, help="FPS used when --face-image is passed.")
    parser.add_argument("--mic-duration", type=float, default=5.0, help="Microphone recording duration in seconds.")
    parser.add_argument("--mic-sample-rate", type=int, default=16000, help="Microphone recording sample rate.")
    parser.add_argument("--mic-device", default=None, help="Optional sounddevice input device name or id.")
    parser.add_argument("--resize-factor", type=int, default=1, help="Downscale input frames by this factor.")
    parser.add_argument("--rotate", action="store_true", help="Rotate input frames clockwise.")
    parser.add_argument("--pad", type=int, nargs=4, default=[0, 10, 0, 0], metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"))
    parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing for detected face boxes.")
    parser.add_argument("--setup-only", action="store_true", help="Only download/convert Wav2Lip OpenVINO models.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.setup_only:
        ensure_openvino_models()
        return 0

    audio_path = args.audio
    if args.mic:
        audio_path = record_microphone(
            duration=args.mic_duration,
            sample_rate=args.mic_sample_rate,
            device=args.mic_device,
            output_path=RUNS_DIR / time.strftime("mic-%Y%m%d-%H%M%S.wav"),
        )
    if audio_path is None:
        raise SystemExit("--audio or --mic is required unless --setup-only is used.")
    if args.face_video is None and args.face_image is None:
        raise SystemExit("--face-video or --face-image is required unless --setup-only is used.")

    run_wav2lip(
        face_video=args.face_video,
        face_image=args.face_image,
        audio_path=audio_path,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        face_det_batch_size=args.face_det_batch_size,
        image_fps=args.image_fps,
        resize_factor=args.resize_factor,
        rotate=args.rotate,
        pads=tuple(args.pad),
        smooth=not args.no_smooth,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
