# OpenVINO Wav2Lip CLI

OpenVINO 版 Wav2Lip をコマンドラインから実行する最小デモです。

初回実行時に `vendor/Wav2Lip` を clone し、Wav2Lip と face detector の重みを取得して `models/wav2lip/` に OpenVINO IR を生成します。生成物、重み、入力サンプル、出力動画は Git 管理対象外です。

## セットアップ

```powershell
uv sync
uv run python app.py --setup-only
```

FFmpeg が PATH に入っている必要があります。

## 実行

動画を入力にする場合:

```powershell
uv run python app.py `
  --face-video path\to\face.mp4 `
  --audio path\to\voice.wav `
  --output runs\result.mp4
```

静止画を入力にする場合:

```powershell
uv run python app.py `
  --face-image path\to\face.png `
  --audio path\to\voice.wav `
  --output runs\result.mp4 `
  --image-fps 25
```

マイクから直接録音する場合:

```powershell
uv run python app.py `
  --face-video path\to\face.mp4 `
  --mic `
  --mic-duration 5 `
  --output runs\result.mp4
```

Intel GPU / OpenVINO AUTO を使う場合:

```powershell
uv run python app.py --face-video path\to\face.mp4 --audio path\to\voice.wav --device AUTO
```

## 主なオプション

- `--device`: Wav2Lip の OpenVINO 推論デバイス。`AUTO`, `CPU`, `GPU`
- `--batch-size`: Wav2Lip 推論 batch size
- `--face-det-batch-size`: face detector batch size
- `--mic`: マイクから録音して音声入力に使います
- `--mic-duration`: マイク録音秒数
- `--mic-device`: `sounddevice` の入力デバイス名または ID
- `--resize-factor`: 入力フレームを縮小して処理を軽くします
- `--pad TOP BOTTOM LEFT RIGHT`: 検出した顔 bbox の余白
- `--no-smooth`: 顔 bbox の平滑化を無効化

出力と一時ファイルは `runs/` に保存されます。
