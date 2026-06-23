
# MuseTalk 口パク生成アプリ

顔画像と音声をアップロードして、MuseTalk でリップシンク動画を生成する Gradio アプリです。

## セットアップ

1. MuseTalk をセットアップします。

```powershell
.\scripts\setup_musetalk.ps1
```

このスクリプトは公式リポジトリ https://github.com/TMElyralab/MuseTalk/tree/main の手順に沿って、`MuseTalk/` の clone、Python 3.10 venv、PyTorch 2.0.1 CUDA 11.8、MuseTalk requirements、MMLab、モデル重みの取得を行います。

clone だけ行う場合:

```powershell
.\scripts\setup_musetalk.ps1 -SkipDependencies -SkipWeights
```

モデル重みのダウンロードだけ後回しにする場合:

```powershell
.\scripts\setup_musetalk.ps1 -SkipWeights
```

FFmpeg が PATH にない場合は、アプリの `FFmpeg bin パス` に `ffmpeg.exe` が入った `bin` フォルダを指定してください。

2. このアプリの依存関係を入れます。

```powershell
uv sync
```

3. 起動します。

```powershell
uv run python app.py
```

ブラウザで表示された URL を開き、顔画像をアップロードします。「OpenVINO リアルタイム推論」タブではマイク録音を MuseTalk のリアルタイム推論に渡し、生成中のフレームをライブ表示します。生成完了後は完成動画も表示されます。

マイクが反応しない場合は、`http://127.0.0.1:7860` で開いていることと、ブラウザのマイク許可が有効になっていることを確認してください。マイク入力は「リアルタイムプレビュー」専用です。

## 設定

- `MUSETALK_DIR`: MuseTalk を別ディレクトリに置く場合に指定します。
- `FFmpeg bin パス`: `ffmpeg` が PATH にない Windows 環境で指定します。
- `Python 実行ファイル`: MuseTalk 用の Python 環境を別にしている場合、その `python.exe` を指定します。
- `補助デバイス`: VAE、Whisper、顔処理など OpenVINO UNet 以外の処理に使う PyTorch デバイスです。Intel GPU 環境では `xpu` を使います。
- `OpenVINO UNet`: 変換済みの `models/openvino/musetalkV15_unet.xml` を指定します。
- `OpenVINO デバイス`: `AUTO` / `GPU` / `CPU` を選べます。

## Intel GPU について

このアプリは MuseTalk v1.5 の UNet を OpenVINO Runtime で推論します。VAE、Whisper、顔処理などは引き続き PyTorch 側で動くため、補助デバイスとして `xpu` を使います。

この環境では既存の CUDA/MMLab 環境を上書きせず、Intel GPU 用の `.xpu-probe` 仮想環境を別に作っています。アプリは `.xpu-probe\Scripts\python.exe` と `実行デバイス: xpu` を既定で使います。

PyTorch XPU 版では `mmpose/mmcv` の Windows wheel が合わないため、DWPose の代わりに顔検出ベースの口元 bbox 推定へフォールバックしています。CUDA 版より切り抜き品質が落ちる可能性はあります。

## OpenVINO UNet

この環境では TMElyralab/MuseTalk の v1.5 UNet を OpenVINO IR に変換済みです。

```powershell
cd MuseTalk
..\.xpu-probe\Scripts\python.exe -m scripts.export_openvino_unet `
  --unet_config models\musetalkV15\musetalk.json `
  --unet_model_path models\musetalkV15\unet.pth `
  --output models\openvino\musetalkV15_unet.xml `
  --batch_size 1
```

アプリは OpenVINO UNet 推論だけを使います。Torch UNet バックエンドと v1 モデル選択は削除済みです。OpenVINO バックエンドは `Batch size: 1` 固定です。

出力と一時ファイルは `runs/` に保存されます。
