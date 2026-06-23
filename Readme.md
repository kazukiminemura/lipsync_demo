
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

ブラウザで表示された URL を開き、顔画像をアップロードします。音声ファイルを使う場合は「音声ファイル」タブ、マイクを使う場合は「マイク」タブで録音して生成します。

マイクが反応しない場合は、`http://127.0.0.1:7860` で開いていることと、ブラウザのマイク許可が有効になっていることを確認してください。マイク入力中はログ欄に秒数と音量が表示されます。

## 設定

- `MUSETALK_DIR`: MuseTalk を別ディレクトリに置く場合に指定します。
- `FFmpeg bin パス`: `ffmpeg` が PATH にない Windows 環境で指定します。
- `Python 実行ファイル`: MuseTalk 用の Python 環境を別にしている場合、その `python.exe` を指定します。

出力と一時ファイルは `runs/` に保存されます。
