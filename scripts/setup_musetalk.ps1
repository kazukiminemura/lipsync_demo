param(
    [string]$MuseTalkDir = "",
    [string]$PythonVersion = "3.10",
    [switch]$SkipDependencies,
    [switch]$SkipWeights,
    [switch]$ForceWeights
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Action
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($MuseTalkDir)) {
    $MuseTalkDir = Join-Path $ProjectRoot "MuseTalk"
}

$MuseTalkDir = [System.IO.Path]::GetFullPath($MuseTalkDir)
$VenvDir = Join-Path $MuseTalkDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

Invoke-Step "Check tools" {
    if (-not (Test-Command git)) {
        throw "git was not found. Install Git for Windows and add it to PATH."
    }
    if (-not (Test-Command uv)) {
        throw "uv was not found. Install uv before running this script."
    }
}

Invoke-Step "Clone or update MuseTalk" {
    if (-not (Test-Path $MuseTalkDir)) {
        git clone https://github.com/TMElyralab/MuseTalk.git $MuseTalkDir
    } elseif (Test-Path (Join-Path $MuseTalkDir ".git")) {
        git -C $MuseTalkDir pull --ff-only
    } else {
        throw "$MuseTalkDir exists but is not a Git repository. Pass a different -MuseTalkDir."
    }
}

if (-not $SkipDependencies) {
    Invoke-Step "Create Python $PythonVersion virtual environment" {
        if (-not (Test-Path $PythonExe)) {
            uv python install $PythonVersion
            uv venv $VenvDir --python $PythonVersion
        }
        & $PythonExe -m pip install --upgrade pip setuptools wheel
    }

    Invoke-Step "Install PyTorch 2.0.1 CUDA 11.8" {
        uv pip install `
            --python $PythonExe `
            torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 `
            --index-url https://download.pytorch.org/whl/cu118
    }

    Invoke-Step "Install MuseTalk requirements" {
        uv pip install --python $PythonExe -r (Join-Path $MuseTalkDir "requirements.txt")
    }

    Invoke-Step "Install MMLab packages" {
        uv pip install --python $PythonExe --no-cache-dir -U openmim
        & $PythonExe -m mim install mmengine
        & $PythonExe -m mim install "mmcv==2.0.1"
        & $PythonExe -m mim install "mmdet==3.1.0"
        & $PythonExe -m mim install "mmpose==1.1.0"
    }
}

if (-not $SkipWeights) {
    Invoke-Step "Download model weights" {
        $WeightsExist = (Test-Path (Join-Path $MuseTalkDir "models\musetalkV15\unet.pth")) -and
            (Test-Path (Join-Path $MuseTalkDir "models\whisper\pytorch_model.bin"))

        if ($WeightsExist -and -not $ForceWeights) {
            Write-Host "Model weights already exist. Pass -ForceWeights to download them again."
        } else {
            $DownloadScript = Join-Path $MuseTalkDir "download_weights.bat"
            if (-not (Test-Path $DownloadScript)) {
                throw "download_weights.bat was not found: $DownloadScript"
            }
            Push-Location $MuseTalkDir
            try {
                & $DownloadScript
            } finally {
                Pop-Location
            }
        }
    }
}

Invoke-Step "Check FFmpeg" {
    if (Test-Command ffmpeg) {
        ffmpeg -version | Select-Object -First 1
    } else {
        Write-Warning "ffmpeg was not found on PATH. In the app, set 'FFmpeg bin path' to the folder containing ffmpeg.exe."
    }
}

Write-Host ""
Write-Host "MuseTalk setup finished." -ForegroundColor Green
Write-Host "MuseTalk directory: $MuseTalkDir"
Write-Host "MuseTalk Python:    $PythonExe"
Write-Host ""
Write-Host "App setup:"
Write-Host "  uv sync"
Write-Host "  uv run python app.py"
