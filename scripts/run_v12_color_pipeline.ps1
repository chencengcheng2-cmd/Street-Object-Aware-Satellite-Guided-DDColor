$ErrorActionPreference = "Stop"

Set-Location "C:\Users\31133\Desktop\satellite_guided_ddcolor"
$env:TORCH_CUDA_ARCH_LIST = "12.0"

$python = ".\.venv\Scripts\python.exe"
$logDir = "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$exportStdout = Join-Path $logDir "dino_semantic_v12_export.stdout.log"
$exportStderr = Join-Path $logDir "dino_semantic_v12_export.stderr.log"
$trainStdout = Join-Path $logDir "street_object_aware_v12_dino_semantic.stdout.log"
$trainStderr = Join-Path $logDir "street_object_aware_v12_dino_semantic.stderr.log"

& $python -u scripts\export_dino_semantic_predictions_v12.py `
  --config configs\dino_semantic_distill_v12.yaml `
  --checkpoint checkpoints\dino_semantic_distill_v12\best.pth `
  --street_dirname street_dino_semantic_v12 `
  --satellite_dirname overhead_satellite_dino_semantic_v12 `
  --splits train val `
  --batch_size 8 `
  --num_workers 2 `
  --use_amp `
  > $exportStdout 2> $exportStderr

if ($LASTEXITCODE -ne 0) {
  throw "v12 DINO semantic export failed with exit code $LASTEXITCODE. See $exportStderr"
}

& $python -u train.py `
  --config configs\street_object_aware_v12_dino_semantic.yaml `
  --exp_name street_object_aware_v12_dino_semantic `
  > $trainStdout 2> $trainStderr

if ($LASTEXITCODE -ne 0) {
  throw "v12 color training failed with exit code $LASTEXITCODE. See $trainStderr"
}
