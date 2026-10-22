$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$pythonPath = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) { throw "Create .venv and install the project using README.md first." }
& $pythonPath -m fedcontext demo --out runs/demo --config configs/demo.yaml --seeds 7 19 43
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
