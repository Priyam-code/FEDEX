param(
    [string]$FinalRun = 'runs/federated_final_v1',
    [string]$ClientPattern = 'runs/rsu_*_v1/client_rounds.csv',
    [string]$CloudRun = 'runs/cloud_v1',
    [string]$Out = 'results_for_figures.zip'
)
$ErrorActionPreference = 'Stop'
if (Test-Path $Out) { throw "Output already exists: $Out. Choose another -Out name." }
if (!(Test-Path (Join-Path $FinalRun 'evaluation'))) { throw 'Evaluation folder missing. Correct -FinalRun.' }
$stage = Join-Path ([System.IO.Path]::GetTempPath()) ('fedcontext_figures_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    $ev = New-Item -ItemType Directory -Path (Join-Path $stage 'evaluation')
    # Explicit result-file allowlist: no models, raw dataset, credentials or auth manifests.
    $names = @('predictions.csv.gz','predictions.csv','metrics.csv','per_scenario_attack.csv','coverage.json','latency.json','episode_delays.csv','episode_summary.json')
    foreach ($name in $names) {
        $p = Join-Path (Join-Path $FinalRun 'evaluation') $name
        if (Test-Path $p) { Copy-Item $p $ev.FullName }
    }
    $cal = Join-Path $FinalRun 'calibration.json'
    if (Test-Path $cal) { Copy-Item $cal $stage }
    $clientFiles = @(Get-ChildItem $ClientPattern -ErrorAction SilentlyContinue)
    foreach ($f in $clientFiles) {
        $dest = Join-Path (Join-Path $stage 'clients') $f.Directory.Name
        New-Item -ItemType Directory -Path $dest -Force | Out-Null
        Copy-Item $f.FullName $dest
    }
    $cloud = New-Item -ItemType Directory -Path (Join-Path $stage 'cloud')
    foreach ($name in @('communication.csv','training.json')) {
        $p = Join-Path $CloudRun $name
        if (Test-Path $p) { Copy-Item $p $cloud.FullName }
    }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $Out
    Write-Host "Created $Out with $($clientFiles.Count) client logs. Upload this ZIP for measured figure generation and interpretation."
} finally {
    Remove-Item -Recurse -Force $stage
}
