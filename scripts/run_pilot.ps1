param(
    [string]$Batches = '26',
    [int]$MaxStudies = 30,
    [double]$MaxTotalGB = 10,
    [ValidateSet('train', 'validation', 'test')][string]$Split = 'validation',
    [string]$Config = 'config/batch26.yaml',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $python = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
    $pilotArgs = @('-m', 'mrrate_vlm', '--config', $Config, 'run', '--batches', $Batches,
                   '--source-batches', $Batches, '--max-studies', $MaxStudies,
                   '--max-total-gb', $MaxTotalGB, '--split', $Split)
    if ($DryRun) { $pilotArgs += '--dry-run' }
    & $python @pilotArgs
    if ($LASTEXITCODE -ne 0) { throw "Pilot failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
