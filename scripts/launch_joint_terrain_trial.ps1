param(
    [Parameter(Mandatory=$true)][string]$ExportDirectory,
    [string]$SourceBundle,
    [string]$ScenePath,
    [int]$Port = 28933,
    [ValidateSet('eager','cuda-graph')][string]$Execution = 'cuda-graph',
    [switch]$VerifyOnly,
    [switch]$UnreviewedTrial,
    [switch]$Interactive
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$exportRoot = (Resolve-Path -LiteralPath $ExportDirectory).Path
$receipt = Get-Content -LiteralPath (Join-Path $exportRoot 'export.json') -Raw | ConvertFrom-Json
if ($receipt.schema -ne 'motionbricks.joint_terrain_export.v1') { throw 'Unsupported export receipt' }
if ([IO.Path]::GetFileName($receipt.weights_file) -ne $receipt.weights_file) { throw 'Expected a checkpoint filename' }
$weightsPath = Join-Path $exportRoot $receipt.weights_file
if (-not $SourceBundle) { $SourceBundle = Join-Path $taskRoot 'out/a100_live_20261010/a100-terrain.pt' }
if (-not $ScenePath) { $ScenePath = Join-Path $taskRoot 'out/a100_live_20261010/flat.npz' }
$SourceBundle = (Resolve-Path -LiteralPath $SourceBundle).Path
$ScenePath = (Resolve-Path -LiteralPath $ScenePath).Path
if ((Get-FileHash -LiteralPath $SourceBundle -Algorithm SHA256).Hash.ToLowerInvariant() -ne $receipt.source_sha256) {
    throw 'Source bundle differs from the exported model dependency'
}
if ((Get-FileHash -LiteralPath $weightsPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $receipt.weights_sha256) {
    throw 'Joint checkpoint differs from the export receipt'
}
$summary = [ordered]@{Source=$SourceBundle;Checkpoint=$weightsPath;Scene=$ScenePath;
    ActualUpdates=$receipt.trained_updates;ReferenceBatchEquivalentUpdates=$receipt.reference_batch_equivalent_updates;
    ModelChecksPassed=$receipt.model_checks_passed;LocalControlVerified=$false}
if ($VerifyOnly) { [pscustomobject]$summary; return }
if (-not $receipt.model_checks_passed -and -not $UnreviewedTrial) { throw 'Unreviewed model requires the explicit UnreviewedTrial option' }
while (@(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).Count -gt 0) {
    $Port++
    if ($Port -gt 65535) { throw 'No available local port' }
}
$env:PYTHONPATH = "$taskRoot;$taskRoot/out/vendor/kimodo"
$trialLabel = if ($UnreviewedTrial) { 'V8 trial (not evaluated)' } else { 'V8-sample-budget' }
$serverArgs = @('-u','-m','scripts.soma_a100_trial_server','--bundle',$SourceBundle,
    '--sha256',$receipt.source_sha256,'--joint-checkpoint',$weightsPath,'--scene',$ScenePath,
    '--label',$trialLabel,'--port',"$Port",'--execution',$Execution)
if ($Interactive) { $serverArgs += '--interactive' }
if (@($serverArgs | Where-Object { $_ -match '["\r\n]' }).Count) { throw 'Unsupported character in server arguments' }
$quoted = ($serverArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
$prefix = Join-Path $exportRoot ("local-trial-" + (Get-Date -Format yyyyMMdd-HHmmss))
$process = Start-Process -FilePath 'D:\.anaconda\envs\motionbricks\python.exe' -ArgumentList $quoted `
    -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$prefix.stdout.log" -RedirectStandardError "$prefix.stderr.log"
$summary.Pid=$process.Id;$summary.Url="http://127.0.0.1:$Port/?mode=terrain";$summary.LogPrefix=$prefix
$summary | ConvertTo-Json | Set-Content -LiteralPath "$prefix.json" -Encoding UTF8
[pscustomobject]$summary
