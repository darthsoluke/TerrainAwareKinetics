param(
    [Parameter(Mandatory=$true)][string]$ExportDirectory,
    [int]$Port = 28939
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$folder = (Resolve-Path -LiteralPath $ExportDirectory).Path
$receipt = Get-Content -LiteralPath (Join-Path $folder 'export.json') -Raw | ConvertFrom-Json
if ($receipt.schema -ne 'motionbricks.experimental_bones_live.v1' -or -not $receipt.user_requested_trial) {
    throw 'Expected the user-requested experimental BONES export'
}
$weights = Join-Path $folder 'weights.pt'
$bundle = Join-Path $root 'out/a100_live_20261010/a100-terrain.pt'
$scene = Join-Path $root 'out/a100_live_20261010/terrain-world.npz'
$sceneReceipt = Get-Content -LiteralPath ([IO.Path]::ChangeExtension($scene, '.json')) -Raw | ConvertFrom-Json
foreach ($item in @(
    @{Path=$weights;Hash=$receipt.checkpoint_sha256},
    @{Path=$bundle;Hash=$receipt.source_bundle_sha256},
    @{Path=$scene;Hash=$sceneReceipt.scene_sha256},
    @{Path=(Join-Path (Split-Path $scene) $sceneReceipt.mesh_file);Hash=$sceneReceipt.mesh_sha256}
)) {
    if ((Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.Hash) {
        throw "Artifact identity mismatch: $($item.Path)"
    }
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is occupied; do not replace an unrelated service"
}
$env:PYTHONPATH = "$root;$root/out/vendor/kimodo"
$arguments = @('-u','-m','scripts.soma_a100_trial_server',
    '--bundle',$bundle,'--sha256',$receipt.source_bundle_sha256,
    '--scene',$scene,'--joint-checkpoint',$weights,'--port',"$Port",
    '--label','BONES + PARC trial','--execution','eager','--interactive',
    '--committed-prefetch','--start-paused','--trace-worker')
if (@($arguments | Where-Object { $_ -match '["\r\n]' }).Count) { throw 'Unsupported argument character' }
$quoted = ($arguments | ForEach-Object { '"'+$_+'"' }) -join ' '
$prefix = Join-Path $folder ('local-'+(Get-Date -Format yyyyMMdd-HHmmss))
$process = Start-Process -FilePath 'D:/.anaconda/envs/motionbricks/python.exe' -ArgumentList $quoted `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$prefix.stdout.log" -RedirectStandardError "$prefix.stderr.log"
$result = [ordered]@{Pid=$process.Id;Url="http://127.0.0.1:$Port/";Checkpoint=$weights;
    CheckpointSHA256=$receipt.checkpoint_sha256;Scene=$scene;LogPrefix=$prefix;
    UserRequestedExperimentalTrial=$true;QualityApproved=$false}
$result | ConvertTo-Json | Set-Content -LiteralPath "$prefix.json" -Encoding UTF8
[pscustomobject]$result
