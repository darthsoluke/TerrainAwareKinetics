param(
    [ValidateSet('A','B')][string]$Model = 'B',
    [ValidateSet('flat','long-terrain','terrain-world')][string]$Scene = 'flat',
    [int]$Port = 28930,
    [switch]$Interactive
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$artifactRoot = Join-Path $taskRoot 'out/a100_live_20261010'
$bundleName = if ($Model -eq 'A') { 'a100-pelvis.pt' } else { 'a100-terrain.pt' }
$bundlePath = Join-Path $artifactRoot $bundleName
$digest = if ($Model -eq 'A') {
    '1bc2312a5ee24c15cb0e175db2deafc76188d52a0fd7254d5b30abce6ced84cc'
} else {
    (Get-Content (Join-Path $artifactRoot 'a100-terrain.json') -Raw | ConvertFrom-Json).sha256
}
if ((Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
    throw 'Downloaded A100 artifact hash mismatch'
}
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $listeners) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if ($owner.CommandLine -notmatch 'scripts\.soma_a100_trial_server' -or
        $owner.CommandLine -notlike "*$artifactRoot*") {
        throw "Port $Port is occupied by another service; choose another port"
    }
    Stop-Process -Id $owner.ProcessId
}
$env:PYTHONPATH = "$taskRoot;$taskRoot/out/vendor/kimodo"
$taskArguments = @('-u','-m','scripts.soma_a100_trial_server','--bundle',$bundlePath,
    '--sha256',$digest,'--scene',(Join-Path $artifactRoot "$Scene.npz"),
    '--label',"A100-$Model",'--port',"$Port")
if ($Interactive) { $taskArguments += '--interactive' }
$logPrefix = "${Model}-${Scene}-$(Get-Date -Format yyyyMMdd-HHmmss)"
$process = Start-Process -FilePath 'D:\.anaconda\envs\motionbricks\python.exe' -ArgumentList $taskArguments `
    -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $artifactRoot "$logPrefix.stdout.log") `
    -RedirectStandardError (Join-Path $artifactRoot "$logPrefix.stderr.log")
$receipt = [pscustomobject]@{Pid=$process.Id;Model=$Model;Scene=$Scene;Sha256=$digest;
    Url="http://127.0.0.1:$Port/?mode=terrain";LogPrefix=$logPrefix;QualityApproved=$false}
$receiptName = if ($Interactive) { 'active_trial_controls.json' } else { 'active_trial.json' }
$receipt | ConvertTo-Json | Set-Content (Join-Path $artifactRoot $receiptName) -Encoding UTF8
$receipt
