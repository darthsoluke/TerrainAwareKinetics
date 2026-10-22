param([int]$Port = 28940)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (@(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).Count) {
    throw "Port $Port is occupied; no existing process will be stopped"
}
$artifacts = Join-Path $taskRoot 'out/a100_live_20261010/flat_recovery_20261013'
New-Item -ItemType Directory -Path $artifacts -Force | Out-Null
$stamp = Get-Date -Format yyyyMMdd-HHmmss
$env:PYTHONPATH = "$taskRoot;$taskRoot/out/vendor/kimodo"
$process = Start-Process -FilePath 'D:/.anaconda/envs/motionbricks/python.exe' `
    -ArgumentList @('-u','-m','scripts.soma_recovered_flat_server','--port',"$Port") `
    -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $artifacts "$stamp.stdout.log") `
    -RedirectStandardError (Join-Path $artifacts "$stamp.stderr.log")
$receipt = [pscustomobject]@{Pid=$process.Id;Url="http://127.0.0.1:$Port/";LogPrefix=$stamp;
    SourceSha256='0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d';
    NumTokens=8;RefreshFrames=24;PathRefinements=0;Execution='cuda-graph';StartPaused=$true;QualityApproved=$false}
$receipt | ConvertTo-Json | Set-Content (Join-Path $artifacts 'launch.json') -Encoding UTF8
$receipt
