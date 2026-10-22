$ErrorActionPreference = 'Continue'
$sshArguments = @(
    '-N', '-T', '-F', 'NUL', '-i', '${SSH_KEY_FILE}',
    '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
    '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
    '-o', 'ServerAliveCountMax=5', '-o', 'LogLevel=ERROR',
    '-R', '127.0.0.1:27890:127.0.0.1:7897',
    '-p', '12304', 'private-user@remote-host.example'
)
while ($true) {
    & "$env:WINDIR\System32\OpenSSH\ssh.exe" @sshArguments
    Start-Sleep -Seconds 10
}
