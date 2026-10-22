"""Install user-authorized private Drive transfer configs without logging tokens."""

import argparse
import configparser
import json
from pathlib import Path
import re
import subprocess


REMOTE_INSTALL = """import configparser,io,json,os,pathlib,sys
data=json.load(sys.stdin)
path=pathlib.Path(data['path'])
path.parent.mkdir(parents=True,exist_ok=True)
os.chmod(path.parent,0o700)
config=configparser.ConfigParser(interpolation=None)
config['mb-transfer']=data['remote']
with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as stream:
    config.write(stream)
os.chmod(path,0o600)
print('private_config_installed')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oauth-output', type=Path, required=True)
    parser.add_argument('--rclone', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--install-servers', action='store_true')
    args = parser.parse_args()
    output = args.oauth_output.read_text(encoding='utf-8-sig')
    match = re.search(r'\{\s*"access_token"\s*:', output)
    if match is None:
        raise ValueError('OAuth has not completed')
    token, _ = json.JSONDecoder().raw_decode(output[match.start():])
    if not token.get('access_token') or not token.get('refresh_token'):
        raise ValueError('OAuth did not return a refreshable token')
    config = configparser.ConfigParser(interpolation=None)
    remote = {'type': 'drive', 'token': json.dumps(token), 'scope': 'drive'}
    config['mb-transfer'] = remote
    args.config.parent.mkdir(parents=True, exist_ok=True)
    with args.config.open('w', encoding='utf-8') as stream:
        config.write(stream)
    prefix = [str(args.rclone), '--config', str(args.config), '--contimeout', '10s', '--timeout', '30s']
    folder = 'MotionBricks Transfer 20261010'
    subprocess.run(prefix + ['mkdir', 'mb-transfer:' + folder], check=True, capture_output=True)
    result = subprocess.run(prefix + ['lsjson', 'mb-transfer:', '--dirs-only', '--max-depth', '1'],
                            check=True, capture_output=True, text=True, encoding='utf-8')
    folder_id = next(item['ID'] for item in json.loads(result.stdout) if item['Name'] == folder)
    # All task copies default to this private task directory, not the Drive root.
    remote['root_folder_id'] = folder_id
    config['mb-transfer'] = remote
    with args.config.open('w', encoding='utf-8') as stream:
        config.write(stream)
    installed = []
    if args.install_servers:
        hosts = [
            ('a100', 'private-user@remote-host.example', 12304,
             '${PRIVATE_STORAGE}/motionbricks-20261010/drive-auth/rclone.conf', None),
            ('4090d', 'private-user@remote-host.example', 32297,
             '${PRIVATE_STORAGE}/projects/motionbricks-parc-training-20261009/drive-auth-20261010/rclone.conf', None),
            ('g3', 'private-user@remote-host.example', 12303,
             '${PRIVATE_STORAGE}/motionbricks-bones/drive-auth-20261010/rclone.conf', '[remote-host.example]:12303'),
        ]
        import shlex
        for name, host, port, path, alias in hosts:
            command = ['ssh', '-i', str(args.key), '-p', str(port), '-o', 'BatchMode=yes']
            if alias:
                command += ['-o', 'HostKeyAlias=' + alias]
            command += [host, 'python3 -c ' + shlex.quote(REMOTE_INSTALL)]
            subprocess.run(command, input=json.dumps({'path': path, 'remote': remote}),
                           text=True, encoding='utf-8', check=True, capture_output=True)
            installed.append(name)
    print(json.dumps({'local_private_folder_ready': True, 'installed': installed,
                      'public_sharing_enabled': False, 'credentials_logged': False}))


if __name__ == '__main__':
    main()
