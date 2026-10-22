"""Check the public source tree using only the Python standard library."""

import ast
import ipaddress
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!?\[[^\]\n]*\]\((<[^>\n]+>|[^)\n]+)\)')
IPV4 = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')
SECRETS = (
    re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{25,}\b'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(r'\bya29\.[A-Za-z0-9_-]{20,}'),
    re.compile(r'\bGOCSPX-[A-Za-z0-9_-]{12,}'),
    re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}'),
)
PAYLOADS = {'.pt', '.pth', '.ckpt', '.onnx', '.npz', '.npy', '.bvh', '.fbx',
            '.usd', '.usda', '.usdc', '.mp4', '.zip', '.tar', '.pem', '.key'}


def package_version(text, match):
    prefix = text[text.rfind('\n', 0, match.start()) + 1:match.start()]
    return re.search(r'''(?:version\s*=\s*|['"]nvidia[-_\w]+['"]\s*:\s*)['"]$''', prefix)


def markdown_links(text):
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif marker.group(1)[0] == fence:
                fence = None
            continue
        if fence is None:
            for match in LINK.finditer(line):
                yield number, match.group(1).strip('<>')


def main():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=ROOT,
    ).decode('utf-8').split('\0')
    paths = sorted({ROOT / name for name in names if name})
    errors = []
    counts = dict(files=0, python=0, json=0, markdown=0, local_links=0)
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            errors.append(f'{relative}: tracked file is missing')
            continue
        counts['files'] += 1
        if path.suffix.lower() in PAYLOADS or path.name.endswith('.tar.gz'):
            errors.append(f'{relative}: private artifact type in source tree')
        raw = path.read_bytes()
        if b'\0' in raw or len(raw) > 3 * 1024**2:
            errors.append(f'{relative}: binary or oversized source file')
            continue
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            errors.append(f'{relative}: source must be UTF-8')
            continue
        if any(pattern.search(text) for pattern in SECRETS):
            errors.append(f'{relative}: credential-shaped content')
        if re.search(r'https?://(?:drive|docs)\.google\.com/[^\s<>]+', text):
            errors.append(f'{relative}: private artifact sharing URL')
        for match in IPV4.finditer(text):
            if package_version(text, match):
                continue
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            if not address.is_loopback and str(address) != '0.0.0.0':
                errors.append(f'{relative}: non-local machine address')
                break
        try:
            if path.suffix == '.py':
                ast.parse(text, filename=relative)
                counts['python'] += 1
            elif path.suffix == '.json':
                json.loads(text)
                counts['json'] += 1
        except (SyntaxError, ValueError) as error:
            errors.append(f'{relative}: {type(error).__name__} at line {getattr(error, "lineno", "?")}')
        if path.suffix == '.md':
            counts['markdown'] += 1
            for number, target in markdown_links(text):
                if target.startswith('#') or urlsplit(target).scheme:
                    continue
                local = unquote(target.split('#', 1)[0])
                if not local:
                    continue
                counts['local_links'] += 1
                if local.startswith('/') or not (path.parent / local).exists():
                    errors.append(f'{relative}:{number}: missing local link {local}')
    print(json.dumps(counts, sort_keys=True))
    if errors:
        print('\n'.join(errors))
        raise SystemExit(1)
    print('Source, links, JSON and public-content checks passed.')


if __name__ == '__main__':
    main()
