"""Import the legacy project's source/docs into its dedicated public repository."""

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


EXTENSIONS = {'.py', '.md', '.json', '.yaml', '.yml', '.txt', '.toml', '.cfg',
              '.ini', '.js', '.mjs', '.html', '.css', '.sh', '.ps1', '.patch', '.license'}
EXCLUDED = {'__pycache__', '.pytest_cache', '.git', 'node_modules', '.venv'}
LINK = re.compile(r'(!?)\[([^\]\n]*)\]\((<[^>\n]+>|[^)\n]+)\)')
FORBIDDEN_CREDENTIALS = (
    re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{25,}\b'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(r'\bya29\.[A-Za-z0-9_-]{20,}'),
    re.compile(r'\bGOCSPX-[A-Za-z0-9_-]{12,}'),
    re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}'),
)
PUBLIC_REPLACEMENTS = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def scrub(text, source):
    # These are publication-time substitutions, never alternate runtime behavior.
    for private, public in PUBLIC_REPLACEMENTS.items():
        text = text.replace(private, public)
    text = text.replace(str(source), '${PROJECT_ROOT}').replace(source.as_posix(), '${PROJECT_ROOT}')
    text = text.replace(str(source.parent), '${UPSTREAM_ROOT}').replace(source.parent.as_posix(), '${UPSTREAM_ROOT}')
    text = re.sub(r'(?i)[A-Z]:[/\\]Users[/\\][^/\\\s"\'<>]+', '${USER_HOME}', text)
    text = re.sub(r'(?i)[A-Z]:[/\\][^\s"\'<>]*[/\\]\.ssh[/\\]id_[\w.]+', '${SSH_KEY_FILE}', text)
    text = re.sub(r'(?<![\w./\\])/(?:home|data-hdd|SSD[0-9]+|data[0-9]*)/[^/\s"\'<>]+', '${PRIVATE_STORAGE}', text)
    text = re.sub(r'https?://(?:drive|docs)\.google\.com/[^\s<>"\')]+', 'PRIVATE_ARTIFACT_URL', text)
    text = re.sub(r'\b[0-9]{8,}-[A-Za-z0-9_-]+\.apps\.googleusercontent\.com\b', 'GOOGLE_OAUTH_CLIENT_ID', text)

    def address(match):
        value = match.group(0)
        prefix = text[text.rfind('\n', 0, match.start()) + 1:match.start()]
        if re.search(r'''(?:version\s*=\s*|['"]nvidia[-_\w]+['"]\s*:\s*)['"]$''', prefix):
            return value
        try:
            ip = ipaddress.ip_address(value)
        except ValueError:
            return value
        return value if ip.is_loopback or value == '0.0.0.0' else 'remote-host.example'

    text = re.sub(r'(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])', address, text)
    return text


def file_map(source):
    result = {}
    for directory in ('motionbricks', 'scripts', 'configs', 'tests', 'web', 'patches', 'tools'):
        for path in sorted((source / directory).rglob('*')):
            if (path.is_file() and not path.is_symlink() and not EXCLUDED.intersection(path.parts)
                    and path.suffix.lower() in EXTENSIONS):
                result[path.resolve()] = path.relative_to(source)
    for path in sorted((source / 'docs').glob('*.md')):
        name = Path('docs/START_HERE.md') if path.name == 'project_overview_beginner_20261023.md' else Path('docs/records') / path.name
        result[path.resolve()] = name
    for name in ('setup.py', 'requirements-map-viewer.txt', 'requirements-reference-ik.txt'):
        result[(source / name).resolve()] = Path(name)
    result[(source / 'README.md').resolve()] = Path('docs/upstream/MotionBricks.md')
    result[(source.parent / 'LICENSE').resolve()] = Path('LICENSE')
    return result


def rewrite_links(text, original, destination, mapping, source, omitted):
    def replace(match):
        image, label, raw = match.groups()
        raw = raw.strip('<>')
        if raw.startswith('#'):
            return match.group(0)
        if raw.startswith(('http://', 'https://')):
            if urlsplit(raw).hostname in ('drive.google.com', 'docs.google.com'):
                return label + '（私有制品地址）'
            return match.group(0)
        if re.match(r'[A-Za-z][A-Za-z0-9+.-]+://', raw):
            return label + '（应用内链接）'
        path, separator, anchor = raw.partition('#')
        line = re.search(r':([0-9]+)$', path)
        if line:
            path, anchor = path[:line.start()], 'L' + line.group(1)
        path = unquote(path)
        if not path:
            return match.group(0)
        candidate = Path(path) if re.match(r'^[A-Za-z]:[/\\]', path) else original.parent / path
        candidate = candidate.resolve()
        if candidate in mapping:
            relative = os.path.relpath(mapping[candidate], destination.parent).replace('\\', '/')
            target = relative + ('#' + anchor if anchor else '')
            return f'{image}[{label}]({target})'
        omitted.append(dict(document=destination.as_posix(), reference=scrub(raw, source)))
        return label + ('（历史制品）' if image else '') + '：`' + scrub(raw, source) + '`'
    return LINK.sub(replace, text)


def main(args):
    if args.redactions:
        PUBLIC_REPLACEMENTS.update(json.loads(args.redactions.read_text(encoding='utf-8')))
    source, target = args.source.resolve(strict=True), args.target.resolve(strict=True)
    if source == target or source in target.parents or target in source.parents or not (target / '.git').exists():
        raise ValueError('Use independent source and destination Git checkouts')
    manifest_path = target / 'docs/development/source_snapshot.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': []}
    owned = {row['target']: row['published_sha256'] for row in previous['files']}
    mapping = file_map(source)
    records, omitted, pending = [], [], []
    for original, destination in mapping.items():
        data = original.read_bytes()
        if len(data) > 3 * 1024**2 or b'\0' in data:
            raise ValueError('Unexpected large/binary source file: ' + str(destination))
        text = data.decode('utf-8-sig')
        if original.suffix.lower() == '.md':
            text = rewrite_links(text, original, destination, mapping, source, omitted)
        public = scrub(text, source).replace('\r\n', '\n').encode('utf-8')
        if any(pattern.search(public.decode('utf-8')) for pattern in FORBIDDEN_CREDENTIALS):
            raise ValueError('Credential-shaped content requires inspection: ' + str(destination))
        output = target / destination
        if output.exists() and output.read_bytes() != public:
            if destination.as_posix() not in owned or digest(output.read_bytes()) != owned[destination.as_posix()]:
                raise ValueError('Destination has an independent edit: ' + str(destination))
        pending.append((output, public))
        records.append(dict(source=os.path.relpath(original, source).replace('\\', '/'),
                            target=destination.as_posix(), source_sha256=digest(data),
                            published_sha256=digest(public), publication_substitutions=public != data))
    for output, public in pending:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(public)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema='terrain-aware-kinetics.source-snapshot.v1',
        source_repository='https://github.com/NVlabs/GR00T-WholeBodyControl',
        base_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        source_state='working-tree source snapshot including local modifications',
        excludes='model weights, motion/scene assets, caches, private machine configuration', files=records)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (target / 'docs/development/private_artifact_references.json').write_text(
        json.dumps(omitted, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(source_files=len(records), substituted_files=sum(r['publication_substitutions'] for r in records),
                          external_artifact_references=len(omitted)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--redactions', type=Path, help='Private JSON object of literal publication replacements; keep outside Git')
    main(parser.parse_args())
