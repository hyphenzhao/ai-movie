#!/usr/bin/env python3
"""Receipts for resumable v3.4 runs; completion binds actual media and code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    path = Path(path).resolve()
    st = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (st.st_size, st.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'File changed while hashing: {path}')
    return dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=digest)


def source_hash() -> str:
    paths = sorted(p for folder in ('ai_movie', 'scripts') for p in (ROOT / folder).rglob('*')
                   if p.suffix in ('.py', '.sh') and '__pycache__' not in p.parts)
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode() + b'\0' + p.read_bytes() + b'\0')
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def context() -> tuple[Path, dict]:
    run_dir = Path(os.environ['V34_RUN_DIR']).resolve()
    run = json.loads((run_dir / 'RUN.json').read_text())
    if run['code_hash'] != os.environ['V34_CODE_HASH'] or run['code_hash'] != source_hash():
        raise RuntimeError('Source code changed during the run; restart with a new run ID')
    return run_dir, run


def output(name: str) -> Path:
    validate_name(name)
    state = json.loads((ROOT / 'workspace' / name / 'state.json').read_text())
    value = (state.get('vc') or {}).get('video')
    if not value:
        raise RuntimeError(f'{name}: current state has no vc.video')
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f'{name}: current output is missing: {path}')
    return path


def validate_record(rec: dict) -> bool:
    path = Path(rec['path'])
    return path.is_file() and path.stat().st_size == rec['size'] and sha256(path) == rec['sha256']


def same_record(a: dict, b: dict) -> bool:
    # SMB may round mtime after a write is uploaded. Full content hashes bind
    # identity, so timestamp precision must not invalidate identical media.
    return (Path(a['path']).resolve() == Path(b['path']).resolve()
            and a['size'] == b['size'] and a['sha256'] == b['sha256'])


def validate_name(name: str) -> None:
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError(f'Invalid film name: {name!r}')


def min_speech_minutes(value=None) -> float:
    result = float(os.environ.get('MIN_SPEECH_MIN', '0.03') if value is None else value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid minimum speech threshold')
    return result


def plan_sources(name: str, plan: dict, minimum: float) -> tuple[dict, dict]:
    dubbed, passthrough = {}, {}
    for c in plan['chunks']:
        cn = f'{name}_p{int(c["index"]):02d}'
        if cn in dubbed or cn in passthrough:
            raise ValueError(f'Duplicate chunk in plan: {cn}')
        path = Path(c['file'])
        if not path.is_absolute():
            path = ROOT / path
        (passthrough if c['speech_minutes'] < minimum else dubbed)[cn] = path.resolve()
    if not dubbed and not passthrough:
        raise ValueError('Empty split plan')
    return dubbed, passthrough


def _profile_path(record: dict | None) -> Path | None:
    return Path(record['path']).resolve() if record else None


def _check(name: str, run_dir: Path, run: dict, *, expected_profiles: Path | None,
           seen: set[str]) -> bool:
    validate_name(name)
    if name in seen:
        return False
    seen = seen | {name}
    receipt = json.loads((run_dir / 'receipts' / f'{name}.json').read_text())
    if receipt['status'] != 'completed' or receipt['run_id'] != run['run_id'] or receipt['code_hash'] != run['code_hash']:
        return False
    if not validate_record(receipt['video']):
        return False
    if receipt.get('state'):
        if (Path(receipt['state']['path']).resolve() != (ROOT / 'workspace' / name / 'state.json').resolve()
                or not validate_record(receipt['state']) or output(name) != Path(receipt['video']['path'])):
            return False
    if receipt.get('profiles') and not validate_record(receipt['profiles']):
        return False
    actual_profiles = _profile_path(receipt.get('profiles'))
    if actual_profiles != expected_profiles:
        return False
    if receipt.get('state'):
        return receipt.get('kind') == 'clip' and not receipt.get('chunks') and not receipt.get('passthrough')
    if receipt.get('kind') != 'long':
        return False
    if Path(receipt['video']['path']).resolve() != (ROOT / 'deliver' / f'{name}_dubbed_full.mp4').resolve():
        return False
    split = ROOT / 'workspace' / name / '_split'
    if (Path(receipt['plan']['path']).resolve() != (split / 'plan.json').resolve()
            or not validate_record(receipt['plan'])):
        return False
    minimum = min_speech_minutes(receipt['min_speech_min'])
    if 'MIN_SPEECH_MIN' in os.environ and minimum != min_speech_minutes():
        return False
    plan = json.loads((split / 'plan.json').read_text())
    dubbed, passthrough = plan_sources(name, plan, minimum)
    if set(receipt.get('chunks', {})) != set(dubbed) or set(receipt.get('passthrough', {})) != set(passthrough):
        return False
    for chunk, digest in receipt['chunks'].items():
        if sha256(run_dir / 'receipts' / f'{chunk}.json') != digest or not _check(
                chunk, run_dir, run, expected_profiles=actual_profiles, seen=seen):
            return False
    for chunk, path in passthrough.items():
        rec = receipt['passthrough'][chunk]
        if Path(rec['path']).resolve() != path or not validate_record(rec):
            return False
    return validate_concat(name, receipt, run_dir)


def validate_concat(name: str, receipt: dict, run_dir: Path) -> bool:
    split = ROOT / 'workspace' / name / '_split'
    rec = receipt['concat']
    if Path(rec['path']).resolve() != (split / 'concat_manifest.json').resolve() or not validate_record(rec):
        return False
    manifest = json.loads(Path(rec['path']).read_text())
    if (manifest.get('version') != 1 or manifest.get('film') != name
            or manifest.get('min_speech_min') != receipt['min_speech_min']
            or not same_record(manifest['video'], receipt['video'])
            or not same_record(manifest['plan'], receipt['plan'])):
        return False
    plan = json.loads(Path(receipt['plan']['path']).read_text())
    expected_names = [f'{name}_p{int(c["index"]):02d}' for c in plan['chunks']]
    if [p['name'] for p in manifest['parts']] != expected_names:
        return False
    for part in manifest['parts']:
        cn = part['name']
        if cn in receipt['chunks']:
            chunk = json.loads((run_dir / 'receipts' / f'{cn}.json').read_text())
            expected = chunk['video']
            if part['mode'] != 'dub':
                return False
        else:
            expected = receipt['passthrough'][cn]
            if part['mode'] != 'pass':
                return False
        if not same_record(part['source'], expected):
            return False
        # The transport streams are the exact files ffmpeg concatenated; the
        # processed input identifies any cached trim or loudness-adjusted pass.
        if not all(validate_record(part[k]) for k in ('source', 'processed', 'transport')):
            return False
    return True


def check(name: str) -> bool:
    try:
        run_dir, run = context()
        validate_name(name)
        receipt = json.loads((run_dir / 'receipts' / f'{name}.json').read_text())
        profile = os.environ.get('PROFILES')
        expected = Path(profile).resolve() if profile else None
        # A publisher checks a finished long film outside the child shell that
        # exported PROFILES. Its recursive chunk checks use the recorded profile.
        if receipt.get('kind') == 'long' and not profile:
            expected = _profile_path(receipt.get('profiles'))
        return _check(name, run_dir, run, expected_profiles=expected, seen=set())
    except (OSError, KeyError, ValueError, TypeError, RuntimeError):
        return False


def record(name: str, *, long: bool = False) -> None:
    validate_name(name)
    run_dir, run = context()
    profile = os.environ.get('PROFILES')
    if long and not profile:
        default = ROOT / 'workspace' / name / 'profiles.json'
        profile = str(default) if default.is_file() else None
    rec = dict(kind='long' if long else 'clip', status='completed', run_id=run['run_id'], code_hash=run['code_hash'],
               completed_at_ns=time.time_ns(), state=None, profiles=file_record(Path(profile)) if profile else None)
    if long:
        split = ROOT / 'workspace' / name / '_split'
        plan = json.loads((split / 'plan.json').read_text())
        minimum = min_speech_minutes()
        dubbed, passthrough = plan_sources(name, plan, minimum)
        chunks = {}
        for cn in dubbed:
            if not _check(cn, run_dir, run, expected_profiles=_profile_path(rec['profiles']), seen={name}):
                raise RuntimeError(f'Invalid chunk receipt: {cn}')
            chunks[cn] = sha256(run_dir / 'receipts' / f'{cn}.json')
        rec.update(chunks=chunks, min_speech_min=minimum,
                   passthrough={cn: file_record(path) for cn, path in passthrough.items()},
                   plan=file_record(split / 'plan.json'), concat=file_record(split / 'concat_manifest.json'))
        video = ROOT / 'deliver' / f'{name}_dubbed_full.mp4'
    else:
        video = output(name)
        rec['state'] = file_record(ROOT / 'workspace' / name / 'state.json')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)], timeout=120))
    if not any(s['codec_type'] == 'video' for s in probe['streams']) or not any(s['codec_type'] == 'audio' for s in probe['streams']) or float(probe['format'].get('duration', 0)) <= 0:
        raise RuntimeError(f'Invalid final video: {video}')
    rec['video'] = file_record(video)
    rec['duration'] = float(probe['format']['duration'])
    if long and not validate_concat(name, rec, run_dir):
        raise RuntimeError(f'Concat manifest does not prove the current complete film: {name}')
    atomic_json(run_dir / 'receipts' / f'{name}.json', rec)
    print(f'Recorded {name}: {video}', flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('hash', 'check', 'record', 'output'))
    p.add_argument('name', nargs='?')
    p.add_argument('--long', action='store_true')
    a = p.parse_args()
    try:
        if a.action == 'hash':
            print(source_hash())
        elif a.action == 'output':
            print(output(a.name))
        elif a.action == 'check':
            return 0 if check(a.name) else 1
        else:
            record(a.name, long=a.long)
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'{a.action}: {exc}', file=__import__('sys').stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
