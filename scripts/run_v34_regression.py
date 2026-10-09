#!/usr/bin/env python3
"""Generate and deliver v3.4 evaluation candidates with durable progress."""
from datetime import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import PurePosixPath

from publish_smb import component, require_mount, wait_for_uploads
from publish_evaluation import destination_parent
from run_evidence import ROOT, atomic_json, check, context, file_record, record, sha256, source_hash

FILMS = ['output_test', 'test_1', 'test_2']
LONG = 'SONE-846'
PY = str(ROOT / '.venv/bin/python')


def confirm_delivery(run_id, *, shorts_only):
    """Verify a new or resumed candidate before declaring it delivered or repairing its pointer."""
    wait_for_uploads()
    parent = destination_parent(require_mount(), 'v3.4')
    package_id = run_id + ('-shorts' if shorts_only else '')
    target = parent / package_id
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError(f'Candidate is not a completed directory: {target}')
    _, run = context()
    manifest_path = target / 'MANIFEST.json'
    if manifest_path.is_symlink():
        raise RuntimeError('Candidate manifest must not be a symlink')
    manifest_digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    scope = 'shorts-only' if shorts_only else 'full'
    expected = dict(status='EVALUATION / NOT ACCEPTED', version='v3.4', run_id=run_id, scope=scope)
    if (any(manifest.get(k) != v for k, v in expected.items())
            or manifest.get('formal_release_accepted') is not False
            or manifest.get('code_hash') != run['code_hash']):
        raise RuntimeError('Candidate manifest belongs to a different run, code or scope')
    rows = manifest.get('files')
    if not isinstance(rows, list) or not rows:
        raise RuntimeError('Candidate manifest has no file inventory')
    inventory = {}
    for row in rows:
        raw = row.get('path') if isinstance(row, dict) else None
        if not isinstance(raw, str):
            raise RuntimeError('Candidate contains an invalid file record')
        rel = PurePosixPath(raw)
        if rel.is_absolute() or not rel.parts or '..' in rel.parts or rel.as_posix() != raw or raw in inventory:
            raise RuntimeError(f'Candidate contains an unsafe or duplicate file: {raw}')
        p = target / raw
        if any((target.joinpath(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
            raise RuntimeError(f'Candidate file must not traverse symlinks: {raw}')
        if not p.is_file() or type(row.get('size')) is not int or p.stat().st_size != row['size'] or sha256(p) != row.get('sha256'):
            raise RuntimeError(f'Candidate file is missing or changed: {raw}')
        inventory[raw] = row
    if 'FINAL_FILMS.json' not in inventory:
        raise RuntimeError('Candidate has no verified final-film index')
    pointers = json.loads((target / 'FINAL_FILMS.json').read_text())
    if any(pointers.get(k) != v for k, v in expected.items()) or pointers.get('films') != manifest.get('films'):
        raise RuntimeError('Candidate final-film index disagrees with its manifest')
    films = pointers.get('films')
    names = FILMS if shorts_only else FILMS + [LONG]
    if (not isinstance(films, list) or len(films) != len(names)
            or {f.get('name') for f in films} != set(names)):
        raise RuntimeError('Candidate final-film index is incomplete')
    for film in films:
        wanted = f"videos/{film['name']}.mp4" if film['name'] in FILMS else f"videos/{LONG}_dubbed_full.mp4"
        entry = inventory.get(wanted)
        if (film.get('path') != wanted or not entry or not entry['size']
                or any(film.get(k) != entry[k] for k in ('size', 'sha256'))):
            raise RuntimeError(f"Candidate film lacks matching file evidence: {film.get('name')}")
    if pointers.get('entry') != '00_从这里开始.html' or pointers['entry'] not in inventory:
        raise RuntimeError('Candidate has no verified entry page')
    if sha256(manifest_path) != manifest_digest:
        raise RuntimeError('Candidate manifest changed during confirmation')
    context()
    # A resumed full package is confirmed below; do not briefly replace its
    # discovery pointer with the earlier short-only candidate.
    if shorts_only and (parent / run_id).exists():
        return target
    latest = {**pointers, 'directory': package_id, 'manifest': f'{package_id}/MANIFEST.json',
              'manifest_sha256': manifest_digest, 'entry': f"{package_id}/{pointers['entry']}",
              'films': [{**f, 'path': f"{package_id}/{f['path']}"} for f in films]}
    temp = parent / f'.LATEST_EVALUATION-{uuid.uuid4().hex[:12]}.json'
    destination_parent(require_mount(), 'v3.4')
    atomic_json(temp, latest)
    wait_for_uploads()
    destination_parent(require_mount(), 'v3.4')
    context()
    os.replace(temp, parent / 'LATEST_EVALUATION.json')
    wait_for_uploads()
    return target


def main():
    os.chdir(ROOT)
    require_mount()
    run_id = component(os.environ.get('V34_RUN_ID') or datetime.now().strftime('%Y%m%dT%H%M%S'))
    run_dir = ROOT / 'workspace/_release/v3.4/runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(V34_RUN_DIR=str(run_dir), V34_CODE_HASH=source_hash(), AI_MOVIE_UPLOAD='0')
    os.environ.pop('PROFILES', None)
    os.environ.setdefault('HF_HUB_OFFLINE', '0')
    run_path = run_dir / 'RUN.json'
    if not run_path.exists():
        atomic_json(run_path, dict(version='v3.4', run_id=run_id, started_at_ns=time.time_ns(),
                                  code_hash=source_hash(), films=FILMS, long=LONG,
                                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
        for folder in ('ai_movie', 'scripts'):
            for p in (ROOT / folder).rglob('*'):
                if p.suffix in ('.py', '.sh') and '__pycache__' not in p.parts:
                    dest = run_dir / 'source' / p.relative_to(ROOT)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, dest)
    context()

    def progress(stage, **extra):
        require_mount()
        atomic_json(run_dir / 'progress.json', dict(run_id=run_id, stage=stage,
                    updated_at=datetime.now().isoformat(), **extra))
        print(f'\n##### {datetime.now().isoformat()} v3.4 {stage} #####', flush=True)

    def command(args, *, quality=False, env=None):
        require_mount()
        context()
        result = subprocess.run(args, env={**os.environ, **(env or {})})
        context()
        if result.returncode not in ((0, 1) if quality else (0,)):
            raise RuntimeError(f'Command failed ({result.returncode}): {args}')
        return result.returncode

    def valid(name):
        try:
            return check(name)
        except (OSError, ValueError, KeyError):
            return False

    def evaluate(long):
        progress('evaluating_full' if long else 'evaluating_shorts')
        release = ROOT / 'workspace/_release/v3.4'
        (release / 'ACCEPTANCE.json').unlink(missing_ok=True)
        args = [PY, '-u', 'scripts/accept_release.py', 'v3.4', '--baseline', 'workspace/_archive_v3.3']
        if long:
            args += ['--long', long]
        rc = command(args, quality=True)
        data = json.loads((release / 'ACCEPTANCE.json').read_text())
        if not data.get('gates') or data.get('version') != 'v3.4' or data.get('passed') != (rc == 0):
            raise RuntimeError('Acceptance did not produce a consistent fresh report')
        report_dir = run_dir / 'reports'
        report_dir.mkdir(exist_ok=True)
        prefix = '' if long else 'shorts-'
        for p in release.iterdir():
            if p.is_file() and p.suffix in ('.json', '.md'):
                shutil.copyfile(p, report_dir / (prefix + p.name))
        atomic_json(run_dir / 'evaluation.json', dict(run_id=run_id, exit_code=rc,
                    report=file_record(report_dir / (prefix + 'ACCEPTANCE.json')), films=FILMS, long=long))
        return rc == 0

    try:
        if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'v3.4-dev':
            raise RuntimeError('Expected v3.4-dev branch')
        if not (ROOT / 'workspace/_archive_v3.3').is_dir():
            raise RuntimeError('Frozen v3.3 baseline is required')
        for name in FILMS:
            progress('short', film=name)
            if valid(name):
                print(f'{name}: verified current-run receipt; resume', flush=True)
                continue
            command(['bash', 'scripts/run_v3.sh', name])
            if not valid(name):
                raise RuntimeError(f'{name}: missing completion evidence')
        shorts_target = require_mount() / 'AI-Movie/evaluations/v3.4' / (run_id + '-shorts')
        if not shorts_target.exists():
            evaluate(None)
            progress('publishing_shorts')
            command([PY, '-u', 'scripts/publish_evaluation.py', 'v3.4', '--run-id', run_id, '--shorts-only'])
        confirm_delivery(run_id, shorts_only=True)
        progress('long', film=LONG)
        if not valid(LONG):
            command(['bash', 'scripts/run_long.sh', 'inputs/SONE-846-C.mp4', LONG],
                    env=dict(STRICT_RUN='1', RERUN=run_id, REBUILD_PROFILES='1', STALL_MIN='40'))
            record(LONG, long=True)
        full_report = ROOT / 'deliver' / f'{LONG}_full/ACCEPTANCE_FULL.json'
        full_report.unlink(missing_ok=True)
        command([PY, '-u', 'scripts/eval_long.py', LONG], quality=True)
        if not json.loads(full_report.read_text()).get('checks'):
            raise RuntimeError('Long-film evaluation did not produce a fresh report')
        accepted = evaluate(LONG)
        progress('publishing_full', accepted=accepted)
        target = require_mount() / 'AI-Movie/evaluations/v3.4' / run_id
        if not target.exists():
            command([PY, '-u', 'scripts/publish_evaluation.py', 'v3.4', '--run-id', run_id, '--long', LONG])
        target = confirm_delivery(run_id, shorts_only=False)
        if accepted and not (require_mount() / 'AI-Movie/releases/v3.4').exists():
            command([PY, '-u', 'scripts/publish_smb.py', 'v3.4', '--long', LONG])
        wait_for_uploads()
        context()
        progress('completed', accepted=accepted, evaluation_folder=str(target))
        return 0
    except Exception as exc:
        progress('failed', error=str(exc))
        print(f'V34_REGRESSION_FAILED: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
