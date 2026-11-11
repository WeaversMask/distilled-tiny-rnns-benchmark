"""Verify saved Notebook 08 and 07 artifacts without calling the training runners."""
import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

import torch

from inventory import NAMES, snapshot

CORE = 'b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372'
GRID = 'a4582314a2b34289883f526366298f6604bb4c1e327856d068ee91e2c494c9df'
FROZEN = '1e8762df6fc01919dd3cfc432e96b78ed1d3fe1a25c26859edd9674cd501821e'
HISTORICAL_SOURCE = Path(__file__).with_name('historical-source.json')
PACKAGES = ('torch', 'numpy', 'matplotlib')
BASELINE_SNAPSHOT = {'summary.json', 'protocol.json', 'verification.json'}
TOL = 1e-10


class Audit:
    def __init__(self):
        self.checks = 0
        self.issues = []

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.issues.append(str(label))
        return bool(condition)

    def equal(self, actual, expected, label):
        self.check(actual == expected, label)

    def close(self, actual, expected, label):
        if expected is None:
            self.equal(actual, None, label)
        else:
            self.check(finite_seconds(actual) and abs(actual - expected) <= TOL, label)

    def hashes(self, listed, expected, actual, label):
        self.check(set(listed) == set(expected), f'{label} list differs, missing={sorted(set(expected)-set(listed))}, extra={sorted(set(listed)-set(expected))}')
        for name, digest in listed.items():
            path = PurePosixPath(name)
            safe = not path.is_absolute() and '..' not in path.parts and path.as_posix() == name and name != '.' and '\\' not in name
            self.check(safe, f'{label} unsafe path {name!r}')
            self.check(bool(re.fullmatch('[0-9a-f]{64}', digest)), f'{label} malformed SHA-256 for {name}')
            self.check(safe and actual.get(name) == digest, f'{label} hash mismatch or missing {name}')


def finite_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def read(path):
    def unique_keys(pairs):
        result = {}
        for key, val in pairs:
            if key in result:
                raise ValueError(f'{path} contains duplicate key {key}')
            result[key] = val
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=unique_keys)


def provenance(audit, source, core_hash, grid_hash):
    historical = read(HISTORICAL_SOURCE)
    runtime_files = [
        path
        for folder in ('main_study', 'foundation', 'src/distilled_rnns', 'grid_study')
        for path in (source / folder).glob('*.py')
    ]
    runtime_files += [source / 'configs' / name for name in (
        'main_study_v1.json', 'grid_baseline_reference_v1.json', 'rarity_tranche.json',
        'rarity_smoke.json', 'main_study_first_tranche.json', 'main_study_smoke.json',
    )]
    actual_hashes = {path.relative_to(source).as_posix(): sha256(path.read_bytes()).hexdigest()
                     for path in runtime_files}
    audit.hashes(historical['runtime_file_sha256'], set(actual_hashes), actual_hashes,
                 'public checkout vs original runtime file hashes')

    binding = read(source / 'configs/colab_rarity_source.json')
    commit = binding.get('commit', '')
    audit.check(binding.get('ready') is True, 'public notebook source binding is ready')
    audit.check(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit) is not None
                and commit != historical['runtime_commit'], 'public runtime commit pin')
    audit.equal(binding['source_sha256'], GRID, 'public binding grid source hash')
    audit.equal(binding['protocol_sha256'], FROZEN, 'public binding frozen design hash')
    notebook_path = source / 'notebooks/08_colab_rarity.ipynb'
    notebook = read(notebook_path)
    notebook_source = {line for cell in notebook['cells'] if cell['cell_type'] == 'code'
                       for line in ''.join(cell['source']).splitlines()}
    for variable, key in (('GITHUB_REPOSITORY', 'repository'), ('PINNED_COMMIT', 'commit'),
                          ('EXPECTED_SOURCE_SHA256', 'source_sha256'),
                          ('EXPECTED_PROTOCOL_SHA256', 'protocol_sha256')):
        audit.check(f'{variable} = {json.dumps(binding[key])}' in notebook_source,
                    f'public notebook source binding {variable}')

    audit.equal(core_hash, CORE, 'computed core source SHA-256')
    audit.equal(grid_hash, GRID, 'computed grid source SHA-256')
    audit.equal(sha256((source / 'configs/main_study_v1.json').read_bytes()).hexdigest(),
                FROZEN, 'computed frozen design SHA-256')
    return {
        'historical_runtime_commit': historical['runtime_commit'],
        'historical_notebook_release_commit': historical['notebook_release_commit'],
        'historical_notebook_sha256': historical['notebook_sha256'],
        'historical_launcher_binding_sha256': historical['launcher_binding_sha256'],
        'runtime_file_comparisons': len(actual_hashes),
        'public_repository': binding['repository'],
        'public_runtime_commit': commit,
        'public_notebook_sha256': sha256(notebook_path.read_bytes()).hexdigest(),
        'core_source_sha256': CORE, 'grid_source_sha256': GRID, 'frozen_design_sha256': FROZEN,
    }


def coverage(audit, entries, label, *, outer=False):
    audit.check(bool(entries), f'{label} entries present')
    known, dates = [], []
    statuses = Counter()
    for entry in entries:
        date = datetime.fromisoformat(entry['started_utc'])
        dates.append(date)
        status = entry['status']
        statuses[status] += 1
        elapsed = entry.get('wall_seconds')
        allowed = ('complete', 'interrupted') if outer else ('complete', 'incomplete', 'interrupted')
        audit.check(status in allowed, f'{label} {date} valid status {status}')
        if elapsed is None:
            audit.equal(status, 'interrupted', f'{label} {date} unknown elapsed time')
        elif audit.check(finite_seconds(elapsed), f'{label} {date} finite elapsed time'):
            known.append(elapsed)
    audit.check(dates == sorted(dates) and len(set(dates)) == len(dates) and all(date.tzinfo is not None for date in dates), f'{label} ordered and unique UTC timestamps')
    audit.equal(entries[-1]['status'], 'complete', f'{label} final entry complete')
    return {'known_seconds': sum(known), 'all_seconds': sum(known) if len(known) == len(entries) else None,
            'recorded': len(known), 'total': len(entries), 'statuses': dict(statuses)}


def check_coverage(audit, timing, wall, kind, label):
    key = 'session' if kind == 'sessions' else 'runner'
    prefix = 'session' if kind == 'sessions' else 'invocation'
    audit.close(timing[f'known_{key}_wall_seconds'], wall['known_seconds'], f'{label} known wall time')
    audit.close(timing[f'{key}_wall_seconds'], wall['all_seconds'], f'{label} total wall time')
    audit.equal(timing[f'{prefix}_wall_time_coverage'], {f'recorded_{kind}': wall['recorded'], f'total_{kind}': wall['total']}, f'{label} wall time coverage')


def local_map(files, prefix):
    return {key[len(prefix)+1:]: value for key, value in files.items() if key.startswith(prefix + '/')}


def population(audit, child, files, item, protocol_sha, config, label):
    issues_before = len(audit.issues)
    manifest = read(child / 'manifest.json')
    report = read(child / 'report.json')
    training = read(child / 'training.json')
    audit.equal(manifest.get('status'), 'complete', f'{label} manifest complete')
    audit.equal(manifest['identity'], {**item, 'protocol_sha256': protocol_sha}, f'{label} population protocol binding')
    audit.hashes(manifest['immutable_file_sha256'], {'data.npz', 'soft_training_logits.npz'}, files, f'{label} immutable')
    audit.hashes(manifest['completed_file_sha256'], set(POPULATION_FILES), files, f'{label} completion')
    audit.equal({key: report[key] for key in ('kind', 'seed', 'status')}, {**item, 'status': 'complete'}, f'{label} report complete')
    audit.equal(report['phases'], training, f'{label} report/training exact correspondence')
    audit.equal(set(training), set(ARMS), f'{label} four phase reports')

    sessions = manifest['sessions']
    session_wall = coverage(audit, sessions, f'{label} sessions')
    check_coverage(audit, report['timing'], session_wall, 'sessions', label)
    environment_files = set()
    latest = None
    session_metadata = []
    for i, session in enumerate(sessions):
        session_label = f'{label} session {i+1}'
        environment = session['environment']
        package = session['package_inventory']
        audit.equal(package['status'], 'recorded', f'{session_label} recorded package status')
        expected_path = f"environments/packages-{package['sha256']}.txt"
        audit.equal(package['path'], expected_path, f'{session_label} content-addressed package inventory path')
        audit.hashes({package['path']: package['sha256']}, {expected_path}, files, session_label)
        environment_files.add(expected_path)
        latest = package['sha256']
        audit.equal(set(environment), {'python','platform','packages','device','gpu','cuda','cudnn','torch_threads'}, f'{session_label} environment metadata coverage')
        audit.check(all(isinstance(environment.get(key), str) and environment[key] for key in ('python','platform','device')), f'{session_label} valid environment strings')
        audit.check(type(environment['torch_threads']) is int and environment['torch_threads'] > 0, f'{session_label} valid thread metadata')
        if environment['device'].split(':', 1)[0] == 'cuda':
            audit.check(all(isinstance(environment.get(key), str) and environment[key] for key in ('gpu','cuda')), f'{session_label} valid CUDA metadata')
            audit.check(type(environment['cudnn']) is int and environment['cudnn'] > 0, f'{session_label} valid cuDNN metadata')
        if expected_path in files and re.fullmatch('[0-9a-f]{64}', latest):
            lines = (child / expected_path).read_text().splitlines()
            audit.check(lines == sorted(set(lines)) and len(lines) > len(PACKAGES) and all(re.fullmatch(r'[^=\s]+==[^=\s]+', line) for line in lines),
                        f'{session_label} normalized package inventory')
            audit.equal(set(environment['packages']), set(PACKAGES), f'{session_label} key package metadata coverage')
            audit.check(all(f'{name}=={environment["packages"][name]}' in lines for name in PACKAGES), f'{session_label} package versions agree with inventory')
        if session['status'] == 'incomplete':
            audit.check(session.get('phase') in ARMS, f'{session_label} timed pause identifies its arm')
        session_metadata.append({'status': session['status'], 'phase': session.get('phase'), 'device': environment['device'], 'gpu': environment['gpu'],
                                 'cuda': environment['cuda'], 'cudnn': environment['cudnn'], 'python': environment['python'],
                                 'platform': environment['platform'], 'torch_threads': environment['torch_threads'],
                                 'packages': environment['packages'], 'package_inventory_sha256': latest,
                                 'started_utc': session['started_utc'], 'wall_seconds': session.get('wall_seconds')})
    audit.equal(files.get('environment-freeze.txt'), latest, f'{label} convenience freeze equals latest session')
    expected_files = set(POPULATION_FILES) | {'manifest.json'} | environment_files | {'environment-freeze.txt'}
    audit.check(set(files) == expected_files, f'{label} unexpected/missing files: {sorted(set(files) ^ expected_files)}')

    elapsed = {}
    for arm in ARMS:
        filename = f'{arm}.pt'
        arm_label = f'{label}/{filename}'
        if not audit.check(files.get(filename) is not None and files.get(filename) == manifest['completed_file_sha256'].get(filename),
                           f'{arm_label} verified checkpoint may be loaded') or len(audit.issues) != issues_before:
            continue
        checkpoint = torch.load(child / filename, map_location='cpu', weights_only=True)
        timing = checkpoint['elapsed_seconds']
        audit.check(finite_seconds(timing), f'{arm_label} finite checkpoint elapsed time')
        audit.equal(timing, training[arm]['elapsed_seconds'], f'{arm_label} checkpoint vs phase elapsed time')
        audit.equal(checkpoint['completed_steps'], training[arm]['completed_steps'], f'{arm_label} checkpoint vs phase completed steps')
        if arm == 'teacher':
            audit.check(
                (training[arm]['stopped_reason'] == 'early_stopping' and checkpoint['early_stopped'] and checkpoint['completed_steps'] <= config.teacher_steps)
                or (training[arm]['stopped_reason'] == 'steps_completed' and checkpoint['completed_steps'] == config.teacher_steps and not checkpoint['early_stopped']),
                f'{arm_label} frozen teacher completion')
        else:
            audit.check(training[arm]['stopped_reason'] == 'steps_completed' and checkpoint['completed_steps'] == config.student_steps and not checkpoint['early_stopped'],
                        f'{arm_label} frozen student completion')
        elapsed[arm] = timing
    total = sum(elapsed.values()) if len(elapsed) == len(ARMS) else None
    all_t4 = all(s['device'].split(':', 1)[0] == 'cuda' and s['gpu'] == 'Tesla T4' for s in session_metadata)
    cost = total / 3600 * config.t4_units_per_hour if total is not None and all_t4 else None
    if total is not None:
        audit.close(report['timing']['cumulative_training_seconds'], total, f'{label} checkpoint times counted once')
        audit.close(report['timing']['t4_equivalent_units_at_user_rate'], cost, f'{label} training-only T4 proxy')
    return {
        **item, 'path': label, 'protocol_sha256': protocol_sha, 'file_count': len(files),
        'sessions': session_metadata, 'session_wall': session_wall, 'arm_seconds': elapsed, 'cumulative_training_seconds': total,
        'training_only_t4_units': cost,
    }, expected_files, {'manifest.json'} | set(POPULATION_FILES) | environment_files


def cell(audit, folder, files, config, label):
    manifest = read(folder / 'run_manifest.json')
    summary = read(folder / 'summary.json')
    protocol = read(folder / 'protocol.json')
    protocol_sha = files['protocol.json']
    identity = {'schema_version': 1, 'study': 'main_study_first_tranche', 'config': config.to_dict(), 'source_sha256': CORE}
    audit.equal(manifest['identity'], identity, f'{label} exact pinned manifest identity')
    audit.equal(protocol, {**identity, 'design': config.design}, f'{label} exact frozen protocol')
    audit.hashes(manifest['immutable_file_sha256'], {'protocol.json'}, files, f'{label} immutable')
    audit.check(manifest['status'] == summary['status'] == 'complete', f'{label} manifest and summary complete')
    audit.equal(summary['config'], config.to_dict(), f'{label} summary config')
    audit.equal(summary['planned_populations'], config.plan, f'{label} summary planned populations')
    audit.equal(summary['completed_populations'], config.plan, f'{label} summary completed populations')
    audit.equal(summary['pending_populations'], [], f'{label} summary empty pending populations')
    expected_kinds = config.design['population']['rare_kinds']
    audit.equal(set(summary['kinds']), set(expected_kinds), f'{label} summary rare kinds')
    for kind in expected_kinds:
        rows = summary['kinds'][kind]
        seeds = [item['seed'] for item in config.plan if item['kind'] == kind]
        audit.equal(rows['planned_seeds'], seeds, f'{label} {kind} planned seed list')
        audit.equal(rows['completed_seeds'], seeds, f'{label} {kind} completed seed list')
        audit.equal([row['seed'] for row in rows['per_seed']], seeds, f'{label} {kind} per-seed table')
        audit.equal(set(rows['groups']), {'rare', 'typical'}, f'{label} {kind} grouped summary coverage')
    actual_dirs = {p.relative_to(folder).as_posix() for p in folder.glob('*/seed-*') if p.is_dir()}
    audit.equal(actual_dirs, {f"{item['kind']}/seed-{item['seed']}" for item in config.plan}, f'{label} planned population folders')
    expected_files = {'protocol.json', 'summary.json', 'run_manifest.json'}
    completed_files = {'protocol.json', 'summary.json'}
    pops = []
    for item in config.plan:
        prefix = f"{item['kind']}/seed-{item['seed']}"
        try:
            result, expected, completed = population(audit, folder / prefix, local_map(files, prefix), item, protocol_sha, config, f'{label}/{prefix}')
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            audit.check(False, f'{label}/{prefix} unreadable or malformed population: {exc}')
            continue
        pops.append(result)
        expected_files.update(f'{prefix}/{name}' for name in expected)
        completed_files.update(f'{prefix}/{name}' for name in completed)
    audit.hashes(manifest['completed_file_sha256'], completed_files, files, f'{label} completion')
    audit.check(set(files) == expected_files, f'{label} unexpected/missing files: {sorted(set(files) ^ expected_files)}')
    wall = coverage(audit, manifest['invocations'], f'{label} runner invocations')
    check_coverage(audit, summary['timing'], wall, 'invocations', label)
    durations = [p['cumulative_training_seconds'] for p in pops]
    costs = [p['training_only_t4_units'] for p in pops]
    training_seconds = sum(durations) if len(pops) == len(config.plan) and None not in durations else None
    units = sum(costs) if len(pops) == len(config.plan) and None not in costs else None
    audit.close(summary['timing']['cumulative_training_seconds'], training_seconds, f'{label} checkpoint times counted once')
    audit.close(summary['timing']['t4_equivalent_units_at_user_rate'], units, f'{label} training-only T4 proxy')
    return {'path': label, 'protocol_sha256': protocol_sha, 'completion_manifest_entries': len(completed_files),
            'population_count': len(pops), 'subjects_per_population': config.n_subjects, 'trials_per_subject': config.n_trials,
            'rare_fraction': config.rare_fraction, 'cumulative_training_seconds': training_seconds, 'training_only_t4_units': units,
            'runner_wall': wall, 'populations': pops}


def grid(audit, folder, files, config, reference, label):
    manifest = read(folder / 'run_manifest.json')
    summary = read(folder / 'summary.json')
    protocol = read(folder / 'protocol.json')
    identity = {'schema_version': 1, 'study': 'main_study_rarity_tranche', 'config': config.to_dict(),
                'grid_source_sha256': GRID, 'core_source_sha256': CORE}
    audit.equal(manifest['identity'], identity, f'{label} exact pinned outer manifest identity')
    audit.equal(protocol, {**identity, 'baseline_reference': reference_binding(reference)}, f'{label} exact outer protocol and baseline binding')
    immutable = {'protocol.json'} | {f'baseline_reference/{name}' for name in BASELINE_SNAPSHOT}
    audit.hashes(manifest['immutable_file_sha256'], immutable, files, f'{label} immutable')
    audit.hashes(manifest['completed_file_sha256'], set(files) - {'run_manifest.json'}, files, f'{label} completion')
    audit.check(manifest['status'] == summary['status'] == 'complete', f'{label} manifest and summary complete')
    audit.equal(summary['config'], config.to_dict(), f'{label} summary config')
    audit.equal(summary['planned_cells'], config.plan, f'{label} planned cells')
    audit.equal(summary['completed_cells'], config.plan, f'{label} completed cells')
    audit.equal(summary['pending_cells'], [], f'{label} pending cells empty')
    audit.equal(set(summary['cells']), set(config.plan), f'{label} embedded cell summaries cover plan')
    audit.equal(summary['baseline_reference'], reference, f'{label} embedded baseline summary, protocol and binding')
    audit.equal({p.name for p in (folder / 'cells').iterdir()}, set(config.plan), f'{label} actual cell directory set')
    root_files = {'run_manifest.json', 'summary.json'} | immutable
    audit.equal({name for name in files if not name.startswith('cells/')}, root_files, f'{label} expected outer-level inventory')
    cells = {}
    for cfg in config.cell_configs:
        prefix = f'cells/{cfg.cell_id}'
        child = folder / prefix
        try:
            audit.equal(summary['cells'].get(cfg.cell_id), read(child / 'summary.json'), f'{label}/{prefix} exact embedded/on-disk cell summary')
            cells[cfg.cell_id] = cell(audit, child, local_map(files, prefix), cfg, f'{label}/{prefix}')
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            audit.check(False, f'{label}/{prefix} unreadable or malformed cell: {exc}')
    wall = coverage(audit, manifest['invocations'], f'{label} outer invocations', outer=True)
    check_coverage(audit, summary['timing'], wall, 'invocations', label)
    invocation_groups = Counter(entry.get('cell_id') for entry in manifest['invocations'] if entry['status'] == 'complete')
    audit.equal(set(invocation_groups), set(config.plan), f'{label} successful invocations cover planned cells')
    audit.equal([entry.get('cell_id') for entry in manifest['invocations'] if entry['status'] == 'complete'],
                sorted(entry.get('cell_id') for entry in manifest['invocations'] if entry['status'] == 'complete'), f'{label} planned cell invocation order')
    training = [row['cumulative_training_seconds'] for row in cells.values()]
    costs = [row['training_only_t4_units'] for row in cells.values()]
    seconds = sum(training) if len(cells) == len(config.plan) and None not in training else None
    units = sum(costs) if len(cells) == len(config.plan) and None not in costs else None
    audit.close(summary['timing']['new_training_seconds'], seconds, f'{label} new training seconds')
    audit.close(summary['timing']['completed_population_training_seconds'], seconds, f'{label} complete populations training seconds')
    audit.close(summary['timing']['baseline_training_seconds'], reference['summary']['timing']['cumulative_training_seconds'], f'{label} prior baseline compute separate')
    audit.close(summary['timing']['t4_equivalent_units_at_user_rate'], units, f'{label} training-only T4 proxy')
    return {'path': label, 'protocol_sha256': files['protocol.json'], 'completion_manifest_entries': len(manifest['completed_file_sha256']),
            'cells': cells, 'population_count': sum(c['population_count'] for c in cells.values()),
            'new_training_seconds': seconds, 'training_only_t4_units': units,
            'baseline_training_seconds': summary['timing']['baseline_training_seconds'], 'runner_wall': wall,
            'outer_invocation_counts_by_cell': dict(invocation_groups)}


def check_tree(audit, folder, name, saved):
    actual = snapshot(folder)
    audit.equal(actual, saved, f'{name} before-audit snapshot equals current snapshot')
    audit.equal(actual['unsafe_entries'], [], f'{name} no symlinks or special files')
    files = actual['file_sha256']
    allowed_directories = {parent.as_posix() for key in files for parent in PurePosixPath(key).parents if parent != PurePosixPath('.')}
    audit.equal(set(actual['directories']), allowed_directories, f'{name} no empty or extraneous directories')
    audit.check(not any(re.search(r'(\.tmp|\.partial|\.part|\.crdownload)$', name, re.I) for name in files), f'{name} no partial or staging artifacts')
    if actual['unsafe_entries']:
        raise ValueError(f'{name} has unsafe entries, refusing file reads')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('source', *NAMES, 'inventory', 'output'):
        parser.add_argument(f'--{field}', required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output_path = args.output.resolve()
    if any(output_path.is_relative_to(root.resolve()) for root in (source, args.new, args.baseline, args.smoke)):
        parser.error('Output must be outside the source checkout and input trees')
    if output_path == args.inventory.resolve():
        parser.error('Output must differ from the input inventory')

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source), str(source / 'src')]
    global ARMS, POPULATION_FILES, reference_binding
    from grid_study.baseline import load_reference, reference_binding, validate_baseline
    from grid_study.config import GridTrancheConfig, grid_source_fingerprint
    from main_study.config import MainStudyConfig
    from main_study.runner import ARMS, POPULATION_FILES, main_source_fingerprint

    audit = Audit()
    output = {'audit_mode': 'Read-only. No runner, fitting, resume, or notebook cells called.',
              'elapsed_time_absolute_tolerance_seconds': TOL,
              'file_sha256_before': args.inventory.name,
              'checkpoint_load': {'torch': torch.__version__, 'python': sys.version, 'map_location': 'cpu', 'weights_only': True}}
    output['provenance'] = provenance(audit, source, main_source_fingerprint(), grid_source_fingerprint())
    before = read(args.inventory)
    baseline_name = NAMES['baseline']
    baseline_folder = args.baseline
    try:
        files = check_tree(audit, baseline_folder, baseline_name, before[baseline_name])
        raw = validate_baseline(baseline_folder, 'production')
        audit.equal(raw['file_sha256'], files, 'trusted production baseline exact 87-file inventory')
        output['prior_baseline'] = cell(audit, baseline_folder, files, MainStudyConfig.from_file(source / 'configs/main_study_first_tranche.json'), baseline_name)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        audit.check(False, f'{baseline_folder} raw baseline verification failed: {exc}')
        raw = None
    for field, mode, cfg_name in (('new', 'production', 'rarity_tranche.json'), ('smoke', 'smoke', 'rarity_smoke.json')):
        name, folder = NAMES[field], getattr(args, field)
        try:
            files = check_tree(audit, folder, name, before[name])
            reference = load_reference(folder / 'baseline_reference', mode)
            if mode == 'production' and raw is not None:
                audit.equal(reference, raw, 'production baseline reference matches independently validated original')
                for filename in ('summary.json', 'protocol.json'):
                    audit.equal((folder / 'baseline_reference' / filename).read_bytes(), (baseline_folder / filename).read_bytes(),
                                f'production baseline snapshot {filename} bytes match original')
            if mode == 'smoke':
                output['smoke_parent'] = {'portable_reference_status': 'verified', 'raw_parent_status': 'not supplied',
                                          'embedded_inventory_entries': len(reference['file_sha256']), 'protocol_sha256': reference['protocol_sha256'],
                                          'planned_populations': reference['planned_populations']}
            output[mode] = grid(audit, folder, files, GridTrancheConfig.from_file(source / 'configs' / cfg_name), reference, name)
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            audit.check(False, f'{folder} verification failed: {exc}')
    output['checks'] = audit.checks
    output['issues'] = audit.issues
    output['status'] = 'passed' if not audit.issues else 'issues_found'
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f'{audit.checks} checks; {len(audit.issues)} issues')
    for issue in audit.issues:
        print(issue)
    for mode in ('prior_baseline', 'production', 'smoke'):
        if mode in output:
            item = output[mode]
            print(mode, item['population_count'], item.get('new_training_seconds', item.get('cumulative_training_seconds')),
                  item['training_only_t4_units'], item['runner_wall'], 'protocol hash', item['protocol_sha256'])
    if audit.issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
