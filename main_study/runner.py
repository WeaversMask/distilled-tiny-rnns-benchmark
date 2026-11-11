"""Resumable first-tranche comparison under the frozen prospective design."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from distilled_rnns.budget_check import _finite, _reject_symlinks
from distilled_rnns.evaluation import make_trial_masks
from distilled_rnns.experiment import (atomic_json, environment_info, resolve_device, run_lock,
    save_environment_inventory, t4_training_cost_proxy, _save_npz)
from distilled_rnns.models import BatchedStudentGRU, TeacherGRU
from distilled_rnns.oracle_control import oracle_training_logits
from distilled_rnns.training import fit_model, _signature
from foundation.strategy_probes import two_trial_probe_bank

from .config import MainStudyConfig
from .simulation import simulate_population
from .evaluation import baseline_predictions, evaluate

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('teacher', 'solo', 'distilled', 'oracle')
POPULATION_FILES = ('data.npz', 'teacher.pt', 'solo.pt', 'distilled.pt', 'oracle.pt',
    'soft_training_logits.npz', 'predictions.npz', 'probe_predictions.npz',
    'training.json', 'report.json', 'primary.png')


def main_source_fingerprint():
    digest = hashlib.sha256()
    paths = [path for folder in ('main_study', 'foundation', 'src/distilled_rnns')
             for path in (ROOT / folder).glob('*.py')]
    for path in sorted(paths, key=lambda value: value.relative_to(ROOT).as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


source_sha256 = main_source_fingerprint


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _checked_hashes(folder, expected):
    for name, digest in expected.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe artifact hash path')
        if not (folder / path).is_file() or _sha(folder / path) != digest:
            raise ValueError('Artifact missing or changed: ' + str(folder / path))


def _check_inventories(folder, sessions):
    if not sessions:
        raise ValueError('Population manifest has no recorded session')
    for session in sessions:
        inventory = session['package_inventory']
        _checked_hashes(folder, {inventory['path']: inventory['sha256']})
        if not isinstance(session.get('environment'), dict):
            raise ValueError('Missing session environment')


def population_path(folder, kind, seed):
    return folder / kind / f'seed-{seed}'


def _population_identity(item, protocol_hash):
    return {**item, 'protocol_sha256': protocol_hash}


def _validate_complete_population(folder, item, protocol_hash):
    manifest = _json(folder / 'manifest.json')
    if manifest.get('identity') != _population_identity(item, protocol_hash) or manifest.get('status') != 'complete':
        raise ValueError('Population identity or completion status changed')
    if set(manifest.get('completed_file_sha256', {})) != set(POPULATION_FILES):
        raise ValueError('Population completed-file manifest differs')
    _checked_hashes(folder, manifest['completed_file_sha256'])
    _check_inventories(folder, manifest['sessions'])
    report = _json(folder / 'report.json')
    if report.get('status') != 'complete' or report.get('seed') != item['seed'] or report.get('kind') != item['kind']:
        raise ValueError('Population report identity changed')
    return report


def _new_model(config, arm):
    if arm == 'teacher':
        return TeacherGRU(4, config.teacher_hidden)
    return BatchedStudentGRU(config.n_subjects, 4, config.student_hidden)


def _training_settings(config, arm, seed):
    settings = {'lr': config.learning_rate, 'seed': seed + (2 if arm == 'teacher' else 3),
                'eval_every': config.eval_every, 'patience': config.patience,
                'teacher_prop': 1.0 if arm in ('distilled', 'oracle') else 0.0,
                'temperature': config.temperature}
    if arm == 'solo' and config.solo_l2:
        settings['l2_strength'] = float(config.solo_l2)
    if arm != 'teacher':
        settings['track_population_selection'] = True
    return settings


def _checkpoint(config, arm, seed, path, model, x, y, train, validation, soft):
    saved = torch.load(path, map_location=x.device, weights_only=True)
    _finite(saved, str(path))
    settings = _training_settings(config, arm, seed)
    signature = _signature(model, settings, [x, y, train, validation, soft])
    if saved.get('signature') != signature:
        raise ValueError('Checkpoint identity differs: ' + str(path))
    completed = saved.get('completed_steps')
    limit = config.teacher_steps if arm == 'teacher' else config.student_steps
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= limit:
        raise ValueError('Invalid checkpoint optimization counter')
    if not isinstance(saved.get('elapsed_seconds'), (float, int)) or saved['elapsed_seconds'] < 0:
        raise ValueError('Invalid checkpoint training time')
    expected_states = ('current_state', 'best_state') if arm == 'teacher' else ('current_state', 'best_state', 'population_best_state')
    template = model.state_dict()
    for key in expected_states:
        state = saved[key]
        if set(state) != set(template) or any(state[name].shape != value.shape or state[name].dtype != value.dtype
                                            for name, value in template.items()):
            raise ValueError('Checkpoint model structure differs')
    if not isinstance(saved.get('early_stopped'), bool) or (arm != 'teacher' and saved['early_stopped']):
        raise ValueError('Students must retain the fixed optimization budget')
    if isinstance(saved.get('stale'), bool) or not isinstance(saved.get('stale'), int) or saved['stale'] < 0:
        raise ValueError('Invalid checkpoint patience counter')
    if saved['early_stopped'] and saved['stale'] < config.patience:
        raise ValueError('Teacher early stop precedes the declared patience')
    optimizer = saved['optimizer']
    expected_adam = {'lr':config.learning_rate, 'betas':(.9,.999), 'eps':1e-8,
                     'weight_decay':0, 'amsgrad':False, 'maximize':False, 'foreach':None,
                     'capturable':False, 'differentiable':False, 'fused':None}
    if len(optimizer['param_groups']) != 1:
        raise ValueError('Adam parameter groups differ')
    group = optimizer['param_groups'][0]
    parameters = list(model.parameters())
    if (group.get('params') != list(range(len(parameters)))
            or any(group.get(key) != value for key,value in expected_adam.items())
            or group.get('decoupled_weight_decay',False)):
        raise ValueError('Adam effective settings or parameter mapping differ')
    expected_ids = set(range(len(parameters))) if completed else set()
    if set(optimizer['state']) != expected_ids:
        raise ValueError('Incomplete Adam state')
    for index, parameter in enumerate(parameters) if completed else []:
        state = optimizer['state'][index]
        if (set(state) != {'step','exp_avg','exp_avg_sq'} or state['step'].item() != completed
                or any(state[key].shape != parameter.shape or state[key].dtype != parameter.dtype
                       for key in ('exp_avg','exp_avg_sq'))):
            raise ValueError('Adam counters or moment shapes/dtypes differ')
    if (saved['rng_state'].dtype != torch.uint8
            or saved['rng_state'].shape != torch.get_rng_state().shape):
        raise ValueError('Checkpoint CPU RNG structure differs')
    history = saved['history']
    expected_steps = [0] + list(range(config.eval_every, completed + 1, config.eval_every))
    if completed == limit and limit not in expected_steps:
        expected_steps.append(limit)
    if [row['step'] for row in history] != expected_steps:
        raise ValueError('Validation schedule differs from cumulative history')
    steps = saved['best_step']
    selected_steps = [steps] if arm == 'teacher' else steps.detach().cpu().tolist()
    if len(selected_steps) != (1 if arm == 'teacher' else config.n_subjects) or any(step not in expected_steps for step in selected_steps):
        raise ValueError('Invalid selected optimization step')
    population_best = min(history, key=lambda row: row['validation_nll'])
    if arm == 'teacher':
        if abs(saved['best_val'].item()-population_best['validation_nll']) > 5e-7:
            raise ValueError('Teacher selected value differs from history')
        if saved['best_step'] != population_best['step']:
            raise ValueError('Teacher selection differs from validation history')
    elif saved['population_best_step'] != population_best['step'] or abs(saved['population_best_val'].item()-population_best['validation_nll']) > 5e-7:
        raise ValueError('Population selection differs from validation history')
    for state_key, value_key in [('best_state', 'best_val')] + ([] if arm == 'teacher' else [('population_best_state', 'population_best_val')]):
        model.load_state_dict(saved[state_key])
        with torch.no_grad():
            losses = F.binary_cross_entropy_with_logits(model(x), y, reduction='none')
            per_person = (losses * validation).sum(1) / validation.sum(1)
            observed = per_person if arm != 'teacher' and state_key == 'best_state' else per_person.mean()
            if not torch.allclose(observed, saved[value_key], rtol=0, atol=5e-6 if observed.ndim else 5e-7):
                raise ValueError('Selected validation predictions differ on the chosen device')
    if history[-1]['step'] == completed:
        model.load_state_dict(saved['current_state'])
        with torch.no_grad():
            losses = F.binary_cross_entropy_with_logits(model(x),y,reduction='none')
            for label,mask in [('train_nll',train),('validation_nll',validation)]:
                observed = ((losses*mask).sum(1)/mask.sum(1)).mean().item()
                if abs(observed-history[-1][label]) > 5e-7:
                    raise ValueError('Current checkpoint predictions differ from recorded history')
    return saved


def _phase_report(saved, arm, config, checkpoint_path):
    report = {key: saved[key] for key in ('history', 'completed_steps', 'elapsed_seconds')}
    report.update(best_step=saved['best_step'].detach().cpu().tolist() if arm != 'teacher' else saved['best_step'],
        best_validation_nll=saved['best_val'].detach().cpu().tolist(),
        stopped_reason='early_stopping' if saved['early_stopped'] else ('steps_completed' if saved['completed_steps'] ==
            (config.teacher_steps if arm == 'teacher' else config.student_steps) else 'time_limit'),
        l2_strength=config.solo_l2 if arm == 'solo' else 0.0, checkpoint_path=str(checkpoint_path))
    if arm != 'teacher':
        report.update(population_best_step=saved['population_best_step'],
                      population_best_validation_nll=saved['population_best_val'].item())
    return report


def probe_forward(model, n_subjects=None, people_chunk=32):
    """Bounded-memory exact parameter replication over the 256 assay histories."""
    _, bank = two_trial_probe_bank()
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    bank = torch.as_tensor(bank, device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        if isinstance(model, TeacherGRU):
            result = torch.sigmoid(model(bank)).cpu().numpy()
            return np.broadcast_to(result, (n_subjects, *result.shape)).copy()
        result = np.empty((model.n_subjects, 256, 3), dtype=np.float32)
        for start in range(0, model.n_subjects, people_chunk):
            stop = min(model.n_subjects, start + people_chunk)
            count = stop - start
            probe_model = BatchedStudentGRU(count * 256, 4, model.hidden_size).to(device=device, dtype=dtype)
            probe_model.load_state_dict({name: value[start:stop].repeat_interleave(256, dim=0)
                                         for name, value in model.state_dict().items()})
            inputs = bank.repeat(count, 1, 1)
            result[start:stop] = torch.sigmoid(probe_model(inputs)).reshape(count, 256, 3).cpu().numpy()
        return result


def _plot_primary(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for rare, label in ((False, 'Typical'), (True, 'Rare')):
        rows = [row['primary']['population'] for row in report['per_subject'] if row['rare'] == rare]
        ax.scatter([row['prediction_gain'] for row in rows], [row['behavioral_distortion'] for row in rows],
                   label=label, alpha=.6, s=16)
    ax.axhline(0, color='grey', linewidth=.7)
    ax.axvline(0, color='grey', linewidth=.7)
    ax.set(xlabel='Solo minus distilled test NLL', ylabel='Distilled minus solo signature RMS error')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _same_inputs(saved, expected):
    if set(saved) != set(expected):
        return False
    for key,value in expected.items():
        prior = saved[key]
        if prior.dtype != value.dtype or prior.shape != value.shape:
            return False
        if key in ('true_prob','reference_prob'):
            if not np.allclose(prior,value,rtol=0,atol=1e-12):
                return False
        elif not np.array_equal(prior,value):
            return False
    return True


def _initial_temp_matches(path, expected):
    serialized = (json.dumps(expected,indent=2,allow_nan=False)+'\n').encode()
    return serialized.startswith(path.read_bytes())


def _adoptable_initial_files(folder, data, expected_manifest):
    """Recover only deterministic inputs and content-addressed pre-fit inventories."""
    for path in folder.rglob('*'):
        if path.is_dir():
            if path.relative_to(folder).as_posix() != 'environments':
                raise ValueError('Unexpected population initialization directory')
            continue
        name = path.relative_to(folder).as_posix()
        if name == 'manifest.tmp':
            if not _initial_temp_matches(path,expected_manifest):
                raise ValueError('Unexpected temporary population manifest')
        elif name == 'data.npz':
            saved = _arrays(path)
            if not _same_inputs(saved,data):
                raise ValueError('Orphaned population data differ from the deterministic inputs')
        elif re.fullmatch(r'environments/packages-[0-9a-f]{64}\.txt',name):
            if _sha(path) != path.stem.removeprefix('packages-'):
                raise ValueError('Orphaned environment inventory hash differs')
        elif name == 'environment-freeze.txt':
            if not (folder/'environments'/f'packages-{_sha(path)}.txt').is_file():
                raise ValueError('Orphaned environment freeze has no matching inventory')
        else:
            raise ValueError('Unexpected population files without a manifest')


def _save_population(config, item, folder, protocol_hash, device, deadline):
    started = time.perf_counter()
    identity = _population_identity(item, protocol_hash)
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        manifest = _json(manifest_path)
        if manifest['identity'] != identity:
            raise ValueError('Population configuration or protocol differs')
        if manifest.get('status') == 'complete':
            return _validate_complete_population(folder, item, protocol_hash)
        if manifest['sessions']:
            _check_inventories(folder, manifest['sessions'])
        elif any(folder.glob('*.pt')):
            raise ValueError('Checkpoint has no recorded training session')
        _checked_hashes(folder, manifest.get('immutable_file_sha256', {}))
    else:
        manifest = None
    seed, kind = item['seed'], item['kind']
    data = simulate_population(config, kind, seed)
    masks = make_trial_masks(config.n_subjects, config.n_trials, seed + 1)
    stored_data = {**data, **{f'mask_{name}': value for name, value in masks.items()}}
    folder.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        manifest = {'identity':identity,'status':'incomplete','sessions':[],'immutable_file_sha256':{}}
        _adoptable_initial_files(folder,stored_data,manifest)
        # Persist identity before any input or inventory can be written.
        atomic_json(manifest_path,manifest)
    data_path = folder / 'data.npz'
    if data_path.exists():
        prior = _arrays(data_path)
        if not _same_inputs(prior,stored_data):
            raise ValueError('Regenerated data or masks differ from saved inputs')
        data = {key:prior[key] for key in data}
        stored_data = prior
    else:
        _save_npz(data_path, **stored_data)
    manifest['immutable_file_sha256']['data.npz'] = _sha(data_path)
    atomic_json(manifest_path,manifest)
    session = {'started_utc': datetime.now(timezone.utc).isoformat(), 'environment': environment_info(device),
               'package_inventory': save_environment_inventory(folder, {f"{package.metadata['Name']}=={package.version}"
                   for package in importlib.metadata.distributions()})}
    for prior_session in manifest['sessions']:
        if 'wall_seconds' not in prior_session:
            prior_session.update(status='interrupted',wall_seconds=None)
    manifest['sessions'].append(session)
    atomic_json(manifest_path, manifest)
    x = torch.as_tensor(data['x'], device=device)
    y = torch.as_tensor(data['choice'], dtype=torch.float32, device=device)
    train = torch.as_tensor(masks['train'], device=device)
    validation = torch.as_tensor(masks['val'], device=device)
    soft = {'oracle': oracle_training_logits(data['true_prob'], masks['train'], device)}
    if (folder / 'soft_training_logits.npz').exists():
        soft = {key: torch.as_tensor(value, device=device) for key, value in _arrays(folder / 'soft_training_logits.npz').items()}
        if set(soft) != {'oracle', 'distilled'} or any(torch.count_nonzero(value[~train]).item() for value in soft.values()):
            raise ValueError('Soft targets must expose training positions only')
        if not torch.equal(soft['oracle'], oracle_training_logits(data['true_prob'], masks['train'], device)):
            raise ValueError('Saved oracle fitting targets differ')
    phases, predictions, probe_predictions = {}, {}, {}
    for arm in ARMS:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            session.update(status='incomplete', phase=arm, wall_seconds=time.perf_counter()-started)
            atomic_json(manifest_path, manifest)
            return None
        initialization = seed + (2 if arm == 'teacher' else 3)
        torch.manual_seed(initialization)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(initialization)
        model = _new_model(config, arm).to(device)
        path = folder / f'{arm}.pt'
        saved = _checkpoint(config, arm, seed, path, model, x, y, train, validation, soft.get(arm)) if path.exists() else None
        limit = config.teacher_steps if arm == 'teacher' else config.student_steps
        done = saved is not None and (saved['completed_steps'] == limit or (arm == 'teacher' and saved['early_stopped']))
        if not done:
            outcome = fit_model(model, x, y, train, validation, steps=limit, lr=config.learning_rate,
                checkpoint_path=path, seed=initialization, eval_every=config.eval_every, patience=config.patience,
                teacher_logits=soft.get(arm), teacher_prop=1.0 if arm in ('distilled', 'oracle') else 0.0,
                temperature=config.temperature, max_seconds=min(config.max_seconds_per_phase, remaining),
                l2_strength=config.solo_l2 if arm == 'solo' else 0.0, track_population_selection=arm != 'teacher')
            saved = _checkpoint(config, arm, seed, path, model, x, y, train, validation, soft.get(arm))
            phases[arm] = _phase_report(saved, arm, config, path)
            atomic_json(folder / 'training.json', phases)
            if outcome['stopped_reason'] == 'time_limit':
                session.update(status='incomplete', phase=arm, wall_seconds=time.perf_counter()-started)
                atomic_json(manifest_path, manifest)
                return None
        phases[arm] = _phase_report(saved, arm, config, path)
        for mode, state in ([('', 'best_state')] if arm == 'teacher' else
                            [('_individual', 'best_state'), ('_population', 'population_best_state')]):
            model.load_state_dict(saved[state])
            model.eval()
            with torch.no_grad():
                logits = model(x)
                predictions[arm + mode] = torch.sigmoid(logits).cpu().numpy()
            probe_predictions[arm + mode] = probe_forward(model, config.n_subjects)
            if arm == 'teacher':
                targets = torch.where(train, logits.detach(), torch.zeros_like(logits))
                if 'distilled' in soft:
                    if not torch.allclose(targets, soft['distilled'], rtol=0, atol=5e-6):
                        raise ValueError('Teacher fitting targets changed on resume device')
                else:
                    soft['distilled'] = targets
                    _save_npz(folder / 'soft_training_logits.npz', **{key: value.cpu().numpy() for key, value in soft.items()})
                    manifest['immutable_file_sha256']['soft_training_logits.npz'] = _sha(folder / 'soft_training_logits.npz')
                    atomic_json(manifest_path, manifest)
        del model, saved
    classical, classical_probes, metadata = baseline_predictions(config, data, masks['train'])
    predictions.update(classical)
    probe_predictions.update(classical_probes)
    _save_npz(folder / 'predictions.npz', **predictions)
    _save_npz(folder / 'probe_predictions.npz', **probe_predictions)
    atomic_json(folder / 'training.json', phases)
    report = evaluate(config, data, predictions, probe_predictions, masks['test'], baseline_metadata=metadata)
    seconds = sum(phase['elapsed_seconds'] for phase in phases.values())
    session.update(status='complete', wall_seconds=time.perf_counter()-started)
    report.update(status='complete', seed=seed, kind=kind, phases=phases,
                  timing={'cumulative_training_seconds': seconds,
                          'session_wall_seconds':sum(s['wall_seconds'] for s in manifest['sessions']) if all(s['wall_seconds'] is not None for s in manifest['sessions']) else None,
                          'known_session_wall_seconds':sum(s['wall_seconds'] for s in manifest['sessions'] if s['wall_seconds'] is not None),
                          'session_wall_time_coverage':{'recorded_sessions':sum(s['wall_seconds'] is not None for s in manifest['sessions']), 'total_sessions':len(manifest['sessions'])},
                          **t4_training_cost_proxy(seconds, config.t4_units_per_hour, manifest['sessions'])})
    _plot_primary(report, folder / 'primary.png')
    atomic_json(folder / 'report.json', report)
    manifest.update(status='complete', completed_file_sha256={name: _sha(folder / name) for name in POPULATION_FILES})
    atomic_json(manifest_path, manifest)
    return report


def _aggregate(values):
    first = values[0]
    if isinstance(first, dict):
        if any(set(value) != set(first) for value in values):
            raise ValueError('Population report metric sets differ')
        return {key: _aggregate([value[key] for value in values]) for key in first}
    if isinstance(first, list):
        return [_aggregate([value[index] for value in values]) for index in range(len(first))]
    present = [value for value in values if value is not None]
    if any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in present):
        raise ValueError('Aggregate values must be finite scalars or null')
    return {'mean': float(np.mean(present)) if present else None, 'min': min(present) if present else None,
            'max': max(present) if present else None, 'supported_seed_count': len(present), 'planned_seed_count':len(values),
            'positive_seed_count':sum(value > 0 for value in present), 'negative_seed_count':sum(value < 0 for value in present),
            'tied_seed_count':sum(value == 0 for value in present)}


def _summary(config, reports, folder, complete):
    completed = [{'kind': report['kind'], 'seed': report['seed']} for report in reports]
    kinds = {}
    for kind in config.design['population']['rare_kinds']:
        planned = [item['seed'] for item in config.plan if item['kind'] == kind]
        present = [report for report in reports if report['kind'] == kind]
        kinds[kind] = {'planned_seeds': planned, 'completed_seeds':[report['seed'] for report in present]}
        if complete:
            kinds[kind].update(groups=_aggregate([report['groups'] for report in present]),
                               per_seed=[{'seed':report['seed'], 'groups':report['groups']} for report in present])
    seconds = sum(report['timing']['cumulative_training_seconds'] for report in reports)
    units = [report['timing']['t4_equivalent_units_at_user_rate'] for report in reports]
    return {'status':'complete' if complete else 'incomplete', 'output_dir':str(folder), 'config':config.to_dict(),
            'planned_populations':config.plan, 'completed_populations':completed,
            'pending_populations':[item for item in config.plan if item not in completed], 'kinds':kinds,
            'timing': {'cumulative_training_seconds':seconds,
                       't4_equivalent_units_at_user_rate':sum(units) if units and all(value is not None for value in units) else None,
                       'note':'Checkpoint training time counted once. Setup, baseline fitting, probes, evaluation, saving and idle billing are additional.'},
            'message':'Complete. Retain every result folder and disconnect the runtime.' if complete else
                      'Resume by rerunning only the fourth code cell with the same run name. Aggregate results require all planned populations.'}


def run_main_study(config: MainStudyConfig, output_dir, device='auto'):
    invocation_started = time.perf_counter()
    if not isinstance(config, MainStudyConfig):
        raise ValueError('config must be a MainStudyConfig')
    requested = Path(output_dir).absolute()
    _reject_symlinks(requested)
    folder = requested.resolve()
    if any(folder.is_relative_to(ROOT / name) or (ROOT / name).is_relative_to(folder)
           for name in ('main_study', 'foundation', 'src')):
        raise ValueError('Output folder overlaps project source')
    selected = resolve_device(device)
    if selected.type == 'cpu':
        torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    folder.mkdir(parents=True, exist_ok=True)
    with run_lock(folder):
        identity = {'schema_version':1, 'study':'main_study_first_tranche', 'config':config.to_dict(),
                    'source_sha256':main_source_fingerprint()}
        manifest_path = folder / 'run_manifest.json'
        if manifest_path.exists():
            manifest = _json(manifest_path)
            if manifest['identity'] != identity:
                raise ValueError('Run configuration or source changed; preserve this run and use its original runtime')
            _checked_hashes(folder, manifest['immutable_file_sha256'])
        else:
            expected_protocol = {**identity,'design':config.design}
            existing = {path.name for path in folder.iterdir()}
            if not existing.issubset({'protocol.json','protocol.tmp','run_manifest.tmp'}):
                raise ValueError('Output folder has files without this study manifest')
            if 'run_manifest.tmp' in existing and 'protocol.json' not in existing:
                raise ValueError('Temporary run manifest has no verified initial protocol')
            if 'protocol.json' in existing and _json(folder/'protocol.json') != expected_protocol:
                raise ValueError('Orphaned protocol differs from this study')
            if 'protocol.tmp' in existing and not _initial_temp_matches(folder/'protocol.tmp',expected_protocol):
                raise ValueError('Unexpected temporary protocol')
            if 'protocol.json' not in existing:
                atomic_json(folder/'protocol.json',expected_protocol)
            manifest = {'identity':identity,'status':'incomplete','invocations':[],
                        'immutable_file_sha256':{'protocol.json':_sha(folder/'protocol.json')}}
            if 'run_manifest.tmp' in existing and not _initial_temp_matches(folder/'run_manifest.tmp',manifest):
                raise ValueError('Unexpected temporary run manifest')
            atomic_json(manifest_path,manifest)
            (folder/'protocol.tmp').unlink(missing_ok=True)
        protocol_hash = manifest['immutable_file_sha256']['protocol.json']
        if manifest.get('status') == 'complete':
            _checked_hashes(folder, manifest['completed_file_sha256'])
            for item in config.plan:
                _validate_complete_population(population_path(folder, **item), item, protocol_hash)
            result = _json(folder / 'summary.json')
            if result.get('status') != 'complete' or result.get('planned_populations') != config.plan or result.get('pending_populations'):
                raise ValueError('Completed summary differs from the planned tranche')
            return result
        invocations = manifest.setdefault('invocations',[])
        for previous in invocations:
            if 'wall_seconds' not in previous:
                previous.update(status='interrupted',wall_seconds=None)
        invocation = {'started_utc':datetime.now(timezone.utc).isoformat()}
        invocations.append(invocation)
        atomic_json(manifest_path,manifest)
        deadline = invocation_started + config.max_seconds_per_invocation
        reports = []
        for item in config.plan:
            if time.perf_counter() >= deadline:
                break
            report = _save_population(config, item, population_path(folder, **item), protocol_hash, selected, deadline)
            if report is None:
                break
            reports.append(report)
        complete = len(reports) == len(config.plan)
        summary = _summary(config, reports, folder, complete)
        invocation.update(status='complete' if complete else 'incomplete',
                          wall_seconds=time.perf_counter()-invocation_started)
        known_wall = [entry['wall_seconds'] for entry in invocations if entry.get('wall_seconds') is not None]
        summary['timing'].update(runner_wall_seconds=sum(known_wall) if len(known_wall)==len(invocations) else None,
                                known_runner_wall_seconds=sum(known_wall),
                                invocation_wall_time_coverage={'recorded_invocations':len(known_wall),'total_invocations':len(invocations)})
        atomic_json(folder / 'summary.json', summary)
        if complete:
            files = ['protocol.json', 'summary.json']
            for item in config.plan:
                prefix = population_path(folder, **item).relative_to(folder)
                files.extend(str(prefix / name) for name in (*POPULATION_FILES, 'manifest.json'))
                sessions = _json(folder / prefix / 'manifest.json')['sessions']
                files.extend(str(prefix / session['package_inventory']['path']) for session in sessions)
            manifest.update(status='complete', completed_file_sha256={name:_sha(folder / name) for name in set(files)})
        atomic_json(manifest_path, manifest)
        return summary
