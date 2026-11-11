"""Real smoke training, provenance rejection and trajectory-preserving resumption."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from distilled_rnns.experiment import run_lock
from distilled_rnns.models import BatchedStudentGRU, copy_student_to_reference
from foundation.strategy_probes import two_trial_probe_bank
from main_study.config import MainStudyConfig
from main_study import runner


def fingerprints(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


@pytest.fixture(scope='module')
def finished(tmp_path_factory):
    folder = tmp_path_factory.mktemp('main-finished').resolve()
    config = MainStudyConfig(mode='smoke')
    result = runner.run_main_study(config, folder, device='cpu')
    return config, folder, result


def test_complete_smoke_has_every_population_arm_and_primary_sign(finished):
    config, folder, result = finished
    assert result['status'] == 'complete' and result['pending_populations'] == []
    assert result['completed_populations'] == config.plan
    assert result['timing']['cumulative_training_seconds'] > 0
    assert result['timing']['t4_equivalent_units_at_user_rate'] is None
    for item in config.plan:
        report = json.loads((runner.population_path(folder,**item)/'report.json').read_text())
        for arm in ('solo','distilled','oracle'):
            assert report['phases'][arm]['completed_steps'] == config.student_steps
        for subject in report['per_subject']:
            for mode in ('individual','population'):
                solo = subject['models']['solo_'+mode]
                distilled = subject['models']['distilled_'+mode]
                primary = subject['primary'][mode]
                assert primary['prediction_gain'] == solo['nll'] - distilled['nll']
                assert primary['behavioral_distortion'] == distilled['signature_rms_error'] - solo['signature_rms_error']
    first_kind = config.plan[0]['kind']
    summary = result['kinds'][first_kind]
    values = [r['groups']['rare']['primary']['population']['prediction_gain'] for r in summary['per_seed']]
    assert summary['groups']['rare']['primary']['population']['prediction_gain']['mean'] == np.mean(values)


def test_completed_resume_is_read_only_and_adds_no_compute(finished, monkeypatch):
    config, folder, before = finished
    hashes = fingerprints(folder)
    monkeypatch.setattr(runner,'fit_model',lambda *args,**kwargs:pytest.fail('Completed run refit'))
    assert runner.run_main_study(config,folder,device='cpu') == before
    assert fingerprints(folder) == hashes


def test_corrupt_completed_file_is_rejected_before_mutation(finished,tmp_path):
    import shutil
    config, folder, _ = finished
    damaged = tmp_path/'copy'
    shutil.copytree(folder,damaged)
    target = runner.population_path(damaged,**config.plan[0])/'predictions.npz'
    target.write_bytes(target.read_bytes()+b'corrupted')
    before = fingerprints(damaged)
    with pytest.raises(ValueError,match='missing or changed'):
        runner.run_main_study(config,damaged,device='cpu')
    assert fingerprints(damaged) == before


def test_mismatched_config_and_source_rejected(finished,monkeypatch):
    config, folder, _ = finished
    before = fingerprints(folder)
    with pytest.raises(ValueError,match='configuration or source changed'):
        runner.run_main_study(replace(config,max_seconds_per_phase=config.max_seconds_per_phase+1),folder,'cpu')
    monkeypatch.setattr(runner,'main_source_fingerprint',lambda:'0'*64)
    with pytest.raises(ValueError,match='configuration or source changed'):
        runner.run_main_study(config,folder,'cpu')
    assert fingerprints(folder) == before


def assert_tensors_equal(left,right):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tensors_equal(left[key],right[key])
    elif isinstance(left,list):
        assert len(left) == len(right)
        for a,b in zip(left,right):
            assert_tensors_equal(a,b)
    else:
        assert left == right


def test_interrupt_inside_student_restores_optimizer_and_current_trajectory(finished,tmp_path,monkeypatch):
    config, complete_folder, complete_result = finished
    folder = (tmp_path/'interrupted').resolve()
    original_fit = runner.fit_model
    def pause_solo(model,*args,**kwargs):
        if isinstance(model,BatchedStudentGRU) and kwargs['l2_strength']:
            kwargs['steps'] = 1
            result = original_fit(model,*args,**kwargs)
            result['stopped_reason'] = 'time_limit'
            return result
        return original_fit(model,*args,**kwargs)
    monkeypatch.setattr(runner,'fit_model',pause_solo)
    partial = runner.run_main_study(config,folder,'cpu')
    assert partial['status'] == 'incomplete' and not partial['completed_populations']
    assert all('groups' not in kind for kind in partial['kinds'].values())
    first = runner.population_path(folder,**config.plan[0])
    saved = torch.load(first/'solo.pt',map_location='cpu',weights_only=True)
    assert saved['completed_steps'] == 1
    assert all(state['step'].item() == 1 for state in saved['optimizer']['state'].values())
    monkeypatch.setattr(runner,'fit_model',original_fit)
    resumed = runner.run_main_study(config,folder,'cpu')
    assert resumed['status'] == 'complete'
    for item in config.plan:
        reference = runner.population_path(complete_folder,**item)
        actual = runner.population_path(folder,**item)
        for name in ('data.npz','predictions.npz','probe_predictions.npz','soft_training_logits.npz'):
            with np.load(reference/name,allow_pickle=False) as a,np.load(actual/name,allow_pickle=False) as b:
                assert set(a)==set(b)
                for key in a:
                    np.testing.assert_array_equal(a[key],b[key])
        for arm in runner.ARMS:
            a=torch.load(reference/(arm+'.pt'),map_location='cpu',weights_only=True)
            b=torch.load(actual/(arm+'.pt'),map_location='cpu',weights_only=True)
            for key in ('signature','current_state','best_state','optimizer','completed_steps','history','rng_state'):
                assert_tensors_equal(a[key],b[key])
    for kind in resumed['kinds']:
        assert resumed['kinds'][kind]['groups'] == complete_result['kinds'][kind]['groups']


def test_probe_forward_matches_every_independent_reference_and_preserves_subject_order():
    torch.manual_seed(6001)
    model=BatchedStudentGRU(5,4,2)
    actual=runner.probe_forward(model,people_chunk=2)
    bank=torch.tensor(two_trial_probe_bank()[1])
    with torch.no_grad():
        expected=np.stack([torch.sigmoid(copy_student_to_reference(model,i)(bank)).numpy() for i in range(5)])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-7)


def test_output_source_overlap_symlink_and_concurrent_writer_rejected(tmp_path):
    config=MainStudyConfig(mode='smoke')
    with pytest.raises(ValueError,match='overlaps project source'):
        runner.run_main_study(config,runner.ROOT/'main_study'/'bad','cpu')
    link=tmp_path/'alias'
    link.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink'):
        runner.run_main_study(config,link/'output','cpu')
    path=(tmp_path/'locked').resolve()
    path.mkdir()
    with run_lock(path):
        with pytest.raises(RuntimeError,match='another process'):
            runner.run_main_study(config,path,'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_real_cuda_smoke_and_device_checkpoint_validation(tmp_path):
    config=MainStudyConfig(mode='smoke')
    folder=tmp_path.resolve()/'cuda'
    first=runner.run_main_study(config,folder,'cuda')
    before=fingerprints(folder)
    assert runner.run_main_study(config,folder,'cuda') == first
    assert before==fingerprints(folder)
    assert first['status']=='complete'


def test_abrupt_disconnect_resumes_and_preserves_unknown_wall_time(tmp_path,monkeypatch):
    config=MainStudyConfig(mode='smoke')
    folder=(tmp_path/'abrupt').resolve()
    original=runner.fit_model
    def disconnect(model,*args,**kwargs):
        kwargs['steps']=1
        original(model,*args,**kwargs)
        raise RuntimeError('Simulated lost runtime after an atomic checkpoint')
    monkeypatch.setattr(runner,'fit_model',disconnect)
    with pytest.raises(RuntimeError,match='lost runtime'):
        runner.run_main_study(config,folder,'cpu')
    first=runner.population_path(folder,**config.plan[0])
    before=json.loads((first/'manifest.json').read_text())
    assert 'wall_seconds' not in before['sessions'][0]
    monkeypatch.setattr(runner,'fit_model',original)
    result=runner.run_main_study(config,folder,'cpu')
    assert result['status']=='complete'
    manifest=json.loads((first/'manifest.json').read_text())
    assert manifest['sessions'][0]['status']=='interrupted'
    assert manifest['sessions'][0]['wall_seconds'] is None
    report=json.loads((first/'report.json').read_text())
    assert report['timing']['session_wall_seconds'] is None
    assert report['timing']['known_session_wall_seconds']>0
    assert report['timing']['session_wall_time_coverage']=={'recorded_sessions':1,'total_sessions':2}


@pytest.mark.parametrize('tamper',['lr','early_stop','moment_dtype','mapping'])
def test_mutable_checkpoint_adam_and_fixed_student_stopping_are_validated(tmp_path,monkeypatch,tamper):
    config=MainStudyConfig(mode='smoke')
    folder=(tmp_path/tamper).resolve()
    original=runner.fit_model
    def pause_solo(model,*args,**kwargs):
        if isinstance(model,BatchedStudentGRU) and kwargs['l2_strength']:
            kwargs['steps']=1
            report=original(model,*args,**kwargs)
            report['stopped_reason']='time_limit'
            return report
        return original(model,*args,**kwargs)
    monkeypatch.setattr(runner,'fit_model',pause_solo)
    assert runner.run_main_study(config,folder,'cpu')['status']=='incomplete'
    checkpoint=runner.population_path(folder,**config.plan[0])/'solo.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if tamper=='lr':
        saved['optimizer']['param_groups'][0]['lr']=.123
    elif tamper=='early_stop':
        saved['early_stopped']=True
    elif tamper=='mapping':
        saved['optimizer']['param_groups'][0]['params'].reverse()
    else:
        saved['optimizer']['state'][0]['exp_avg']=saved['optimizer']['state'][0]['exp_avg'].double()
    torch.save(saved,checkpoint)
    monkeypatch.setattr(runner,'fit_model',original)
    with pytest.raises(ValueError,match='Adam|fixed optimization budget'):
        runner.run_main_study(config,folder,'cpu')


def test_dotdot_alias_uses_canonical_lock_and_source_overlap(tmp_path):
    config=MainStudyConfig(mode='smoke')
    source_alias=runner.ROOT/'main_study'/'unmade'/'..'/'bad'
    with pytest.raises(ValueError,match='overlaps project source'):
        runner.run_main_study(config,source_alias,'cpu')
    path=(tmp_path/'same').resolve()
    path.mkdir()
    alias=path/'unmade'/'..'
    with run_lock(path):
        with pytest.raises(RuntimeError,match='another process'):
            runner.run_main_study(config,alias,'cpu')


def test_protocol_only_initialization_recovers_but_unknown_files_do_not(tmp_path,monkeypatch):
    config=MainStudyConfig(mode='smoke')
    folder=(tmp_path/'protocol-crash').resolve()
    original=runner.atomic_json
    def fail_manifest(path,value):
        if path.name=='run_manifest.json':
            raise RuntimeError('root initialization crash')
        original(path,value)
    monkeypatch.setattr(runner,'atomic_json',fail_manifest)
    with pytest.raises(RuntimeError,match='initialization crash'):
        runner.run_main_study(config,folder,'cpu')
    assert {p.name for p in folder.iterdir()}=={'protocol.json'}
    monkeypatch.setattr(runner,'atomic_json',original)
    assert runner.run_main_study(config,folder,'cpu')['status']=='complete'
    invalid=tmp_path/'invalid'
    invalid.mkdir()
    (invalid/'unknown.pt').write_bytes(b'not a valid study')
    with pytest.raises(ValueError,match='without this study manifest'):
        runner.run_main_study(config,invalid,'cpu')


def test_data_written_before_manifest_can_be_adopted_only_when_exact(tmp_path,monkeypatch):
    config=MainStudyConfig(mode='smoke')
    folder=(tmp_path/'data-init').resolve()
    original=runner._save_npz
    def fail_after_data(path,**arrays):
        original(path,**arrays)
        if path.name=='data.npz':
            raise RuntimeError('lost after data initialization')
    monkeypatch.setattr(runner,'_save_npz',fail_after_data)
    with pytest.raises(RuntimeError,match='data initialization'):
        runner.run_main_study(config,folder,'cpu')
    first=runner.population_path(folder,**config.plan[0])
    assert (first/'data.npz').exists()
    # Simulate a download made by an older initializer before its manifest write.
    (first/'manifest.json').unlink()
    monkeypatch.setattr(runner,'_save_npz',original)
    assert runner.run_main_study(config,folder,'cpu')['status']=='complete'


def test_reconnection_tolerates_only_tiny_truth_regeneration_and_keeps_saved_bytes(tmp_path,monkeypatch):
    config=MainStudyConfig(mode='smoke')
    folder=(tmp_path/'portable').resolve()
    original_fit=runner.fit_model
    def disconnect(model,*args,**kwargs):
        kwargs['steps']=1
        original_fit(model,*args,**kwargs)
        raise RuntimeError('simulate disconnect')
    monkeypatch.setattr(runner,'fit_model',disconnect)
    with pytest.raises(RuntimeError,match='disconnect'):
        runner.run_main_study(config,folder,'cpu')
    first=runner.population_path(folder,**config.plan[0])
    original_hash=hashlib.sha256((first/'data.npz').read_bytes()).hexdigest()
    original_simulate=runner.simulate_population
    def tiny_roundoff(*args):
        data=original_simulate(*args)
        data['true_prob']+=1e-14
        data['reference_prob']+=1e-14
        return data
    monkeypatch.setattr(runner,'simulate_population',tiny_roundoff)
    monkeypatch.setattr(runner,'fit_model',original_fit)
    result=runner.run_main_study(config,folder,'cpu')
    assert result['status']=='complete'
    assert hashlib.sha256((first/'data.npz').read_bytes()).hexdigest()==original_hash


@pytest.mark.parametrize('target_name',['protocol.json','run_manifest.json','manifest.json'])
def test_partial_initial_atomic_write_recovers_only_verified_staging_prefix(tmp_path,monkeypatch,target_name):
    config=MainStudyConfig(mode='smoke')
    folder=(tmp_path/target_name).resolve()
    original=runner.atomic_json
    triggered=False
    def fail_partial(path,value):
        nonlocal triggered
        if path.name==target_name and not triggered:
            triggered=True
            temporary=path.with_suffix('.tmp')
            payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
            temporary.write_bytes(payload[:max(1,len(payload)//2)])
            raise RuntimeError('atomic initialization interrupted')
        original(path,value)
    monkeypatch.setattr(runner,'atomic_json',fail_partial)
    with pytest.raises(RuntimeError,match='atomic initialization'):
        runner.run_main_study(config,folder,'cpu')
    monkeypatch.setattr(runner,'atomic_json',original)
    assert runner.run_main_study(config,folder,'cpu')['status']=='complete'


def test_arbitrary_protocol_staging_file_is_not_adopted(tmp_path):
    config=MainStudyConfig(mode='smoke')
    folder=tmp_path/'foreign'
    folder.mkdir()
    (folder/'protocol.tmp').write_bytes(b'{"different":true}')
    before=fingerprints(folder)
    with pytest.raises(ValueError,match='temporary protocol'):
        runner.run_main_study(config,folder,'cpu')
    assert fingerprints(folder)==before
