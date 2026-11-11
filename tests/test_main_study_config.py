import json
import pytest

from main_study.config import MainStudyConfig, PROTOCOL_SHA256


def test_frozen_production_plan_and_smoke_are_separate():
    config = MainStudyConfig()
    assert [row['seed'] for row in config.plan] == [202611401,202611411,202611421,202621401,202621411,202621421]
    assert config.n_subjects==1000 and config.n_trials==100 and config.rare_fraction==.02
    assert config.student_steps==4000 and config.teacher_steps==500
    assert config.to_dict()['protocol_sha256']==PROTOCOL_SHA256
    smoke=MainStudyConfig(mode='smoke')
    assert len(smoke.plan)==4 and not {p['seed'] for p in smoke.plan}&{p['seed'] for p in config.plan}


def test_runtime_config_cannot_silently_change_a_scientific_setting(tmp_path):
    path=tmp_path/'config.json'
    values={'mode':'production','max_seconds_per_phase':600,'max_seconds_per_invocation':1800,'student_steps':8000}
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError,match='Only mode'):
        MainStudyConfig.from_file(path)
    for value in (0,False,float('nan')):
        with pytest.raises(ValueError):
            MainStudyConfig(max_seconds_per_phase=value)
