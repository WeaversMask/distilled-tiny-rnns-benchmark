"""Freeze the production contrast and exclude populations used to develop it."""
from pathlib import Path

from distilled_rnns.baseline_audit import BaselineAuditConfig
from distilled_rnns.config import ExperimentConfig
from distilled_rnns.followup import FollowupConfig
from distilled_rnns.oracle_control import OracleControlConfig

ROOT = Path(__file__).resolve().parents[1]


def expanded(seeds):
    return {seed + offset for seed in seeds for offset in range(4)}


def test_production_oracle_contrast_keeps_training_recipe_and_uses_fresh_populations():
    previous = BaselineAuditConfig.from_file(ROOT / "configs/robustness.json")
    current = OracleControlConfig.from_file(ROOT / "configs/oracle_control.json")
    expected_base = previous.base.to_dict()
    expected_base["seed"] = 20261411
    assert current.base.to_dict() == expected_base
    assert current.solo_l2 == .1
    assert current.replication_seeds == (20261411, 20261421, 20261431, 20261441, 20261451)
    assert current.max_seconds_per_invocation == 1800
    assert len(expanded(current.replication_seeds)) == 4 * len(current.replication_seeds)
    previous_seeds = {ExperimentConfig.from_file(ROOT / "configs/pilot.json").seed}
    for name, cls in (("followup", FollowupConfig), ("baseline_audit", BaselineAuditConfig),
                      ("robustness", BaselineAuditConfig)):
        prior = cls.from_file(ROOT / f"configs/{name}.json")
        previous_seeds.update((prior.base.seed, *prior.replication_seeds))
    assert expanded(current.replication_seeds).isdisjoint(expanded(previous_seeds))


def test_oracle_smoke_uses_same_contrast_with_smaller_workload():
    production = OracleControlConfig.from_file(ROOT / "configs/oracle_control.json")
    smoke = OracleControlConfig.from_file(ROOT / "configs/oracle_control_smoke.json")
    assert smoke.base.n_subjects == 10 and smoke.base.n_trials == 15
    assert smoke.base.teacher_steps == 2 and smoke.base.student_steps == 4
    assert smoke.solo_l2 == production.solo_l2
    assert smoke.base.teacher_prop == smoke.base.temperature == 1
    assert len(smoke.replication_seeds) == 2
    assert expanded(smoke.replication_seeds).isdisjoint(expanded(production.replication_seeds))
