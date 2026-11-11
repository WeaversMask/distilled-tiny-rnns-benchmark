"""Guard the fresh, predeclared robustness configuration and supported loader."""
from pathlib import Path

from distilled_rnns.baseline_audit import BaselineAuditConfig
from distilled_rnns.config import ExperimentConfig
from distilled_rnns.followup import FollowupConfig

ROOT = Path(__file__).resolve().parents[1]


def used_seeds(config):
    # Simulation, split, teacher initialization, and student initialization.
    seeds = [config.base.seed, *config.replication_seeds]
    expanded = [seed + offset for seed in seeds for offset in range(4)]
    assert len(expanded) == len(set(expanded))
    return set(expanded)


def test_robustness_is_fresh_and_changes_only_declared_design_settings():
    baseline = BaselineAuditConfig.from_file(ROOT / "configs/baseline_audit.json")
    current = BaselineAuditConfig.from_file(ROOT / "configs/robustness.json")
    expected = baseline.to_dict()
    expected["base"].update(seed=20261301, student_steps=4000)
    expected["solo_l2_grid"] = [0, .001, .01, .1, .3, 1, 3]
    expected["replication_seeds"] = [20261311, 20261321, 20261331, 20261341, 20261351]
    assert current.to_dict() == expected
    prior = used_seeds(baseline)
    prior |= used_seeds(FollowupConfig.from_file(ROOT / "configs/followup.json"))
    pilot = ExperimentConfig.from_file(ROOT / "configs/pilot.json")
    prior |= {pilot.seed + i for i in range(4)}
    assert used_seeds(current).isdisjoint(prior)


def test_robustness_smoke_exercises_extended_penalty_and_distinct_output_identity():
    smoke = BaselineAuditConfig.from_file(ROOT / "configs/robustness_smoke.json")
    old = BaselineAuditConfig.from_file(ROOT / "configs/baseline_audit_smoke.json")
    assert max(smoke.solo_l2_grid) == 3
    assert smoke.legacy_solo_l2 in smoke.solo_l2_grid
    assert smoke.base.student_steps > old.base.student_steps
    assert used_seeds(smoke).isdisjoint(used_seeds(old))
    assert smoke.base.n_subjects <= 10 and len(smoke.replication_seeds) == 2
