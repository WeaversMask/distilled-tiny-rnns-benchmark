"""Frozen scientific design and bounded runtime settings for notebook 07."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = '1e8762df6fc01919dd3cfc432e96b78ed1d3fe1a25c26859edd9674cd501821e'


def load_design():
    payload = (ROOT / 'configs/main_study_v1.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('Prospective protocol changed. Preserve version 1 and declare a new protocol before evaluation.')
    return json.loads(payload)


@dataclass(frozen=True)
class MainStudyConfig:
    mode: str = 'production'
    max_seconds_per_phase: float = 600.0
    max_seconds_per_invocation: float = 1800.0

    def __post_init__(self):
        if self.mode not in ('production', 'smoke'):
            raise ValueError('mode must be production or smoke')
        for name in ('max_seconds_per_phase', 'max_seconds_per_invocation'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be positive and finite')
        load_design()

    @classmethod
    def from_file(cls, path):
        settings = json.loads(Path(path).read_text())
        if not isinstance(settings, dict) or set(settings) != {'mode', 'max_seconds_per_phase', 'max_seconds_per_invocation'}:
            raise ValueError('Only mode and the two runtime time budgets may be configured')
        return cls(**settings)

    @property
    def design(self):
        return load_design()

    @property
    def plan(self):
        if self.mode == 'smoke':
            return [{'kind': kind, 'seed': seed} for kind, seeds in
                    (('parametric_model_free', (701, 711)), ('reward_cue', (801, 811))) for seed in seeds]
        return [{'kind': kind, 'seed': 202611401 + 10000 * index + 10 * replication}
                for index, kind in enumerate(self.design['population']['rare_kinds']) for replication in range(3)]

    @property
    def n_subjects(self):
        return 10 if self.mode == 'smoke' else self.design['population']['subjects']

    @property
    def n_trials(self):
        return 15 if self.mode == 'smoke' else self.design['release_order']['first_tranche_history']

    @property
    def rare_fraction(self):
        return .2 if self.mode == 'smoke' else self.design['release_order']['first_tranche_rarity']

    @property
    def teacher_hidden(self):
        return 4 if self.mode == 'smoke' else self.design['neural_recipe']['teacher_hidden']

    @property
    def student_hidden(self):
        return self.design['neural_recipe']['student_hidden']

    @property
    def teacher_steps(self):
        return 2 if self.mode == 'smoke' else self.design['neural_recipe']['teacher_steps_maximum']

    @property
    def student_steps(self):
        return 4 if self.mode == 'smoke' else self.design['neural_recipe']['student_steps']

    @property
    def eval_every(self):
        return 1 if self.mode == 'smoke' else self.design['neural_recipe']['eval_every']

    @property
    def learning_rate(self):
        return self.design['neural_recipe']['learning_rate']

    @property
    def patience(self):
        return self.design['neural_recipe']['teacher_patience']

    @property
    def solo_l2(self):
        return self.design['neural_recipe']['solo_l2']

    @property
    def temperature(self):
        return self.design['neural_recipe']['temperature']

    @property
    def diagnostic_threshold(self):
        return self.design['secondary_outcomes']['diagnostic_threshold']

    @property
    def t4_units_per_hour(self):
        return 1.07

    def to_dict(self):
        fields = ('mode', 'max_seconds_per_phase', 'max_seconds_per_invocation', 'n_subjects', 'n_trials',
                  'rare_fraction', 'teacher_hidden', 'student_hidden', 'teacher_steps', 'student_steps',
                  'eval_every', 'learning_rate', 'patience', 'solo_l2', 'temperature', 'diagnostic_threshold',
                  't4_units_per_hour', 'plan')
        return {**{name: getattr(self, name) for name in fields}, 'protocol_sha256': PROTOCOL_SHA256}
