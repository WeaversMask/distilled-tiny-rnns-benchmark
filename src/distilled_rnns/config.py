"""Explicit, serializable settings for bounded experiments."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    n_subjects: int = 1000
    n_trials: int = 100
    rare_fraction: float = 0.02
    seed: int = 20261029
    teacher_hidden: int = 32
    student_hidden: int = 2
    teacher_steps: int = 100
    student_steps: int = 150
    learning_rate: float = 0.01
    teacher_prop: float = 1.0
    temperature: float = 1.0
    eval_every: int = 10
    patience: int = 20
    max_seconds_per_phase: float = 600.0
    diagnostic_threshold: float = 0.2
    t4_units_per_hour: float = 1.07

    def __post_init__(self):
        for name in ("n_subjects", "n_trials", "teacher_hidden", "student_hidden",
                     "teacher_steps", "student_steps", "eval_every", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32 - 10:
            raise ValueError("seed must be an integer in [0, 2**32 - 10)")
        if self.n_trials < 10:
            raise ValueError("At least 10 trials are required for train/validation/test splits")
        for name in ("rare_fraction", "learning_rate", "teacher_prop", "temperature",
                     "max_seconds_per_phase", "diagnostic_threshold", "t4_units_per_hour"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and numeric")
        if not 0 < self.rare_fraction < 1 or not 0 < math.floor(self.n_subjects * self.rare_fraction + 0.5) < self.n_subjects:
            raise ValueError("The population must include at least one rare and one typical subject")
        if not 0 <= self.teacher_prop <= 1:
            raise ValueError("teacher_prop must be in [0, 1]")
        if not 0 < self.diagnostic_threshold < 1:
            raise ValueError("diagnostic_threshold must be in (0, 1)")
        if any(getattr(self, k) <= 0 for k in ("learning_rate", "temperature", "max_seconds_per_phase", "t4_units_per_hour")):
            raise ValueError("Rates, temperature, and time budget must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_file(cls, path):
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration settings: {sorted(unknown)}")
        return cls(**values)
