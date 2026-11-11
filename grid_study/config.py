"""Fixed 100-trial rarity tranche and unchanged neural fitting recipe."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from main_study.config import MainStudyConfig, PROTOCOL_SHA256
from main_study.runner import main_source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE_SHA256 = "b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372"
RARITY_INDICES = (1, 2, 3)


def grid_source_fingerprint():
    """Bind every numerical module, this adapter, and the trusted baseline map."""
    if main_source_fingerprint() != CORE_SOURCE_SHA256:
        raise ValueError("The frozen main-study numerical engine changed")
    paths = [
        path
        for folder in ("main_study", "foundation", "src/distilled_rnns", "grid_study")
        for path in (ROOT / folder).glob("*.py")
    ]
    paths += [ROOT / "configs/main_study_v1.json", ROOT / "configs/grid_baseline_reference_v1.json"]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.relative_to(ROOT).as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class GridCellConfig(MainStudyConfig):
    rarity_index: int = 1
    history_index: int = 0

    def __post_init__(self):
        super().__post_init__()
        if (
            isinstance(self.rarity_index, bool)
            or not isinstance(self.rarity_index, int)
            or self.rarity_index not in RARITY_INDICES
        ):
            raise ValueError("This tranche uses only rarity indices 1, 2 and 3")
        if (
            isinstance(self.history_index, bool)
            or not isinstance(self.history_index, int)
            or self.history_index != 0
        ):
            raise ValueError("This tranche uses only history index zero")

    @property
    def cell_id(self):
        fraction = self.design["population"]["rarity_levels"][self.rarity_index]
        trials = self.design["population"]["history_levels"][self.history_index]
        return f"rarity-{round(100 * fraction):02d}_history-{trials}"

    @property
    def rare_fraction(self):
        if self.mode == "smoke":
            return (0.3, 0.4, 0.5)[self.rarity_index - 1]
        return self.design["population"]["rarity_levels"][self.rarity_index]

    @property
    def n_trials(self):
        if self.mode == "smoke":
            return 15
        return self.design["population"]["history_levels"][self.history_index]

    @property
    def plan(self):
        kinds = self.design["population"]["rare_kinds"]
        if self.mode == "smoke":
            return [
                {"kind": kind, "seed": 81001 + 1000 * (self.rarity_index - 1) + 100 * index + 10 * rep}
                for index, kind in enumerate(kinds)
                for rep in range(2)
            ]
        return [
            {
                "kind": kind,
                "seed": 202611401
                + 10000 * index
                + 1000 * self.rarity_index
                + 100 * self.history_index
                + 10 * rep,
            }
            for index, kind in enumerate(kinds)
            for rep in range(3)
        ]

    def to_dict(self):
        return {
            **super().to_dict(),
            "adapter_schema_version": 1,
            "cell_id": self.cell_id,
            "rarity_index": self.rarity_index,
            "history_index": self.history_index,
            "grid_source_sha256": grid_source_fingerprint(),
        }


@dataclass(frozen=True)
class GridTrancheConfig(MainStudyConfig):
    """Runtime budgets and a sealed three-cell prospective schedule."""

    @property
    def cell_configs(self):
        return tuple(
            GridCellConfig(
                mode=self.mode,
                max_seconds_per_phase=self.max_seconds_per_phase,
                max_seconds_per_invocation=self.max_seconds_per_invocation,
                rarity_index=index,
                history_index=0,
            )
            for index in RARITY_INDICES
        )

    @property
    def plan(self):
        return [cell.cell_id for cell in self.cell_configs]

    def to_dict(self):
        return {
            "schema_version": 1,
            "mode": self.mode,
            "max_seconds_per_phase": self.max_seconds_per_phase,
            "max_seconds_per_invocation": self.max_seconds_per_invocation,
            "protocol_sha256": PROTOCOL_SHA256,
            "core_source_sha256": CORE_SOURCE_SHA256,
            "grid_source_sha256": grid_source_fingerprint(),
            "cells": [cell.to_dict() for cell in self.cell_configs],
            "scheduling": "At most one pending cell is advanced per invocation; no pooling across rarity.",
        }
