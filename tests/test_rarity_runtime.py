"""Real adapter resumption and scientific-cell separation checks."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from distilled_rnns.experiment import run_lock
from main_study.config import MainStudyConfig
from main_study import runner as core
from grid_study.config import CORE_SOURCE_SHA256, GridCellConfig, GridTrancheConfig, grid_source_fingerprint
from grid_study import runner


def hashes(folder):
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    folder = tmp_path_factory.mktemp("rarity-parent").resolve()
    summary = core.run_main_study(
        MainStudyConfig(mode="smoke", max_seconds_per_phase=120, max_seconds_per_invocation=120),
        folder,
        "cpu",
    )
    assert summary["status"] == "complete"
    return folder


@pytest.fixture(scope="module")
def finished(parent, tmp_path_factory):
    folder = tmp_path_factory.mktemp("rarity-finished").resolve()
    config = GridTrancheConfig(mode="smoke", max_seconds_per_phase=120, max_seconds_per_invocation=120)
    before = hashes(parent)
    outputs = [runner.run_rarity_tranche(config, parent, folder, "cpu") for _ in range(3)]
    assert [len(result["completed_cells"]) for result in outputs] == [1, 2, 3]
    assert hashes(parent) == before
    return config, folder, outputs[-1]


def test_production_schedule_exactly_matches_unseen_frozen_cells():
    config = GridTrancheConfig()
    assert config.plan == ["rarity-05_history-100", "rarity-10_history-100", "rarity-25_history-100"]
    all_seeds = []
    for rarity_index, cell in zip((1, 2, 3), config.cell_configs):
        assert cell.n_subjects == 1000 and cell.n_trials == 100
        assert cell.rare_fraction == (0.05, 0.1, 0.25)[rarity_index - 1]
        assert cell.student_steps == 4000 and cell.teacher_steps == 500
        expected = [
            {"kind": kind, "seed": 202611401 + 10000 * kind_index + 1000 * rarity_index + 10 * rep}
            for kind_index, kind in enumerate(cell.design["population"]["rare_kinds"])
            for rep in range(3)
        ]
        assert cell.plan == expected
        all_seeds.extend(row["seed"] for row in cell.plan)
    assert len(set(all_seeds)) == 18
    assert not set(all_seeds) & {row["seed"] for row in MainStudyConfig().plan}
    for index in (True, 1.0, 0, 4):
        with pytest.raises(ValueError):
            GridCellConfig(rarity_index=index)
    with pytest.raises(ValueError):
        GridCellConfig(history_index=1)


def test_all_cells_report_separately_and_baseline_compute_is_inherited(finished, parent):
    config, folder, result = finished
    assert result["status"] == "complete" and result["pending_cells"] == []
    assert result["completed_cells"] == result["planned_cells"] == config.plan
    assert "groups" not in result and "kinds" not in result
    assert result["baseline_reference"]["summary"] == json.loads((parent / "summary.json").read_text())
    expected = sum(value["timing"]["cumulative_training_seconds"] for value in result["cells"].values())
    assert result["timing"]["new_training_seconds"] == expected
    assert (
        result["timing"]["baseline_training_seconds"]
        == result["baseline_reference"]["summary"]["timing"]["cumulative_training_seconds"]
    )
    assert result["timing"]["t4_equivalent_units_at_user_rate"] is None
    for cell in config.cell_configs:
        value = result["cells"][cell.cell_id]
        assert value["completed_populations"] == cell.plan
        assert value["config"]["grid_source_sha256"] == grid_source_fingerprint()
        for kind in value["kinds"].values():
            assert (
                kind["groups"]["rare"]["primary"]["population"]["prediction_gain"]["planned_seed_count"] == 2
            )


def test_completed_outer_resume_is_immutable_and_does_not_refit(finished, parent, monkeypatch):
    config, folder, result = finished
    before = hashes(folder)
    parent_before = hashes(parent)
    monkeypatch.setattr(
        core, "run_main_study", lambda *a, **k: pytest.fail("Completed cell entered the writer")
    )
    assert runner.run_rarity_tranche(config, parent, folder, "cpu") == result
    assert hashes(folder) == before and hashes(parent) == parent_before


def test_corruption_and_changed_source_are_rejected_before_mutation(finished, parent, tmp_path, monkeypatch):
    config, folder, _ = finished
    corrupted = tmp_path / "corrupted"
    shutil.copytree(folder, corrupted)
    file = next(corrupted.glob("cells/*/*/seed-*/predictions.npz"))
    file.write_bytes(file.read_bytes() + b"corrupt")
    before = hashes(corrupted)
    with pytest.raises(ValueError, match="missing or changed"):
        runner.run_rarity_tranche(config, parent, corrupted, "cpu")
    assert hashes(corrupted) == before
    before = hashes(folder)
    monkeypatch.setattr(runner, "grid_source_fingerprint", lambda: "0" * 64)
    with pytest.raises(ValueError, match="configuration or source changed"):
        runner.run_rarity_tranche(config, parent, folder, "cpu")
    assert hashes(folder) == before


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_interrupted_cell_restores_current_optimizer_and_keeps_all_source_bytes(
    finished, parent, tmp_path, monkeypatch
):
    config, reference, _ = finished
    folder = (tmp_path / "resumed").resolve()
    original_fit = core.fit_model

    def pause_solo(model, *args, **kwargs):
        if kwargs["l2_strength"]:
            kwargs["steps"] = 1
            result = original_fit(model, *args, **kwargs)
            result["stopped_reason"] = "time_limit"
            return result
        return original_fit(model, *args, **kwargs)

    monkeypatch.setattr(core, "fit_model", pause_solo)
    partial = runner.run_rarity_tranche(config, parent, folder, "cpu")
    assert partial["status"] == "incomplete" and not partial["completed_cells"]
    assert partial["timing"]["new_training_seconds"] is None
    assert all("groups" not in kind for kind in partial["cells"][config.plan[0]]["kinds"].values())
    monkeypatch.setattr(core, "fit_model", original_fit)
    for _ in range(3):
        final = runner.run_rarity_tranche(config, parent, folder, "cpu")
    assert final["status"] == "complete"
    for cell in config.cell_configs:
        assert (
            final["cells"][cell.cell_id]["kinds"]
            == json.loads((reference / "cells" / cell.cell_id / "summary.json").read_text())["kinds"]
        )
        for item in cell.plan:
            first = core.population_path(reference / "cells" / cell.cell_id, **item)
            second = core.population_path(folder / "cells" / cell.cell_id, **item)
            for arm in core.ARMS:
                a = torch.load(first / (arm + ".pt"), weights_only=True, map_location="cpu")
                b = torch.load(second / (arm + ".pt"), weights_only=True, map_location="cpu")
                for key in (
                    "signature",
                    "current_state",
                    "best_state",
                    "optimizer",
                    "completed_steps",
                    "history",
                    "rng_state",
                ):
                    assert_nested_equal(a[key], b[key])


@pytest.mark.parametrize(
    "target", ["outer_protocol", "outer_manifest", "cell_protocol", "cell_manifest", "population_manifest"]
)
def test_real_atomic_initialization_write_interruptions_recover(parent, tmp_path, monkeypatch, target):
    config = GridTrancheConfig(mode="smoke", max_seconds_per_phase=120, max_seconds_per_invocation=120)
    folder = (tmp_path / target).resolve()
    original = core.os.replace
    interrupted = False

    def crash_once(source, destination):
        nonlocal interrupted
        destination = Path(destination)
        relative = destination.relative_to(folder) if destination.is_relative_to(folder) else None
        match = {
            "outer_protocol": relative == Path("protocol.json"),
            "outer_manifest": relative == Path("run_manifest.json"),
            "cell_protocol": relative == Path("cells") / config.plan[0] / "protocol.json",
            "cell_manifest": relative == Path("cells") / config.plan[0] / "run_manifest.json",
            "population_manifest": relative
            == Path("cells")
            / config.plan[0]
            / config.cell_configs[0].plan[0]["kind"]
            / ("seed-" + str(config.cell_configs[0].plan[0]["seed"]))
            / "manifest.json",
        }[target]
        if match and not interrupted:
            interrupted = True
            assert str(source).endswith(".json.tmp")
            raise RuntimeError("Injected actual atomic write interruption")
        return original(source, destination)

    monkeypatch.setattr(core.os, "replace", crash_once)
    with pytest.raises(RuntimeError, match="Injected actual"):
        runner.run_rarity_tranche(config, parent, folder, "cpu")
    monkeypatch.setattr(core.os, "replace", original)
    result = runner.run_rarity_tranche(config, parent, folder, "cpu")
    assert result["completed_cells"] == config.plan[:1]
    assert not list((folder / "cells").rglob("*.tmp"))
    assert (
        result["timing"]["runner_wall_seconds"] is None
        if target.startswith("cell_") or target == "population_manifest"
        else result["timing"]["runner_wall_seconds"] is not None
    )


def test_paths_unknown_outputs_and_concurrent_writers_preserve_existing_files(parent, tmp_path):
    config = GridTrancheConfig(mode="smoke")
    with pytest.raises(ValueError, match="must be separate"):
        runner.run_rarity_tranche(config, parent, parent / "child", "cpu")
    with pytest.raises(ValueError, match="overlaps project source"):
        runner.run_rarity_tranche(config, parent, runner.ROOT / "grid_study" / "bad", "cpu")
    folder = (tmp_path / "unknown").resolve()
    folder.mkdir()
    (folder / "user.txt").write_text("keep")
    before = hashes(folder)
    with pytest.raises(ValueError, match="without this tranche"):
        runner.run_rarity_tranche(config, parent, folder, "cpu")
    assert hashes(folder) == before
    with run_lock(folder):
        with pytest.raises(RuntimeError, match="another process"):
            runner.run_rarity_tranche(config, parent, folder, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_rarity_smoke_and_completed_resume(parent, tmp_path):
    config = GridTrancheConfig(mode="smoke", max_seconds_per_phase=120, max_seconds_per_invocation=120)
    folder = tmp_path.resolve() / "cuda-rarity"
    for _ in range(3):
        summary = runner.run_rarity_tranche(config, parent, folder, "cuda")
    before = hashes(folder)
    assert summary["status"] == "complete"
    assert runner.run_rarity_tranche(config, parent, folder, "cuda") == summary
    assert hashes(folder) == before


def test_adopted_initial_staging_is_cleaned_when_training_then_raises(parent, tmp_path, monkeypatch):
    config = GridTrancheConfig(mode="smoke", max_seconds_per_phase=120, max_seconds_per_invocation=120)
    folder = (tmp_path / "nested-interrupt").resolve()
    original_replace = core.os.replace
    stopped = False

    def crash_manifest(source, destination):
        nonlocal stopped
        destination = Path(destination)
        if destination.name == "manifest.json" and "seed-" in str(destination) and not stopped:
            stopped = True
            raise RuntimeError("first actual manifest staging")
        return original_replace(source, destination)

    monkeypatch.setattr(core.os, "replace", crash_manifest)
    with pytest.raises(RuntimeError, match="first actual"):
        runner.run_rarity_tranche(config, parent, folder, "cpu")
    monkeypatch.setattr(core.os, "replace", original_replace)
    original_fit = core.fit_model
    monkeypatch.setattr(
        core, "fit_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("later fitting interruption"))
    )
    with pytest.raises(RuntimeError, match="later fitting"):
        runner.run_rarity_tranche(config, parent, folder, "cpu")
    assert not list((folder / "cells").rglob("manifest.tmp"))
    monkeypatch.setattr(core, "fit_model", original_fit)
    assert runner.run_rarity_tranche(config, parent, folder, "cpu")["completed_cells"] == config.plan[:1]


def test_baseline_smoke_helper_refuses_production_and_grid_configs(tmp_path):
    path = (tmp_path / "forbidden-production").resolve()
    for config in (MainStudyConfig(), GridCellConfig(mode="smoke"), MainStudyConfig(mode="smoke")):
        with pytest.raises(ValueError, match="exact original smoke"):
            runner.run_baseline_smoke(config, path, "cpu")
        assert not path.exists()


@pytest.mark.parametrize("destination", ["protocol.json", "run_manifest.json", "manifest.json"])
def test_original_baseline_smoke_real_atomic_initialization_recovers(tmp_path, monkeypatch, destination):
    config = MainStudyConfig.from_file(runner.ROOT / "configs/main_study_smoke.json")
    folder = (tmp_path / "original-smoke").resolve()
    original = core.os.replace
    crashed = False

    def crash_once(source, target):
        nonlocal crashed
        target = Path(target)
        if target.name == destination and not crashed:
            crashed = True
            assert str(source).endswith(".json.tmp")
            raise RuntimeError("baseline initial atomic interruption")
        return original(source, target)

    monkeypatch.setattr(core.os, "replace", crash_once)
    with pytest.raises(RuntimeError, match="baseline initial"):
        runner.run_baseline_smoke(config, folder, "cpu")
    monkeypatch.setattr(core.os, "replace", original)
    result = runner.run_baseline_smoke(config, folder, "cpu")
    assert result["status"] == "complete"
    assert not list(folder.rglob("*.tmp"))
    before = hashes(folder)
    assert runner.run_baseline_smoke(config, folder, "cpu") == result
    assert hashes(folder) == before
