"""Read-only baseline trust, context portability and interrupted snapshot recovery."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from main_study.config import MainStudyConfig
from main_study.runner import run_main_study
from grid_study import baseline


def file_hashes(folder):
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }


@pytest.fixture(scope="module")
def completed_parent(tmp_path_factory):
    path = tmp_path_factory.mktemp("rarity-baseline-parent").resolve()
    config = MainStudyConfig.from_file(baseline.ROOT / "configs/main_study_smoke.json")
    assert run_main_study(config, path, "cpu")["status"] == "complete"
    return path


def test_production_trust_matches_previously_reviewed_entire_download():
    reference = baseline._trusted_reference()
    audit = json.loads((baseline.ROOT / "review/main-study-integrity-verification.json").read_text())[
        "production"
    ]
    assert reference["file_sha256"] == audit["file_sha256_before"]
    assert len(reference["file_sha256"]) == 87
    config = MainStudyConfig.from_file(baseline.ROOT / "configs/main_study_first_tranche.json")
    assert reference["config"] == config.to_dict()
    assert reference["planned_populations"] == config.plan


def test_validate_and_portable_snapshot_never_modify_original(completed_parent, tmp_path, monkeypatch):
    before = file_hashes(completed_parent)
    monkeypatch.setattr(
        baseline.core, "run_main_study", lambda *a, **k: pytest.fail("Validator invoked a runner")
    )
    audit = baseline.validate_baseline(completed_parent, "smoke")
    reference = tmp_path / "context"
    assert baseline.snapshot_baseline(completed_parent, reference, "smoke") == audit
    assert set(p.name for p in reference.iterdir()) == baseline.REFERENCE_FILES
    assert (reference / "summary.json").read_bytes() == (completed_parent / "summary.json").read_bytes()
    copied = tmp_path / "moved-reference"
    shutil.copytree(reference, copied)
    assert baseline.load_reference(copied, "smoke") == audit
    reference_before = file_hashes(reference)
    monkeypatch.setattr(
        baseline, "_atomic_payload", lambda *a, **k: pytest.fail("Completed reference was rewritten")
    )
    assert baseline.snapshot_baseline(completed_parent, reference, "smoke") == audit
    assert file_hashes(reference) == reference_before
    assert file_hashes(completed_parent) == before


def test_smoke_cannot_replace_the_frozen_production_parent(completed_parent):
    before = file_hashes(completed_parent)
    with pytest.raises(ValueError, match="87 files"):
        baseline.validate_baseline(completed_parent, "production")
    assert file_hashes(completed_parent) == before


def test_changed_or_incomplete_parent_rejected_without_mutation(completed_parent, tmp_path):
    parent = tmp_path / "changed"
    shutil.copytree(completed_parent, parent)
    prediction = next(parent.glob("*/seed-*/predictions.npz"))
    prediction.write_bytes(prediction.read_bytes() + b"changed")
    before = file_hashes(parent)
    with pytest.raises(ValueError, match="missing or changed"):
        baseline.validate_baseline(parent, "smoke")
    assert file_hashes(parent) == before
    parent = tmp_path / "incomplete"
    shutil.copytree(completed_parent, parent)
    summary = parent / "summary.json"
    values = json.loads(summary.read_text())
    values["status"] = "incomplete"
    summary.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="all be complete"):
        baseline.validate_baseline(parent, "smoke")


def test_reference_corruption_and_committed_missing_payload_rejected(completed_parent, tmp_path):
    target = tmp_path / "reference"
    baseline.snapshot_baseline(completed_parent, target, "smoke")
    summary = target / "summary.json"
    summary.write_bytes(summary.read_bytes() + b"\n")
    before = file_hashes(target)
    with pytest.raises(ValueError, match="differs"):
        baseline.load_reference(target, "smoke")
    with pytest.raises(ValueError, match="differs"):
        baseline.snapshot_baseline(completed_parent, target, "smoke")
    assert file_hashes(target) == before
    summary.unlink()
    with pytest.raises(ValueError, match="three complete files"):
        baseline.snapshot_baseline(completed_parent, target, "smoke")


def test_snapshot_retries_partial_atomic_copy(completed_parent, tmp_path, monkeypatch):
    target = tmp_path / "partial"
    before = file_hashes(completed_parent)
    original = baseline._atomic_payload

    def interrupted(path, payload):
        if path.name == "protocol.json":
            path.with_name(path.name + ".tmp").write_bytes(payload[:21])
            raise RuntimeError("simulated disconnect")
        original(path, payload)

    monkeypatch.setattr(baseline, "_atomic_payload", interrupted)
    with pytest.raises(RuntimeError, match="disconnect"):
        baseline.snapshot_baseline(completed_parent, target, "smoke")
    assert (target / "summary.json").exists() and not (target / "verification.json").exists()
    monkeypatch.setattr(baseline, "_atomic_payload", original)
    result = baseline.snapshot_baseline(completed_parent, target, "smoke")
    assert result == baseline.load_reference(target, "smoke")
    assert file_hashes(completed_parent) == before


def test_unknown_partial_copy_overlap_and_symlinks_rejected(completed_parent, tmp_path):
    target = tmp_path / "unknown"
    target.mkdir()
    (target / "protocol.json.tmp").write_bytes(b"unrelated")
    before = file_hashes(target)
    with pytest.raises(ValueError, match="Unexpected temporary"):
        baseline.snapshot_baseline(completed_parent, target, "smoke")
    assert file_hashes(target) == before
    with pytest.raises(ValueError, match="overlap"):
        baseline.snapshot_baseline(completed_parent, completed_parent / "reference", "smoke")
    alias = tmp_path / "alias"
    alias.symlink_to(completed_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        baseline.validate_baseline(alias, "smoke")
    with pytest.raises(ValueError, match="overlaps source"):
        baseline.snapshot_baseline(completed_parent, baseline.ROOT / "grid_study" / "invalid", "smoke")
