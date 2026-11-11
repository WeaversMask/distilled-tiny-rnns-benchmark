"""Resumable, separately reported rarity cells and an immutable 2% baseline."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from distilled_rnns.budget_check import _reject_symlinks
from distilled_rnns.experiment import atomic_json, run_lock
from main_study import runner as core
from main_study.config import MainStudyConfig

from .config import CORE_SOURCE_SHA256, GridTrancheConfig, ROOT, grid_source_fingerprint
from .baseline import load_reference, reference_binding, snapshot_baseline, validate_baseline


def _same_json_prefix(path, expected):
    serialized = (json.dumps(expected, indent=2, allow_nan=False) + "\n").encode()
    return serialized.startswith(path.read_bytes())


def _output_path(output_dir):
    output = Path(output_dir).absolute()
    _reject_symlinks(output)
    output = output.resolve()
    for name in (
        "grid_study",
        "main_study",
        "foundation",
        "src",
        "configs",
        "scripts",
        "tests",
        "notebooks",
        "docs",
        "review",
    ):
        source = (ROOT / name).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output folder overlaps project source")
    return output


def _paths(parent_dir, output_dir):
    parent = Path(parent_dir).absolute()
    _reject_symlinks(parent)
    parent = parent.resolve()
    output = _output_path(output_dir)
    if output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("Baseline and new output folders must be separate")
    return parent, output


def _reference_hashes(folder):
    return {
        f"baseline_reference/{name}": core._sha(folder / name)
        for name in ("protocol.json", "summary.json", "verification.json")
    }


def _read_cell(config, folder):
    """Validate a completed engine output without opening a writer or refitting."""
    manifest_path = folder / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = core._json(manifest_path)
    identity = {
        "schema_version": 1,
        "study": "main_study_first_tranche",
        "config": config.to_dict(),
        "source_sha256": CORE_SOURCE_SHA256,
    }
    if manifest.get("identity") != identity:
        raise ValueError("Cell configuration or source changed")
    protocol = folder / "protocol.json"
    if core._json(protocol) != {**identity, "design": config.design}:
        raise ValueError("Cell prospective protocol changed")
    protocol_hash = core._sha(protocol)
    if manifest.get("immutable_file_sha256") != {"protocol.json": protocol_hash}:
        raise ValueError("Cell protocol hash list differs")
    core._checked_hashes(folder, manifest["immutable_file_sha256"])
    if manifest.get("status") != "complete":
        return None
    core._checked_hashes(folder, manifest["completed_file_sha256"])
    for item in config.plan:
        core._validate_complete_population(core.population_path(folder, **item), item, protocol_hash)
    summary = core._json(folder / "summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("config") != config.to_dict()
        or summary.get("planned_populations") != config.plan
        or summary.get("completed_populations") != config.plan
        or summary.get("pending_populations")
    ):
        raise ValueError("Completed cell summary differs from the fixed plan")
    return summary


def _normalize_initial_temp(actual, legacy, expected):
    found = actual.is_file() or legacy.is_file()
    if actual.is_file() and not _same_json_prefix(actual, expected):
        raise ValueError("Unexpected initial engine staging file: " + actual.name)
    if legacy.is_file() and not _same_json_prefix(legacy, expected):
        raise ValueError("Conflicting initial engine staging file: " + legacy.name)
    if actual.is_file():
        if legacy.exists():
            actual.unlink()
        else:
            actual.rename(legacy)
    return found


def _recover_engine_initialization(config, folder):
    """Translate validated actual atomic-json suffixes for the unchanged engine."""
    if not folder.exists():
        return []
    cleanup = []
    identity = {
        "schema_version": 1,
        "study": "main_study_first_tranche",
        "config": config.to_dict(),
        "source_sha256": CORE_SOURCE_SHA256,
    }
    protocol = {**identity, "design": config.design}
    with run_lock(folder):
        if (folder / "run_manifest.json").is_file() and core._json(folder / "run_manifest.json").get(
            "identity"
        ) != identity:
            raise ValueError("Engine configuration or source changed")
        if (folder / "protocol.json").is_file() and core._json(folder / "protocol.json") != protocol:
            raise ValueError("Engine prospective protocol changed")
        if not (folder / "run_manifest.json").exists():
            actual = folder / "protocol.json.tmp"
            if not (folder / "protocol.json").exists():
                if _normalize_initial_temp(actual, folder / "protocol.tmp", protocol):
                    cleanup.append((folder / "protocol.tmp", protocol))
            elif actual.exists():
                if not _same_json_prefix(actual, protocol):
                    raise ValueError("Unexpected temporary engine protocol")
                actual.unlink()
            if (folder / "run_manifest.json.tmp").exists() or (folder / "run_manifest.tmp").exists():
                if not (folder / "protocol.json").is_file():
                    raise ValueError("Temporary engine manifest has no verified protocol")
                if core._json(folder / "protocol.json") != protocol:
                    raise ValueError("Engine initialization protocol changed")
                initial = {
                    "identity": identity,
                    "status": "incomplete",
                    "invocations": [],
                    "immutable_file_sha256": {"protocol.json": core._sha(folder / "protocol.json")},
                }
                if _normalize_initial_temp(
                    folder / "run_manifest.json.tmp", folder / "run_manifest.tmp", initial
                ):
                    cleanup.append((folder / "run_manifest.tmp", initial))
        if (folder / "protocol.tmp").exists():
            if not _same_json_prefix(folder / "protocol.tmp", protocol):
                raise ValueError("Unexpected legacy temporary engine protocol")
            cleanup.append((folder / "protocol.tmp", protocol))
        if (folder / "protocol.json").is_file():
            protocol_hash = core._sha(folder / "protocol.json")
            initial = {
                "identity": identity,
                "status": "incomplete",
                "invocations": [],
                "immutable_file_sha256": {"protocol.json": protocol_hash},
            }
            if (folder / "run_manifest.tmp").exists():
                if not _same_json_prefix(folder / "run_manifest.tmp", initial):
                    raise ValueError("Unexpected legacy temporary engine manifest")
                cleanup.append((folder / "run_manifest.tmp", initial))
            for item in config.plan:
                child = core.population_path(folder, **item)
                expected = {
                    "identity": {**item, "protocol_sha256": protocol_hash},
                    "status": "incomplete",
                    "sessions": [],
                    "immutable_file_sha256": {},
                }
                if child.is_dir() and not (child / "manifest.json").exists():
                    if _normalize_initial_temp(child / "manifest.json.tmp", child / "manifest.tmp", expected):
                        cleanup.append((child / "manifest.tmp", expected))
                elif (child / "manifest.tmp").exists():
                    if not _same_json_prefix(child / "manifest.tmp", expected):
                        raise ValueError("Unexpected legacy temporary population manifest")
                    cleanup.append((child / "manifest.tmp", expected))
    return list({path: expected for path, expected in cleanup}.items())


def _remove_adopted_staging(cleanup):
    for path, expected in cleanup:
        if path.exists():
            if not _same_json_prefix(path, expected):
                raise ValueError("Adopted staging file changed")
            path.unlink()


def _initialize(folder, identity, baseline):
    expected_protocol = {**identity, "baseline_reference": reference_binding(baseline)}
    path = folder / "run_manifest.json"
    if path.is_file():
        manifest = core._json(path)
        if manifest.get("identity") != identity:
            raise ValueError("Rarity tranche configuration or source changed")
        core._checked_hashes(folder, manifest["immutable_file_sha256"])
        if core._json(folder / "protocol.json") != expected_protocol:
            raise ValueError("Rarity tranche baseline binding or protocol changed")
        return manifest
    allowed = {"baseline_reference", "protocol.json", "protocol.json.tmp", "run_manifest.json.tmp"}
    if {p.name for p in folder.iterdir()} - allowed:
        raise ValueError("Output folder contains files without this tranche identity")
    protocol_path = folder / "protocol.json"
    temporary = folder / "protocol.json.tmp"
    if protocol_path.exists() and core._json(protocol_path) != expected_protocol:
        raise ValueError("Orphaned rarity protocol differs")
    if temporary.exists() and not _same_json_prefix(temporary, expected_protocol):
        raise ValueError("Unexpected temporary rarity protocol")
    if not protocol_path.exists():
        if (folder / "run_manifest.json.tmp").exists():
            raise ValueError("Temporary manifest has no verified rarity protocol")
        atomic_json(protocol_path, expected_protocol)
    immutable = {
        "protocol.json": core._sha(protocol_path),
        **_reference_hashes(folder / "baseline_reference"),
    }
    manifest = {
        "identity": identity,
        "status": "incomplete",
        "immutable_file_sha256": immutable,
        "invocations": [],
    }
    temporary = folder / "run_manifest.json.tmp"
    if temporary.exists() and not _same_json_prefix(temporary, manifest):
        raise ValueError("Unexpected initial rarity manifest")
    atomic_json(path, manifest)
    (folder / "protocol.json.tmp").unlink(missing_ok=True)
    return manifest


def _timing(cells, baseline, invocations, complete):
    completed_population_seconds = sum(
        value["timing"]["cumulative_training_seconds"] for value in cells.values()
    )
    units = [value["timing"]["t4_equivalent_units_at_user_rate"] for value in cells.values()]
    known = [entry["wall_seconds"] for entry in invocations if entry.get("wall_seconds") is not None]
    return {
        "new_training_seconds": completed_population_seconds if complete else None,
        "completed_population_training_seconds": completed_population_seconds,
        "t4_equivalent_units_at_user_rate": sum(units)
        if complete and all(v is not None for v in units)
        else None,
        "baseline_training_seconds": baseline["summary"]["timing"]["cumulative_training_seconds"],
        "runner_wall_seconds": sum(known) if len(known) == len(invocations) else None,
        "known_runner_wall_seconds": sum(known),
        "invocation_wall_time_coverage": {
            "recorded_invocations": len(known),
            "total_invocations": len(invocations),
        },
        "note": "New checkpoint time counted once. Baseline compute is separate. Incomplete cells may contain additional saved work. Setup, preflight and idle billing are additional.",
    }


def _summary(config, folder, baseline, cells, invocations):
    complete = all(key in cells and cells[key]["status"] == "complete" for key in config.plan)
    completed = [key for key in config.plan if key in cells and cells[key]["status"] == "complete"]
    return {
        "status": "complete" if complete else "incomplete",
        "output_dir": str(folder),
        "config": config.to_dict(),
        "planned_cells": config.plan,
        "completed_cells": completed,
        "pending_cells": [key for key in config.plan if key not in completed],
        "cells": cells,
        "baseline_reference": baseline,
        "timing": _timing(cells, baseline, invocations, complete),
        "message": "Complete. Preserve the entire tranche and smoke folders and disconnect the GPU."
        if complete
        else "Rerun only the fourth code cell with the same run name. At most one pending rarity cell advances per invocation.",
    }


def run_baseline_smoke(config, output_dir, device="auto"):
    """Run only the original tiny baseline fixture with safe startup retry."""
    expected = MainStudyConfig.from_file(ROOT / "configs/main_study_smoke.json")
    if (
        type(config) is not MainStudyConfig
        or config.mode != "smoke"
        or config.to_dict() != expected.to_dict()
    ):
        raise ValueError("Baseline generation requires the exact original smoke configuration")
    folder = _output_path(output_dir)
    manifest = folder / "run_manifest.json"
    if manifest.is_file() and core._json(manifest).get("status") == "complete":
        return core.run_main_study(config, folder, device=device)
    cleanup = _recover_engine_initialization(config, folder)
    try:
        return core.run_main_study(config, folder, device=device)
    finally:
        with run_lock(folder):
            _remove_adopted_staging(cleanup)


def run_rarity_tranche(config, parent_dir, output_dir, device="auto"):
    started = time.perf_counter()
    if not isinstance(config, GridTrancheConfig):
        raise ValueError("config must be GridTrancheConfig")
    parent, folder = _paths(parent_dir, output_dir)
    # Parent validation is deliberately read-only and precedes any child write.
    baseline = validate_baseline(parent, mode=config.mode)
    identity = {
        "schema_version": 1,
        "study": "main_study_rarity_tranche",
        "config": config.to_dict(),
        "grid_source_sha256": grid_source_fingerprint(),
        "core_source_sha256": CORE_SOURCE_SHA256,
    }
    folder.mkdir(parents=True, exist_ok=True)
    with run_lock(folder):
        manifest_path = folder / "run_manifest.json"
        if manifest_path.exists():
            preview = core._json(manifest_path)
            if preview.get("identity") != identity:
                raise ValueError("Rarity tranche configuration or source changed")
            core._checked_hashes(folder, preview["immutable_file_sha256"])
            reference = load_reference(folder / "baseline_reference", mode=config.mode)
            if reference_binding(reference) != reference_binding(baseline):
                raise ValueError("Baseline changed after this tranche was initialized")
        else:
            allowed = {"baseline_reference", "protocol.json", "protocol.json.tmp", "run_manifest.json.tmp"}
            if {p.name for p in folder.iterdir()} - allowed:
                raise ValueError("Output folder contains files without this tranche identity")
            expected_protocol = {**identity, "baseline_reference": reference_binding(baseline)}
            if (folder / "protocol.json").exists() and core._json(
                folder / "protocol.json"
            ) != expected_protocol:
                raise ValueError("Orphaned rarity protocol differs")
            if (folder / "protocol.json.tmp").exists() and not _same_json_prefix(
                folder / "protocol.json.tmp", expected_protocol
            ):
                raise ValueError("Unexpected temporary rarity protocol")
            if (folder / "run_manifest.json.tmp").exists() and not (folder / "protocol.json").exists():
                raise ValueError("Temporary manifest has no verified rarity protocol")
            snapshot_baseline(parent, folder / "baseline_reference", mode=config.mode)
        manifest = _initialize(folder, identity, baseline)
        if manifest["status"] == "complete":
            actual_files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            if actual_files != set(manifest["completed_file_sha256"]) | {"run_manifest.json"}:
                raise ValueError("Completed rarity tranche file set changed")
            core._checked_hashes(folder, manifest["completed_file_sha256"])
            for cell in config.cell_configs:
                if _read_cell(cell, folder / "cells" / cell.cell_id) is None:
                    raise ValueError("Completed rarity tranche contains an incomplete cell")
            summary = core._json(folder / "summary.json")
            if (
                summary.get("status") != "complete"
                or summary.get("planned_cells") != config.plan
                or summary.get("pending_cells")
            ):
                raise ValueError("Completed rarity summary differs from the fixed plan")
            return summary
        invocations = manifest["invocations"]
        for prior in invocations:
            if "wall_seconds" not in prior:
                prior.update(status="interrupted", wall_seconds=None)
        invocation = {"started_utc": datetime.now(timezone.utc).isoformat()}
        invocations.append(invocation)
        atomic_json(manifest_path, manifest)
        cells = {}
        pending = None
        for cell in config.cell_configs:
            path = folder / "cells" / cell.cell_id
            summary = _read_cell(cell, path)
            if summary is not None:
                _remove_adopted_staging(_recover_engine_initialization(cell, path))
                cells[cell.cell_id] = summary
            elif pending is None:
                pending = cell
        if pending is not None:
            path = folder / "cells" / pending.cell_id
            cleanup = _recover_engine_initialization(pending, path)
            invocation["cell_id"] = pending.cell_id
            try:
                cells[pending.cell_id] = core.run_main_study(pending, path, device=device)
            finally:
                with run_lock(path):
                    _remove_adopted_staging(cleanup)
        invocation.update(status="complete", wall_seconds=time.perf_counter() - started)
        summary = _summary(config, folder, baseline, cells, invocations)
        atomic_json(folder / "summary.json", summary)
        if summary["status"] == "complete":
            paths = [p for p in folder.rglob("*") if p.is_file() and p != manifest_path]
            manifest.update(
                status="complete",
                completed_file_sha256={p.relative_to(folder).as_posix(): core._sha(p) for p in paths},
            )
        atomic_json(manifest_path, manifest)
        return summary
