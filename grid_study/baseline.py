"""Read-only validation and portable context for the verified two-percent cell."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from main_study.config import MainStudyConfig, PROTOCOL_SHA256
from main_study import runner as core

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SOURCE_SHA256 = "b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372"
REFERENCE_SHA256 = "d6e2494971c9f5457553cd8cf91c3d8c87ba7082ed6704a59cb2dcd7900e7e36"
REFERENCE_FILES = {"summary.json", "protocol.json", "verification.json"}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(path):
    return json.loads(path.read_bytes())


def _canonical(path):
    requested = Path(path).absolute()
    core._reject_symlinks(requested)
    return requested.resolve()


def _file_hashes(folder):
    return {
        path.relative_to(folder).as_posix(): core._sha(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _trusted_reference():
    payload = (ROOT / "configs/grid_baseline_reference_v1.json").read_bytes()
    if _sha(payload) != REFERENCE_SHA256:
        raise ValueError("Trusted production baseline specification changed")
    reference = json.loads(payload)
    if (
        reference["source_sha256"] != BASELINE_SOURCE_SHA256
        or reference["prospective_protocol_sha256"] != PROTOCOL_SHA256
        or len(reference["file_sha256"]) != 87
    ):
        raise ValueError("Trusted production baseline identity differs")
    return reference


def _expected_config(mode):
    if mode not in ("production", "smoke"):
        raise ValueError("Baseline mode must be production or smoke")
    filename = "main_study_first_tranche.json" if mode == "production" else "main_study_smoke.json"
    config = MainStudyConfig.from_file(ROOT / "configs" / filename)
    if config.mode != mode:
        raise ValueError("Baseline configuration role differs")
    return config


def reference_binding(audit):
    """Return portable audit identity with large summary/protocol bodies excluded."""
    return {key: value for key, value in audit.items() if key not in ("summary", "protocol")}


def _validate_binding(binding, summary, protocol, mode):
    config = _expected_config(mode)
    expected_identity = {
        "schema_version": 1,
        "study": "main_study_first_tranche",
        "config": config.to_dict(),
        "source_sha256": BASELINE_SOURCE_SHA256,
    }
    expected_keys = {
        "schema_version",
        "mode",
        "source_sha256",
        "protocol_sha256",
        "prospective_protocol_sha256",
        "config",
        "planned_populations",
        "file_sha256",
    }
    if set(binding) != expected_keys:
        raise ValueError("Baseline reference binding fields differ")
    if (
        binding["schema_version"] != 1
        or binding["mode"] != mode
        or binding["source_sha256"] != BASELINE_SOURCE_SHA256
        or binding["prospective_protocol_sha256"] != PROTOCOL_SHA256
        or binding["config"] != config.to_dict()
        or binding["planned_populations"] != config.plan
    ):
        raise ValueError("Baseline reference configuration or source differs")
    if protocol != {**expected_identity, "design": config.design}:
        raise ValueError("Baseline protocol differs from its trusted configuration")
    if (
        summary.get("status") != "complete"
        or summary.get("config") != config.to_dict()
        or summary.get("planned_populations") != config.plan
        or summary.get("completed_populations") != config.plan
        or summary.get("pending_populations") != []
    ):
        raise ValueError("Baseline planned populations must all be complete")
    if set(summary.get("kinds", {})) != set(config.design["population"]["rare_kinds"]):
        raise ValueError("Baseline rare kinds differ")
    for kind, values in summary["kinds"].items():
        seeds = [item["seed"] for item in config.plan if item["kind"] == kind]
        if (
            values.get("planned_seeds") != seeds
            or values.get("completed_seeds") != seeds
            or [row.get("seed") for row in values.get("per_seed", [])] != seeds
            or set(values.get("groups", {})) != {"rare", "typical"}
        ):
            raise ValueError("Baseline grouped summary is incomplete")
    mapping = binding["file_sha256"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Baseline file inventory is missing")
    for name, digest in mapping.items():
        path = Path(name)
        if (
            not isinstance(name, str)
            or path.is_absolute()
            or ".." in path.parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Unsafe baseline file inventory")
    if mapping.get("protocol.json") != binding["protocol_sha256"]:
        raise ValueError("Baseline protocol hash differs from inventory")
    required = {"protocol.json", "summary.json", "run_manifest.json"}
    for item in config.plan:
        prefix = f"{item['kind']}/seed-{item['seed']}"
        required.update(f"{prefix}/{name}" for name in (*core.POPULATION_FILES, "manifest.json"))
    if not required.issubset(mapping):
        raise ValueError("Baseline file inventory omits planned artifacts")
    if mode == "production":
        trusted = _trusted_reference()
        if (
            mapping != trusted["file_sha256"]
            or binding["config"] != trusted["config"]
            or binding["planned_populations"] != trusted["planned_populations"]
        ):
            raise ValueError("Production baseline differs from the independently audited 87 files")


def validate_baseline(parent, mode="production"):
    """Verify a completed original cell, without invoking any fit or resume path.

    Production binds every file to the prior independently verified download.
    Smoke accepts freshly generated instances of the original tiny protocol,
    checks their complete manifests, and records their exact file inventory.
    """
    folder = _canonical(parent)
    if not folder.is_dir():
        raise ValueError("Baseline folder is missing")
    if core.main_source_fingerprint() != BASELINE_SOURCE_SHA256:
        raise ValueError("Original main-study numerical implementation changed")
    before = _file_hashes(folder)
    if mode == "production" and before != _trusted_reference()["file_sha256"]:
        raise ValueError("Production baseline differs from the independently audited 87 files")
    config = _expected_config(mode)
    summary, protocol = _json(folder / "summary.json"), _json(folder / "protocol.json")
    protocol_hash = before["protocol.json"]
    binding = {
        "schema_version": 1,
        "mode": mode,
        "source_sha256": BASELINE_SOURCE_SHA256,
        "protocol_sha256": protocol_hash,
        "prospective_protocol_sha256": PROTOCOL_SHA256,
        "config": config.to_dict(),
        "planned_populations": config.plan,
        "file_sha256": before,
    }
    _validate_binding(binding, summary, protocol, mode)
    manifest = _json(folder / "run_manifest.json")
    expected_identity = {key: protocol[key] for key in ("schema_version", "study", "config", "source_sha256")}
    if (
        manifest.get("identity") != expected_identity
        or manifest.get("status") != "complete"
        or manifest.get("immutable_file_sha256") != {"protocol.json": protocol_hash}
    ):
        raise ValueError("Baseline root manifest identity or completion differs")
    expected_root_files = {"protocol.json", "summary.json"}
    actual_dirs = {path.relative_to(folder).as_posix() for path in folder.glob("*/seed-*")}
    expected_dirs = {f"{item['kind']}/seed-{item['seed']}" for item in config.plan}
    if actual_dirs != expected_dirs:
        raise ValueError("Baseline population directories differ")
    for item in config.plan:
        child = core.population_path(folder, **item)
        report = core._validate_complete_population(child, item, protocol_hash)
        child_manifest = _json(child / "manifest.json")
        if child_manifest["sessions"][-1].get("status") != "complete":
            raise ValueError("Baseline population's final session is incomplete")
        core._checked_hashes(child, child_manifest["immutable_file_sha256"])
        if report.get("phases") != _json(child / "training.json"):
            raise ValueError("Baseline phase reports differ")
        for arm in core.ARMS:
            phase = report["phases"][arm]
            completed = phase.get("completed_steps")
            if isinstance(completed, bool) or not isinstance(completed, int):
                raise ValueError("Baseline optimization counter is invalid")
            if arm == "teacher":
                if not 0 <= completed <= config.teacher_steps or phase.get("stopped_reason") not in (
                    "early_stopping",
                    "steps_completed",
                ):
                    raise ValueError("Baseline teacher is incomplete")
            elif completed != config.student_steps or phase.get("stopped_reason") != "steps_completed":
                raise ValueError("Baseline student is incomplete")
            seconds = phase.get("elapsed_seconds")
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or seconds < 0
            ):
                raise ValueError("Baseline training time is invalid")
        prefix = child.relative_to(folder).as_posix()
        expected_root_files.update(f"{prefix}/{name}" for name in (*core.POPULATION_FILES, "manifest.json"))
        for session in child_manifest["sessions"]:
            expected_root_files.add(f"{prefix}/{session['package_inventory']['path']}")
        if (child / "environment-freeze.txt").exists():
            inventories = [session["package_inventory"]["sha256"] for session in child_manifest["sessions"]]
            if core._sha(child / "environment-freeze.txt") not in inventories:
                raise ValueError("Baseline convenience package inventory differs")
    if set(manifest.get("completed_file_sha256", {})) != expected_root_files:
        raise ValueError("Baseline root completion manifest omits or adds artifacts")
    core._checked_hashes(folder, manifest["completed_file_sha256"])
    if _file_hashes(folder) != before:
        raise ValueError("Baseline files changed during read-only verification")
    return {**binding, "summary": summary, "protocol": protocol}


def load_reference(destination, mode="production"):
    """Read the portable three-file reference without requiring its original path."""
    folder = _canonical(destination)
    if not folder.is_dir() or {path.name for path in folder.iterdir()} != REFERENCE_FILES:
        raise ValueError("Baseline reference must contain exactly its three complete files")
    binding = _json(folder / "verification.json")
    summary, protocol = _json(folder / "summary.json"), _json(folder / "protocol.json")
    _validate_binding(binding, summary, protocol, mode)
    mapping = binding["file_sha256"]
    if (
        core._sha(folder / "summary.json") != mapping["summary.json"]
        or core._sha(folder / "protocol.json") != mapping["protocol.json"]
    ):
        raise ValueError("Copied baseline summary or protocol differs from the verified input")
    return {**binding, "summary": summary, "protocol": protocol}


def _atomic_payload(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() and not payload.startswith(temporary.read_bytes()):
        raise ValueError("Unexpected temporary baseline reference payload")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def snapshot_baseline(parent, destination, mode="production"):
    """Copy only verified context, leaving every original baseline file untouched."""
    source, target = _canonical(parent), _canonical(destination)
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Baseline source and reference destination must not overlap")
    if any(
        target.is_relative_to(ROOT / name) or (ROOT / name).is_relative_to(target)
        for name in ("grid_study", "main_study", "foundation", "src", "configs")
    ):
        raise ValueError("Baseline reference destination overlaps source")
    audit = validate_baseline(source, mode)
    binding = reference_binding(audit)
    payloads = {name: (source / name).read_bytes() for name in ("summary.json", "protocol.json")}
    for name, payload in payloads.items():
        if _sha(payload) != binding["file_sha256"][name]:
            raise ValueError("Baseline context changed before copying")
    payloads["verification.json"] = (
        json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if target.exists():
        if (target / "verification.json").exists():
            loaded = load_reference(target, mode)
            if reference_binding(loaded) != binding:
                raise ValueError("Existing baseline reference belongs to different input")
            return loaded
        allowed = REFERENCE_FILES | {name + ".tmp" for name in REFERENCE_FILES}
        if not {path.name for path in target.iterdir()}.issubset(allowed):
            raise ValueError("Unexpected files in incomplete baseline reference")
        for name, payload in payloads.items():
            if (target / name).exists() and (target / name).read_bytes() != payload:
                raise ValueError("Existing incomplete baseline reference payload differs")
            temporary = target / (name + ".tmp")
            if temporary.exists() and not payload.startswith(temporary.read_bytes()):
                raise ValueError("Unexpected temporary baseline reference payload")
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        if not (target / name).exists():
            _atomic_payload(target / name, payload)
        (target / (name + ".tmp")).unlink(missing_ok=True)
    if _file_hashes(source) != binding["file_sha256"]:
        raise ValueError("Original baseline files changed while snapshotting context")
    return load_reference(target, mode)
