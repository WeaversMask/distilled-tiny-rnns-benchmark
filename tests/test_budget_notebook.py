import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import types
from unittest.mock import patch

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ("01_colab_pilot.ipynb", "02_colab_followup.ipynb", "03_colab_baseline_audit.ipynb", "04_colab_robustness.ipynb", "05_colab_oracle_control.ipynb")
METRICS = ("nll", "kl", "diagnostic_nll", "diagnostic_kl")
MODELS = ("teacher", "solo", "distilled", "oracle", "solo_population", "solo_individual",
          "distilled_population", "distilled_individual", "oracle_population", "oracle_individual", "coinflip", "frequency")


def code_cells():
    notebook = nbformat.read(ROOT / "notebooks/06_colab_budget_check.ipynb", as_version=4)
    nbformat.validate(notebook)
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_clean_pins_and_reviewed_bootstrap_setup_guards():
    cells = code_cells()
    assert len(cells) == 5
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is None and cell.outputs == []
    settings = json.loads((ROOT / "configs/colab_budget_source.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", settings["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", settings["source_sha256"])
    assert isinstance(settings["ready"], bool)
    prior = nbformat.read(ROOT / "notebooks/05_colab_oracle_control.ipynb", as_version=4)
    prior_cells = [cell for cell in prior.cells if cell.cell_type == "code"]
    expected = prior_cells[0].source
    for variable, key in (("EXPECTED_SOURCE_SHA256", "source_sha256"), ("GITHUB_REPOSITORY", "repository"),
                          ("PINNED_COMMIT", "commit")):
        expected = re.sub(rf"^{variable} = .+$", lambda _: f"{variable} = {json.dumps(settings[key])}", expected, flags=re.MULTILINE)
    expected = expected.replace("ORACLE_REVISION_READY", "BUDGET_REVISION_READY")
    expected = re.sub(r"^BUDGET_REVISION_READY = .+$", f"BUDGET_REVISION_READY = {settings['ready']!r}", expected, flags=re.MULTILINE)
    expected = expected.replace("This oracle-control launcher", "This budget-check launcher")
    assert cells[0].source == expected
    assert cells[0].source.startswith((ROOT / "scripts/colab_bootstrap.py").read_text())
    assert cells[1].source == prior_cells[1].source
    assert 'stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True' in cells[1].source
    assert cells[1].source.index('print(verification.stdout') < cells[1].source.index('verification.check_returncode()')
    joined = "\n".join(cell.source for cell in cells)
    assert joined.index('"pytest"') < joined.index('parent_smoke = run_oracle_control') < joined.index('smoke = run_budget_check') < joined.index('report = run_budget_check')
    assert joined.index('if IN_COLAB and not BUDGET_REVISION_READY:') < joined.index('PROJECT = fetch_public_project(')
    assert joined.count('expected_parent_source_sha256=EXPECTED_SOURCE_SHA256') == 1


def test_builder_reproducible_and_preserves_all_previous_notebooks(tmp_path):
    for relative in (*(f"notebooks/{name}" for name in PREVIOUS), "configs/colab_budget_source.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    originals = {name: (tmp_path / "notebooks" / name).read_bytes() for name in PREVIOUS}
    builder = load_script("build_budget_colab")
    expected = (ROOT / "notebooks/06_colab_budget_check.ipynb").read_bytes()
    assert builder.build(tmp_path).read_bytes() == expected
    assert builder.build(tmp_path).read_bytes() == expected
    assert all((tmp_path / "notebooks" / name).read_bytes() == contents for name, contents in originals.items())


@pytest.mark.parametrize("field,value", [("commit", "main"), ("commit", None), ("source_sha256", "bad"), ("ready", "true")])
def test_builder_rejects_unpinned_or_ambiguous_settings(tmp_path, field, value):
    settings = json.loads((ROOT / "configs/colab_budget_source.json").read_text())
    settings[field] = value
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/colab_budget_source.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        load_script("build_budget_colab").build(tmp_path)


def test_unreleased_colab_stops_before_source_download():
    source = re.sub(r"^BUDGET_REVISION_READY = .+$", "BUDGET_REVISION_READY = False", code_cells()[0].source, flags=re.MULTILINE)
    calls = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: calls.append(path))
    google = types.ModuleType("google")
    google.colab = colab
    with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
        with patch("urllib.request.build_opener") as opener:
            with pytest.raises(RuntimeError, match="awaiting its verified runtime release"):
                exec(compile(source, "<budget bootstrap>", "exec"), {})
        opener.assert_not_called()
    assert calls == []


def test_actual_bootstrap_rejects_changed_package_before_training(tmp_path):
    project = tmp_path / "project"
    package = project / "src/distilled_rnns"
    package.mkdir(parents=True)
    (project / "pyproject.toml").write_text("")
    (package / "__init__.py").write_text("# changed package\n")
    environment = dict(os.environ, DISTILLED_PROJECT_ROOT=str(project), DISTILLED_OUTPUT_ROOT=str(tmp_path / "runs"))
    # The local fixture must not detect an installed google.colab package.
    # This bootstrap uses only the standard library; -S excludes site packages.
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code_cells()[0].source],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Downloaded source differs from the verified notebook revision" in result.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("stale,cuda,test_failure", [(False, True, False), (True, True, False),
                                                    (False, False, False), (False, True, True)])
def test_actual_setup_kernel_activation_stale_guard_and_gpu_preflight(tmp_path, stale, cuda, test_failure):
    setup_file = tmp_path / "setup.py"
    setup_file.write_text(code_cells()[1].source)
    project = tmp_path / "downloaded-project"
    package = project / "src/distilled_rnns"
    package.mkdir(parents=True)
    for filename in ("__init__.py", "config.py"):
        shutil.copyfile(ROOT / "src/distilled_rnns" / filename, package / filename)
    script = r'''
from pathlib import Path
import importlib.util, subprocess as real_subprocess, sys, types
PROJECT = Path(sys.argv[1])
setup = Path(sys.argv[2]).read_text()
stale, cuda = sys.argv[3] == 'True', sys.argv[4] == 'True'
test_failure = sys.argv[5] == 'True'
calls = []
def run(command, **kwargs):
    calls.append(command)
    if 'pip' in command:
        assert importlib.util.find_spec('distilled_rnns') is None
    if 'pytest' in command:
        assert kwargs['stdout'] == real_subprocess.PIPE
        assert kwargs['stderr'] == real_subprocess.STDOUT and kwargs['text'] is True
        return real_subprocess.CompletedProcess(command, int(test_failure),
            stdout='FAILED example: useful verification details' if test_failure else 'verification passed')
    return real_subprocess.CompletedProcess(command, 0)
subprocess = types.SimpleNamespace(run=run, PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT)
sys.modules['torch'] = types.SimpleNamespace(__version__='test', cuda=types.SimpleNamespace(
    is_available=lambda: cuda, get_device_name=lambda index: 'test GPU'))
if stale:
    sys.modules['distilled_rnns.config'] = types.SimpleNamespace(__file__=str(PROJECT.parent / 'old/src/distilled_rnns/config.py'))
IN_COLAB = True
try:
    exec(compile(setup, '<budget setup>', 'exec'))
except RuntimeError as error:
    if stale:
        assert 'Restart' in str(error) and calls == []
    else:
        assert not cuda and 'Select a GPU runtime' in str(error)
        assert len(calls) == 1 and 'pip' in calls[0]
except real_subprocess.CalledProcessError:
    assert test_failure and not stale and cuda
    assert len(calls) == 2
    print('Failed verification blocks continuation')
else:
    assert not stale and cuda and not test_failure
    import distilled_rnns.config as config_module
    assert Path(config_module.__file__).resolve() == (PROJECT / 'src/distilled_rnns/config.py').resolve()
    assert len(calls) == 2 and 'pip' in calls[0] and 'pytest' in calls[1]
print('Kernel guard and GPU preflight passed')
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(project), str(setup_file), str(stale), str(cuda), str(test_failure)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Kernel guard and GPU preflight passed" in result.stdout
    if test_failure:
        assert result.stdout.index('FAILED example: useful verification details') < result.stdout.index('Failed verification blocks continuation')


@pytest.mark.parametrize("in_colab", [True, False])
def test_study_fixed_config_routing_and_incomplete_report(tmp_path, capsys, in_colab):
    calls = []
    config = types.SimpleNamespace(to_dict=lambda: {"test_config": "production"})
    smoke_config = types.SimpleNamespace(to_dict=lambda: {"test_config": "smoke"})
    def from_file(path):
        calls.append(path)
        return config
    def run(config_arg, parent, child, device):
        assert config_arg is (config if in_colab else smoke_config) and device == "cpu"
        calls.append((parent, child))
        return {"status": "incomplete", "message": "Time budget reached."}
    namespace = {"IN_COLAB": in_colab, "os": os, "PROJECT": tmp_path, "RUNS": tmp_path / "runs", "DEVICE": "cpu", "json": json,
                 "BudgetCheckConfig": types.SimpleNamespace(from_file=from_file), "run_budget_check": run,
                 "smoke_config": smoke_config, "SMOKE_PARENT": tmp_path / "runs/smoke-parent"}
    with patch.dict(os.environ, {"DISTILLED_BUDGET_CONFIG": "untrusted", "DISTILLED_ORACLE_CONFIG": "untrusted"}, clear=True):
        exec(compile(code_cells()[3].source, "<budget study>", "exec"), namespace)
    expected = [tmp_path / "configs/budget_check.json"] if in_colab else []
    expected.append((tmp_path / "runs" / ("oracle-control-v1" if in_colab else "smoke-parent"), tmp_path / "runs/budget-check-v1"))
    assert calls == expected
    exec(compile(code_cells()[4].source, "<budget incomplete report>", "exec"), namespace)
    output = capsys.readouterr().out
    assert "ONLY the fourth code cell with the same RUN_NAME" in output
    assert "Do NOT use Run all or restart the runtime" in output
    assert "No aggregation until all planned seeds complete" in output


def test_actual_smoke_cell_reuses_immutable_parent_after_rerun_and_kernel_loss(tmp_path, monkeypatch):
    """A reconnect must not rerun the parent and invalidate the child's identity."""
    from distilled_rnns import oracle_control
    from distilled_rnns.experiment import source_fingerprint

    root = tmp_path.resolve()
    source = code_cells()[2].source
    parent_runner = oracle_control.run_oracle_control
    calls = []

    def tracked_parent_runner(*args, **kwargs):
        calls.append(args[1])
        return parent_runner(*args, **kwargs)

    monkeypatch.setattr(oracle_control, "run_oracle_control", tracked_parent_runner)

    def fresh_namespace():
        return {"RUNS": root / "restored-drive/runs", "PROJECT": ROOT,
                "EXPECTED_SOURCE_SHA256": source_fingerprint(), "DEVICE": "cpu",
                "hashlib": hashlib, "json": json}

    namespace = fresh_namespace()
    exec(compile(source, "<actual budget smoke>", "exec"), namespace)
    assert len(calls) == 1
    parent_hashes = namespace["_file_fingerprints"](namespace["SMOKE_PARENT"])
    predictions = namespace["_file_fingerprints"](namespace["SMOKE_RUN"], "*predictions.npz")
    timing = deepcopy(namespace["smoke"]["timing"])
    results = deepcopy(namespace["smoke"]["groups"])
    # First repeat in the live namespace; then discard all globals as Colab does
    # when a deleted runtime reconnects to the same persistent Drive folders.
    for repeated in (namespace, fresh_namespace()):
        exec(compile(source, "<reconnected budget smoke>", "exec"), repeated)
        assert len(calls) == 1
        assert repeated["_file_fingerprints"](repeated["SMOKE_PARENT"]) == parent_hashes
        assert repeated["_file_fingerprints"](repeated["SMOKE_RUN"], "*predictions.npz") == predictions
        assert repeated["smoke"]["timing"] == timing
        assert repeated["smoke"]["groups"] == results

    # An existing child must never authorize rebuilding a lost/incomplete parent.
    (namespace["SMOKE_PARENT"] / "summary.json").unlink()
    with pytest.raises(RuntimeError, match="child already exists but its parent is incomplete"):
        exec(compile(source, "<missing restored parent>", "exec"), fresh_namespace())
    assert len(calls) == 1


def test_actual_smoke_cell_routes_gpu_into_completed_parent_validation(tmp_path, monkeypatch):
    """A reconnected GPU notebook must not silently audit its parent on CPU."""
    from distilled_rnns import budget_check, oracle_control
    from distilled_rnns.experiment import source_fingerprint

    root = tmp_path.resolve()
    fingerprint = source_fingerprint()
    parent = root / ("budget-check-parent-smoke-" + fingerprint[:12])
    parent.mkdir()
    (parent / "summary.json").write_text(json.dumps({"status": "complete"}))
    calls = []

    class AuditReached(RuntimeError):
        pass

    def audit(config, folder, *, device):
        assert folder == parent and config.expected_parent_source_sha256 == fingerprint
        calls.append(device)
        raise AuditReached("GPU audit reached without training")

    monkeypatch.setattr(budget_check, "validate_parent", audit)
    monkeypatch.setattr(oracle_control, "run_oracle_control",
                        lambda *args, **kwargs: pytest.fail("completed parent was retrained"))
    monkeypatch.setattr(budget_check, "run_budget_check",
                        lambda *args, **kwargs: pytest.fail("child ran before parent audit"))
    namespace = {"RUNS": root, "PROJECT": ROOT, "EXPECTED_SOURCE_SHA256": fingerprint,
                 "DEVICE": "cuda", "hashlib": hashlib, "json": json}
    with pytest.raises(AuditReached):
        exec(compile(code_cells()[2].source, "<GPU smoke parent validation>", "exec"), namespace)
    assert calls == ["cuda"]


def _fixture_report(supported_count=3, units=None):
    def aggregate(values, directions=False):
        available = [value for value in values if value is not None]
        result = dict(mean=sum(available) / len(available) if available else None,
                      min=min(available) if available else None, max=max(available) if available else None,
                      supported_seed_count=len(available), planned_seed_count=3)
        if directions:
            result.update(positive_seed_count=sum(value > 0 for value in available),
                          negative_seed_count=sum(value < 0 for value in available),
                          tied_seed_count=sum(value == 0 for value in available))
        return result
    rows = []
    for index, seed in enumerate((71, 81, 91)):
        supported = index < supported_count
        baseline = {name: {metric: None if metric.startswith("diagnostic") and not supported else .625
                           for metric in METRICS} for name in MODELS}
        extended = deepcopy(baseline)
        comparisons = {stage: {mode: {} for mode in ("population", "individual")} for stage in ("baseline", "extended")}
        changes = {mode: {} for mode in ("population", "individual")}
        arm_changes = {mode: {arm: {} for arm in ("distilled", "oracle")} for mode in ("population", "individual")}
        for mode in ("population", "individual"):
            change = [.125, -.25, 0][index] * (1 if mode == "population" else -1)
            for metric in METRICS:
                available = not metric.startswith("diagnostic") or supported
                for arm, old, new in (("distilled", .5, .45), ("oracle", .375, .325 - change)):
                    baseline[f"{arm}_{mode}"][metric] = old if available else None
                    extended[f"{arm}_{mode}"][metric] = new if available else None
                    arm_changes[mode][arm][metric] = new - old if available else None
                for stage, models in (("baseline", baseline), ("extended", extended)):
                    comparisons[stage][mode][metric + "_gain"] = models[f"distilled_{mode}"][metric] - models[f"oracle_{mode}"][metric] if available else None
                changes[mode][metric + "_gain"] = change if available else None
        rows.append(dict(seed=seed, subject_count=2, test_count=8,
                         diagnostic_count=4 if supported else 0, diagnostic_subject_count=2 if supported else 0,
                         baseline_models=baseline, extended_models=extended,
                         baseline_target_comparisons=comparisons["baseline"], extended_target_comparisons=comparisons["extended"],
                         paired_gain_change=changes, arm_changes=arm_changes))
    group = {"per_seed": rows}
    for section in ("baseline_models", "extended_models"):
        group[section] = {name: {metric: aggregate([row[section][name][metric] for row in rows]) for metric in METRICS} for name in MODELS}
    for section in ("baseline_target_comparisons", "extended_target_comparisons", "paired_gain_change"):
        group[section] = {mode: {metric + "_gain": aggregate([row[section][mode][metric + "_gain"] for row in rows], True) for metric in METRICS}
                          for mode in ("population", "individual")}
    group["arm_changes"] = {mode: {arm: {metric: aggregate([row["arm_changes"][mode][arm][metric] for row in rows], True)
                                          for metric in METRICS} for arm in ("distilled", "oracle")}
                            for mode in ("population", "individual")}
    group["support"] = {key: aggregate([row[key] for row in rows])
                        for key in ("subject_count", "test_count", "diagnostic_count", "diagnostic_subject_count")}
    return {"status": "complete", "output_dir": "/example/budget-check-v1", "groups": {"rare": group, "typical": deepcopy(group)},
            "planned_seeds": [71, 81, 91], "completed_seeds": [71, 81, 91], "pending_seeds": [],
            "selection": {"solo_l2": .1, "primary_mode": "population", "secondary_mode": "individual"},
            "convergence_warnings": ["Inspect the final checkpoint."],
            "timing": {"incremental_training_seconds": 3, "parent_training_seconds": 60, "t4_equivalent_units_at_user_rate": units}}


@pytest.mark.parametrize("supported_count", [0, 1, 3])
@pytest.mark.parametrize("units", [None, 0, .25])
def test_report_baseline_extended_signed_changes_null_zero_support_and_incremental_compute(capsys, supported_count, units):
    report = _fixture_report(supported_count, units)
    config = types.SimpleNamespace(parent_config=types.SimpleNamespace(base=types.SimpleNamespace(student_steps=4000)), student_steps=8000)
    rendered = []
    display_module = types.SimpleNamespace(display=rendered.append, Markdown=lambda value: value)
    with patch.dict(sys.modules, {"IPython.display": display_module}):
        exec(compile(code_cells()[4].source, "<budget report>", "exec"), {"report": report, "config": config})
    text = "\n".join(rendered)
    assert "Primary outcome: rare-group diagnostic ΔKL with population checkpoints" in text
    assert "positive favors oracle" in text
    assert "population (primary)" in text and "individual (secondary)" in text
    assert "Rare group: original versus extended" in text and "Typical group: original versus extended" in text
    assert "Baseline Δ (4,000) | Extended Δ (8,000) | Paired gain change" in text
    assert "negative change means improvement" in text
    assert "Fixed teacher" in text and "Fixed solo L2=0.1" in text
    assert "Teacher-target error change | Oracle-target error change" in text
    assert f"Teacher-target diagnostic KL error change (population): {supported_count} improved, 0 worsened, 0 tied; {supported_count}/3 seeds supported." in text
    positive, negative, tied = (1, 1, 1) if supported_count == 3 else (supported_count, 0, 0)
    assert f"Paired diagnostic gain change (population): {positive} positive, {negative} negative, {tied} tied; {supported_count}/3 seeds supported." in text
    assert f"Paired diagnostic gain change (individual): {negative} positive, {positive} negative, {tied} tied; {supported_count}/3 seeds supported." in text
    if supported_count:
        assert "| 71 | 4 / 2 | population | +0.1250 | +0.2500 | +0.1250 |" in text
    if supported_count < 3:
        assert "| 91 | 0 / 0 | population | unavailable | unavailable | unavailable |" in text
    if supported_count == 0:
        assert "unavailable (0/3 seeds)" in text
    if supported_count == 3:
        assert "| 91 | 4 / 2 | population | +0.1250 | +0.1250 | +0.0000 |" in text
        assert "-0.0417 [-0.2500, +0.1250] (3/3 seeds)" in text
        assert "+0.0417 [-0.1250, +0.2500] (3/3 seeds)" in text
    output = capsys.readouterr().out
    assert "CHECK REQUIRED: Inspect the final checkpoint." in output
    assert "Incremental extension training: 0.05 minutes" in output
    assert "Original parent training (not new work): 1.00 minutes" in output
    assert "Setup, evaluation, saving and idle billing are additional" in output
    if units is None:
        assert "no proxy is inferred from mixed or unrecognized devices" in output
    else:
        assert f"Incremental training-only T4 compute-unit proxy: {units:.4f}" in output


@pytest.mark.parametrize("supported_count", [0, 1, 3])
def test_verifier_accepts_supported_null_zero_and_paired_changes(supported_count):
    load_script("verify_budget_notebook").verify_summary(_fixture_report(supported_count))


@pytest.mark.parametrize("corruption,message", [
    ("seed", "omitted a planned seed"), ("gain", "Wrong direction"), ("change", "Wrong direction"),
    ("arm", "Wrong direction"), ("null", "Unavailable"), ("reference", "fixed reference changed"),
    ("support", "Aggregate support"), ("mean", "Aggregate value"), ("timing", "Invalid training time"),
    ("directions", "Aggregate directions"), ("arm_aggregate", "Aggregate value"),
    ("model_aggregate", "Aggregate value"), ("diagnostic_support", "Aggregate value"),
])
def test_verifier_rejects_misleading_results(corruption, message):
    report = _fixture_report(1)
    group = report["groups"]["rare"]
    row = group["per_seed"][0]
    if corruption == "seed":
        report["planned_seeds"].append(101)
    elif corruption == "gain":
        row["baseline_target_comparisons"]["population"]["diagnostic_kl_gain"] *= -1
    elif corruption == "change":
        row["paired_gain_change"]["population"]["diagnostic_kl_gain"] *= -1
    elif corruption == "arm":
        row["arm_changes"]["population"]["oracle"]["diagnostic_kl"] *= -1
    elif corruption == "null":
        group["per_seed"][2]["paired_gain_change"]["population"]["diagnostic_kl_gain"] = 0
    elif corruption == "reference":
        row["extended_models"]["teacher"]["kl"] += .1
    elif corruption == "support":
        group["paired_gain_change"]["population"]["diagnostic_kl_gain"]["supported_seed_count"] = 3
    elif corruption == "mean":
        group["paired_gain_change"]["population"]["diagnostic_kl_gain"]["mean"] = 100
    elif corruption == "timing":
        report["timing"]["incremental_training_seconds"] = -1
    elif corruption == "directions":
        group["paired_gain_change"]["population"]["diagnostic_kl_gain"]["positive_seed_count"] = 2
    elif corruption == "arm_aggregate":
        group["arm_changes"]["population"]["oracle"]["diagnostic_kl"]["mean"] *= -1
    elif corruption == "model_aggregate":
        group["extended_models"]["oracle_population"]["diagnostic_kl"]["mean"] += .1
    elif corruption == "diagnostic_support":
        group["support"]["diagnostic_count"]["mean"] = 400
    with pytest.raises(AssertionError, match=message):
        load_script("verify_budget_notebook").verify_summary(report)


def test_notebook_fixed_extension_documentation_and_real_resume_checks():
    notebook = nbformat.read(ROOT / "notebooks/06_colab_budget_check.ipynb", as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    for phrase in ("4,000 to 8,000", "read-only parent", "more checkpoint candidates", "does not prove convergence",
                   "Do not repeatedly extend the horizon", "same parent and child folders", "completed child is final"):
        assert phrase in text
    smoke = code_cells()[2].source
    assert 'parent_smoke = run_oracle_control' in smoke
    assert 'assert parent_smoke["status"] == "complete"' in smoke
    assert 'assert smoke_resumed["timing"] == smoke["timing"]' in smoke
    assert 'assert _file_fingerprints(SMOKE_PARENT) == parent_before' in smoke
    assert 'assert _file_fingerprints(SMOKE_RUN, "*predictions.npz") == predictions_before' in smoke
    assert 'config = BudgetCheckConfig.from_file(PROJECT / "configs/budget_check.json")' in code_cells()[3].source


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="Pinned archive checks require a Git checkout")
def test_oracle_verifier_defaults_to_frozen_archive_after_budget_module_added(tmp_path):
    verifier = load_script("verify_oracle_notebook")
    project, settings = verifier.prepare_archived_project(tmp_path / "oracle")
    assert settings == json.loads((ROOT / "configs/colab_oracle_source.json").read_text())
    assert not (project / "src/distilled_rnns/budget_check.py").exists()
    assert 'action="store_true", default=True' in (ROOT / "scripts/verify_oracle_notebook.py").read_text()
