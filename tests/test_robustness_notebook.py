import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import types
from unittest.mock import patch

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ("01_colab_pilot.ipynb", "02_colab_followup.ipynb", "03_colab_baseline_audit.ipynb")


def code_cells():
    notebook = nbformat.read(ROOT / "notebooks/04_colab_robustness.ipynb", as_version=4)
    nbformat.validate(notebook)
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robustness_clean_pin_and_reviewed_bootstrap_guards():
    cells = code_cells()
    assert len(cells) == 5
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is None and cell.outputs == []
    settings = json.loads((ROOT / "configs/colab_robustness_source.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", settings["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", settings["source_sha256"])
    assert isinstance(settings["ready"], bool)
    prior = nbformat.read(ROOT / "notebooks/03_colab_baseline_audit.ipynb", as_version=4)
    prior_cells = [cell for cell in prior.cells if cell.cell_type == "code"]
    expected = prior_cells[0].source
    for variable, key in (("EXPECTED_SOURCE_SHA256", "source_sha256"),
                          ("GITHUB_REPOSITORY", "repository"), ("PINNED_COMMIT", "commit")):
        expected = re.sub(rf"^{variable} = .+$", lambda _: f"{variable} = {json.dumps(settings[key])}",
                          expected, flags=re.MULTILINE)
    expected = expected.replace("BASELINE_REVISION_READY", "ROBUSTNESS_REVISION_READY")
    expected = re.sub(r"^ROBUSTNESS_REVISION_READY = .+$",
                      f"ROBUSTNESS_REVISION_READY = {settings['ready']!r}", expected, flags=re.MULTILINE)
    expected = expected.replace("This baseline-audit launcher", "This robustness launcher")
    assert cells[0].source == expected
    assert cells[0].source.startswith((ROOT / "scripts/colab_bootstrap.py").read_text())
    assert cells[1].source == prior_cells[1].source
    joined = "\n".join(cell.source for cell in cells)
    assert joined.index('"pytest"') < joined.index("smoke = run_baseline_audit") < joined.index("report = run_baseline_audit")
    assert joined.index("if IN_COLAB and not ROBUSTNESS_REVISION_READY:") < joined.index("PROJECT = fetch_public_project(")


def test_builder_preserves_all_previous_notebooks_and_reproduces_fourth(tmp_path):
    for relative in (*(f"notebooks/{name}" for name in PREVIOUS), "configs/colab_robustness_source.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    originals = {name: (tmp_path / "notebooks" / name).read_bytes() for name in PREVIOUS}
    builder = load_script("build_robustness_colab")
    generated = builder.build(tmp_path)
    expected = (ROOT / "notebooks/04_colab_robustness.ipynb").read_bytes()
    assert generated.read_bytes() == expected
    assert builder.build(tmp_path).read_bytes() == expected
    assert all((tmp_path / "notebooks" / name).read_bytes() == contents for name, contents in originals.items())


@pytest.mark.parametrize("field,value", [("commit", "main"), ("source_sha256", "bad"), ("ready", "true")])
def test_builder_rejects_unpinned_or_ambiguous_release_settings(tmp_path, field, value):
    settings = json.loads((ROOT / "configs/colab_robustness_source.json").read_text())
    settings[field] = value
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/colab_robustness_source.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        load_script("build_robustness_colab").build(tmp_path)


def test_unreleased_colab_launcher_stops_before_source_download():
    # Exercise the release gate in the actual bootstrap, even after a release.
    source = re.sub(r"^ROBUSTNESS_REVISION_READY = .+$", "ROBUSTNESS_REVISION_READY = False",
                    code_cells()[0].source, flags=re.MULTILINE)
    calls = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: calls.append(path))
    google = types.ModuleType("google")
    google.colab = colab
    with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
        with patch("urllib.request.build_opener") as opener:
            with pytest.raises(RuntimeError, match="awaiting its verified runtime release"):
                exec(compile(source, "<robustness bootstrap>", "exec"), {})
        opener.assert_not_called()
    assert calls == []


@pytest.mark.parametrize("in_colab,override,expected", [
    (True, "custom_check", "robustness"), (False, None, "robustness_smoke"),
    (False, "custom_check", "custom_check"),
])
def test_actual_study_cell_routes_config_and_incomplete_resume(tmp_path, capsys, in_colab, override, expected):
    calls = []
    config = types.SimpleNamespace(to_dict=lambda: {"test_config": expected})

    def from_file(path):
        calls.append(path)
        return config

    def run(config_arg, path, device):
        assert config_arg is config and device == "cpu"
        calls.append(path)
        return {"status": "incomplete", "message": "Time budget reached."}

    namespace = {"IN_COLAB": in_colab, "os": os, "PROJECT": tmp_path,
                 "RUNS": tmp_path / "runs", "DEVICE": "cpu", "json": json,
                 "BaselineAuditConfig": types.SimpleNamespace(from_file=from_file), "run_baseline_audit": run}
    environment = {} if override is None else {"DISTILLED_ROBUSTNESS_CONFIG": override}
    with patch.dict(os.environ, environment, clear=True):
        exec(compile(code_cells()[3].source, "<robustness study>", "exec"), namespace)
    assert calls == [tmp_path / "configs" / f"{expected}.json", tmp_path / "runs/robustness-v1"]
    # An incomplete result has no aggregate fields: the actual display cell must not read them.
    exec(compile(code_cells()[4].source, "<robustness incomplete results>", "exec"), namespace)
    output = capsys.readouterr().out
    assert "ONLY the fourth code cell with the same RUN_NAME" in output
    assert "Do NOT use Run all or restart the runtime" in output
    assert "No aggregation until all planned seeds complete" in output


@pytest.mark.parametrize("diagnostic_seed_count", [0, 1, 2])
def test_actual_report_renders_diagnostic_kl_pairs_support_and_teacher(tmp_path, capsys, diagnostic_seed_count):
    supported = dict(mean=.67, min=.66, max=.68, supported_seed_count=2, planned_seed_count=2)
    unavailable = dict(mean=None, min=None, max=None, supported_seed_count=0, planned_seed_count=2)
    diagnostic = (dict(supported, supported_seed_count=diagnostic_seed_count) if diagnostic_seed_count else unavailable)
    diagnostic_difference = (dict(mean=-.125, min=-.125, max=-.125,
                                  supported_seed_count=diagnostic_seed_count, planned_seed_count=2)
                             if diagnostic_seed_count else unavailable)
    names = ("teacher", "solo", "distilled", "coinflip", "frequency", "solo_legacy",
             "solo_individual", "solo_population", "distilled_individual", "distilled_population")
    comparisons = {"likelihood_gain": supported, "fidelity_loss": supported,
                   "diagnostic_likelihood_gain": diagnostic, "diagnostic_fidelity_loss": diagnostic_difference}
    group = {
        "models": {name: {metric: (diagnostic if metric.startswith("diagnostic") else supported)
                           for metric in ("nll", "kl", "diagnostic_nll", "diagnostic_kl")} for name in names},
        "comparisons": {name: comparisons for name in ("solo", "solo_legacy", "coinflip", "frequency")},
        "matched_comparisons": {name: comparisons for name in ("individual", "population")},
        "per_seed": [dict(seed=seed, diagnostic_count=4 if index < diagnostic_seed_count else 0,
                          diagnostic_subject_count=2 if index < diagnostic_seed_count else 0,
                          models={name: {"diagnostic_kl": value if index < diagnostic_seed_count else None}
                                  for name, value in (("teacher", .75), ("solo", .625), ("distilled", .5))},
                          comparisons={"solo": {"diagnostic_fidelity_loss": -.125 if index < diagnostic_seed_count else None}})
                     for index, seed in enumerate((71, 81))],
    }
    report = {
        "status": "complete", "output_dir": str(tmp_path),
        "calibration": {"solo_l2": .01, "solo_selection": "population", "distilled_selection": "individual"},
        "convergence_warnings": [], "groups": {"rare": group, "typical": group},
        "timing": {"cumulative_training_seconds": 3, "t4_equivalent_units_at_user_rate": None},
    }
    for folder in ("calibration", "seed-71", "seed-81"):
        (tmp_path / folder).mkdir()
        for filename in ("selection_audit.png", "convergence.png"):
            (tmp_path / folder / filename).write_bytes(b"placeholder for mocked image display")
    rendered = []
    display_module = types.SimpleNamespace(display=rendered.append, Markdown=lambda value: value,
                                           Image=lambda **kwargs: kwargs)
    namespace = {"report": report, "Path": Path,
                 "config": types.SimpleNamespace(base=types.SimpleNamespace(student_steps=4000))}
    with patch.dict(sys.modules, {"IPython.display": display_module}):
        exec(compile(code_cells()[4].source, "<robustness results>", "exec"), namespace)
    text = "\n".join(value for value in rendered if isinstance(value, str))
    assert text.count("| Diagnostic KL loss |") == 4  # Two comparison tables in each group.
    assert "Fixed solo L2=0.01, individual (4,000 steps)" in text
    assert "Previous solo recipe" not in text
    assert "Fixed 50/50" in text and "Training choice frequency" in text
    assert "| Seed | Teacher | Calibrated solo | Calibrated distilled | Distilled minus solo |" in text
    if diagnostic_seed_count:
        assert text.count(f"-0.1250 [-0.1250, -0.1250] ({diagnostic_seed_count}/2 seeds)") == 14
        assert "| 71 | 0.7500 | 0.6250 | 0.5000 | -0.1250 |" in text
        assert "71: 4 trials / 2 people" in text
    else:
        assert "unavailable (0/2 seeds)" in text
        assert "| 71 | unavailable | unavailable | unavailable | unavailable |" in text
        assert "71: 0 trials / 0 people" in text
    assert f"Direction counts: 0 higher, {diagnostic_seed_count} lower, 0 tied; {diagnostic_seed_count}/2 seeds supported." in text
    if diagnostic_seed_count == 1:
        assert "| 81 | unavailable | unavailable | unavailable | unavailable |" in text
    images = [value["filename"] for value in rendered if isinstance(value, dict)]
    assert images == [str(tmp_path / "calibration/selection_audit.png"),
                      str(tmp_path / "seed-71/selection_audit.png"),
                      str(tmp_path / "seed-71/convergence.png")]
    output = capsys.readouterr().out
    assert "Additional selection-audit figure: " + str(tmp_path / "seed-81/selection_audit.png") in output
    assert "T4 compute-unit proxy unavailable" in output


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="Local Git archive verification needs a checkout; Colab downloads have no .git")
def test_local_pinned_archive_matches_frozen_numerical_source(tmp_path):
    verifier = load_script("verify_robustness_notebook")
    project, settings = verifier.prepare_archived_project(tmp_path / "project")
    assert settings == json.loads((ROOT / "configs/colab_robustness_source.json").read_text())
    # Notebook 04 reuses the reviewed implementation rather than modifying the metrics.
    assert (project / "src/distilled_rnns/baseline_audit.py").read_bytes() == (ROOT / "src/distilled_rnns/baseline_audit.py").read_bytes()
