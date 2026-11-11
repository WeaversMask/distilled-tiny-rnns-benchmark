import ast
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
PREVIOUS = ("01_colab_pilot.ipynb", "02_colab_followup.ipynb", "03_colab_baseline_audit.ipynb", "04_colab_robustness.ipynb")
METRICS = ("nll", "kl", "diagnostic_nll", "diagnostic_kl")
MODELS = ("teacher", "solo", "distilled", "oracle", "solo_population", "solo_individual",
          "distilled_population", "distilled_individual", "oracle_population", "oracle_individual", "coinflip", "frequency")


def code_cells():
    notebook = nbformat.read(ROOT / "notebooks/05_colab_oracle_control.ipynb", as_version=4)
    nbformat.validate(notebook)
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oracle_clean_pins_and_reviewed_bootstrap_setup_guards():
    cells = code_cells()
    assert len(cells) == 5
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is None and cell.outputs == []
    settings = json.loads((ROOT / "configs/colab_oracle_source.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", settings["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", settings["source_sha256"])
    assert isinstance(settings["ready"], bool)
    prior = nbformat.read(ROOT / "notebooks/04_colab_robustness.ipynb", as_version=4)
    prior_cells = [cell for cell in prior.cells if cell.cell_type == "code"]
    expected = prior_cells[0].source
    for variable, key in (("EXPECTED_SOURCE_SHA256", "source_sha256"), ("GITHUB_REPOSITORY", "repository"),
                          ("PINNED_COMMIT", "commit")):
        expected = re.sub(rf"^{variable} = .+$", lambda _: f"{variable} = {json.dumps(settings[key])}",
                          expected, flags=re.MULTILINE)
    expected = expected.replace("ROBUSTNESS_REVISION_READY", "ORACLE_REVISION_READY")
    expected = re.sub(r"^ORACLE_REVISION_READY = .+$", f"ORACLE_REVISION_READY = {settings['ready']!r}", expected, flags=re.MULTILINE)
    expected = expected.replace("This robustness launcher", "This oracle-control launcher")
    assert cells[0].source == expected
    assert cells[0].source.startswith((ROOT / "scripts/colab_bootstrap.py").read_text())
    old_preflight = 'subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)'
    assert cells[1].source.split("verification = subprocess.run(")[0] == prior_cells[1].source.split(old_preflight)[0]
    assert "verification.check_returncode()" in cells[1].source
    joined = "\n".join(cell.source for cell in cells)
    assert joined.index('"pytest"') < joined.index("smoke = run_oracle_control") < joined.index("report = run_oracle_control")
    assert joined.index("if IN_COLAB and not ORACLE_REVISION_READY:") < joined.index("PROJECT = fetch_public_project(")


def test_builder_reproducible_and_preserves_all_previous_notebooks(tmp_path):
    for relative in (*(f"notebooks/{name}" for name in PREVIOUS), "configs/colab_oracle_source.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    originals = {name: (tmp_path / "notebooks" / name).read_bytes() for name in PREVIOUS}
    builder = load_script("build_oracle_colab")
    expected = (ROOT / "notebooks/05_colab_oracle_control.ipynb").read_bytes()
    assert builder.build(tmp_path).read_bytes() == expected
    assert builder.build(tmp_path).read_bytes() == expected
    assert all((tmp_path / "notebooks" / name).read_bytes() == contents for name, contents in originals.items())


@pytest.mark.parametrize("field,value", [("commit", "main"), ("commit", None), ("source_sha256", "bad"), ("ready", "true")])
def test_builder_rejects_unpinned_or_ambiguous_settings(tmp_path, field, value):
    settings = json.loads((ROOT / "configs/colab_oracle_source.json").read_text())
    settings[field] = value
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/colab_oracle_source.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        load_script("build_oracle_colab").build(tmp_path)


def test_unreleased_colab_stops_before_source_download():
    source = re.sub(r"^ORACLE_REVISION_READY = .+$", "ORACLE_REVISION_READY = False", code_cells()[0].source, flags=re.MULTILINE)
    calls = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: calls.append(path))
    google = types.ModuleType("google")
    google.colab = colab
    with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
        with patch("urllib.request.build_opener") as opener:
            with pytest.raises(RuntimeError, match="awaiting its verified runtime release"):
                exec(compile(source, "<oracle bootstrap>", "exec"), {})
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
    exec(compile(setup, '<oracle setup>', 'exec'))
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


@pytest.mark.parametrize("in_colab,override,expected", [(True, "custom_check", "oracle_control"),
    (False, None, "oracle_control_smoke"), (False, "custom_check", "custom_check")])
def test_study_routes_config_and_incomplete_report_has_no_aggregation(tmp_path, capsys, in_colab, override, expected):
    calls = []
    config = types.SimpleNamespace(to_dict=lambda: {"test_config": expected})
    def from_file(path):
        calls.append(path)
        return config
    def run(config_arg, path, device):
        assert config_arg is config and device == "cpu"
        calls.append(path)
        return {"status": "incomplete", "message": "Time budget reached."}
    namespace = {"IN_COLAB": in_colab, "os": os, "PROJECT": tmp_path, "RUNS": tmp_path / "runs", "DEVICE": "cpu", "json": json,
                 "OracleControlConfig": types.SimpleNamespace(from_file=from_file), "run_oracle_control": run}
    environment = {} if override is None else {"DISTILLED_ORACLE_CONFIG": override}
    with patch.dict(os.environ, environment, clear=True):
        exec(compile(code_cells()[3].source, "<oracle study>", "exec"), namespace)
    assert calls == [tmp_path / "configs" / f"{expected}.json", tmp_path / "runs/oracle-control-v1"]
    exec(compile(code_cells()[4].source, "<oracle incomplete report>", "exec"), namespace)
    output = capsys.readouterr().out
    assert "ONLY the fourth code cell with the same RUN_NAME" in output
    assert "Do NOT use Run all or restart the runtime" in output
    assert "No aggregation until all planned seeds complete" in output


@pytest.mark.parametrize("supported_count", [0, 1, 3])
@pytest.mark.parametrize("units", [None, .25])
def test_report_primary_secondary_signed_differences_support_and_timing(tmp_path, capsys, supported_count, units):
    values = [.125, -.25, 0]
    def aggregate(items):
        available = [value for value in items if value is not None]
        return dict(mean=sum(available) / len(available) if available else None,
                    min=min(available) if available else None, max=max(available) if available else None,
                    supported_seed_count=len(available), planned_seed_count=3)
    available = [.5 if i < supported_count else None for i in range(3)]
    paired = [values[i] if i < supported_count else None for i in range(3)]
    secondary = [-values[i] if i < supported_count else None for i in range(3)]
    all_score = aggregate([.6, .7, .8])
    model_scores = {metric: aggregate(available) if metric.startswith("diagnostic") else all_score for metric in METRICS}
    target = {mode: {f"{metric}_gain": aggregate(data) for metric in METRICS}
              for mode, data in (("population", paired), ("individual", secondary))}
    group = {"models": {name: model_scores for name in MODELS}, "target_comparisons": target,
             "contextual_comparisons": {name: target["population"] for name in ("distilled_vs_solo", "oracle_vs_solo", "oracle_vs_teacher")},
             "per_seed": []}
    for index, seed in enumerate((71, 81, 91)):
        has_support = index < supported_count
        row = dict(seed=seed, subject_count=20, diagnostic_count=4 if has_support else 0,
                   diagnostic_subject_count=2 if has_support else 0,
                   models={name: {"diagnostic_kl": score if has_support else None}
                           for name, score in (("teacher", .75), ("solo", .625), ("distilled", .5), ("oracle", .375))},
                   target_comparisons={"population": {"diagnostic_kl_gain": paired[index]},
                                       "individual": {"diagnostic_kl_gain": secondary[index]}})
        group["per_seed"].append(row)
    report = {"status": "complete", "output_dir": str(tmp_path), "groups": {"rare": group, "typical": group},
              "selection": {"solo_l2": .1, "primary_mode": "population", "secondary_mode": "individual"},
              "convergence_warnings": ["Inspect the final checkpoint."],
              "timing": {"cumulative_training_seconds": 3, "t4_equivalent_units_at_user_rate": units}}
    for seed in (71, 81):
        folder = tmp_path / f"seed-{seed}"
        folder.mkdir()
        for filename in ("selection_audit.png", "convergence.png"):
            (folder / filename).write_bytes(b"mock display placeholder")
    rendered = []
    display_module = types.SimpleNamespace(display=rendered.append, Markdown=lambda value: value, Image=lambda **kwargs: kwargs)
    with patch.dict(sys.modules, {"IPython.display": display_module}):
        exec(compile(code_cells()[4].source, "<oracle report>", "exec"), {"report": report, "Path": Path})
    text = "\n".join(value for value in rendered if isinstance(value, str))
    assert "Primary outcome: rare-group diagnostic ΔKL with population checkpoints" in text
    assert "positive favors oracle-target training" in text
    assert "Teacher model" in text and "Oracle-target student" in text and "Fixed solo L2=0.1" in text
    assert "population (primary)" in text and "individual (secondary)" in text
    assert "Population ΔKL (primary) | Individual ΔKL (secondary)" in text
    assert "Contextual population-checkpoint comparisons" in text
    positive, negative, tied = (1, 1, 1) if supported_count == 3 else (supported_count, 0, 0)
    assert f"Direction counts (population): {positive} favor oracle, {negative} favor teacher-target, {tied} tied; {supported_count}/3 seeds supported." in text
    assert f"Direction counts (individual): {negative} favor oracle, {positive} favor teacher-target, {tied} tied; {supported_count}/3 seeds supported." in text
    if supported_count:
        assert "| 71 | 4 / 2 | 0.7500 | 0.6250 | 0.5000 | 0.3750 | +0.1250 | -0.1250 |" in text
    if supported_count < 3:
        assert "| 91 | 0 / 0 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |" in text
    if supported_count == 0:
        assert "unavailable (0/3 seeds)" in text
    if supported_count == 3:
        assert "-0.0417 [-0.2500, +0.1250] (3/3 seeds)" in text
        assert "+0.0417 [-0.1250, +0.2500] (3/3 seeds)" in text
    images = [value["filename"] for value in rendered if isinstance(value, dict)]
    assert images == [str(tmp_path / "seed-71/selection_audit.png"), str(tmp_path / "seed-71/convergence.png")]
    output = capsys.readouterr().out
    assert "CHECK REQUIRED: Inspect the final checkpoint." in output
    assert "setup, evaluation and idle billing are additional" in output
    if units is None:
        assert "no proxy is inferred from mixed or unrecognized devices" in output
    else:
        assert "Training-only T4 compute-unit proxy: 0.2500" in output


def test_verifier_rejects_missing_seeds_wrong_gain_sign_and_unavailable_coercion():
    verifier = load_script("verify_oracle_notebook")
    models = {name: {"diagnostic_kl": value} for name, value in (("distilled_population", .4), ("oracle_population", .1),
               ("distilled_individual", None), ("oracle_individual", None))}
    row = {"models": models, "target_comparisons": {"population": {"diagnostic_kl_gain": .3}, "individual": {"diagnostic_kl_gain": None}}}
    group = {"per_seed": [row], "target_comparisons": {"population": {"diagnostic_kl_gain": {}}, "individual": {"diagnostic_kl_gain": {}}}}
    summary = {"status": "complete", "pending_seeds": [], "completed_seeds": [71], "planned_seeds": [71], "groups": {"rare": group}}
    verifier.verify_summary(summary)
    summary["planned_seeds"] = [71, 81]
    with pytest.raises(AssertionError, match="omitted a planned seed"):
        verifier.verify_summary(summary)
    summary["planned_seeds"] = [71]
    row["target_comparisons"]["population"]["diagnostic_kl_gain"] = -.3
    with pytest.raises(AssertionError, match="wrong direction"):
        verifier.verify_summary(summary)
    row["target_comparisons"]["population"]["diagnostic_kl_gain"] = .3
    row["target_comparisons"]["individual"]["diagnostic_kl_gain"] = 0
    with pytest.raises(AssertionError, match="Unavailable"):
        verifier.verify_summary(summary)


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="Pinned archive checks require a Git checkout")
def test_older_notebook_verifiers_default_to_frozen_archives(tmp_path):
    # The added module changes today's package hash; old launchers must keep their old source.
    for name, config in (("baseline", "baseline"), ("robustness", "robustness")):
        verifier = load_script(f"verify_{name}_notebook")
        project, settings = verifier.prepare_archived_project(tmp_path / name)
        assert settings == json.loads((ROOT / f"configs/colab_{config}_source.json").read_text())
        assert not (project / "src/distilled_rnns/oracle_control.py").exists()
        assert (project / "src/distilled_rnns/baseline_audit.py").read_bytes() == (ROOT / "src/distilled_rnns/baseline_audit.py").read_bytes()
        assert 'action="store_true", default=True' in (ROOT / f"scripts/verify_{name}_notebook.py").read_text()
