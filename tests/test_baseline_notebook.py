import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return nbformat.read(ROOT / "notebooks/03_colab_baseline_audit.ipynb", as_version=4)


def test_baseline_notebook_is_clean_and_pinned():
    nb = notebook()
    nbformat.validate(nb)
    cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert len(cells) == 5
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is None
        assert cell.outputs == []
    settings = json.loads((ROOT / "configs/colab_baseline_source.json").read_text())
    assert len(settings["commit"]) == 40
    int(settings["commit"], 16)
    assert len(settings["source_sha256"]) == 64
    int(settings["source_sha256"], 16)
    first = cells[0].source
    assert first.startswith((ROOT / "scripts/colab_bootstrap.py").read_text())
    assert f'PINNED_COMMIT = "{settings["commit"]}"' in first
    assert f'EXPECTED_SOURCE_SHA256 = "{settings["source_sha256"]}"' in first
    assert "PROJECT = fetch_public_project(GITHUB_REPOSITORY, PINNED_COMMIT, destination)" in first
    assert "userdata" not in first
    code = "\n".join(cell.source for cell in cells)
    assert code.index('"pytest"') < code.index("smoke = run_baseline_audit") < code.index("report = run_baseline_audit")
    assert 'if IN_COLAB and not torch.cuda.is_available()' in code
    assert f"BASELINE_REVISION_READY = {settings.get('ready', False)!r}" in first
    assert code.index("if IN_COLAB and not BASELINE_REVISION_READY:") < code.index("PROJECT = fetch_public_project(")


def test_builder_preserves_previous_notebooks_and_reproduces_third(tmp_path):
    for relative in ("notebooks/01_colab_pilot.ipynb", "notebooks/02_colab_followup.ipynb", "configs/colab_baseline_source.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    original = {name: (tmp_path / "notebooks" / name).read_bytes() for name in ("01_colab_pilot.ipynb", "02_colab_followup.ipynb")}
    spec = importlib.util.spec_from_file_location("build_baseline_test", ROOT / "scripts/build_baseline_colab.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    generated = builder.build(tmp_path)
    assert generated.read_bytes() == (ROOT / "notebooks/03_colab_baseline_audit.ipynb").read_bytes()
    for name, contents in original.items():
        assert (tmp_path / "notebooks" / name).read_bytes() == contents


@pytest.mark.parametrize("stale_module", [False, True])
def test_running_kernel_import_activation_and_stale_revision_guard(tmp_path, stale_module):
    """Exercise the real setup cell in an isolated interpreter, without pip/GPU."""
    setup = [cell.source for cell in notebook().cells if cell.cell_type == "code"][1]
    setup_file = tmp_path / "setup.py"
    setup_file.write_text(setup)
    project = tmp_path / "downloaded-project"
    package = project / "src/distilled_rnns"
    package.mkdir(parents=True)
    for filename in ("__init__.py", "config.py"):
        shutil.copyfile(ROOT / "src/distilled_rnns" / filename, package / filename)
    script = r'''
from pathlib import Path
import importlib.util, sys, types
PROJECT = Path(sys.argv[1])
setup = Path(sys.argv[2]).read_text()
stale = sys.argv[3] == 'True'
calls = []
def run(command, **kwargs):
    calls.append(command)
    if 'pip' in command:
        (PROJECT / 'new-editable.pth').write_text(str(PROJECT / 'src') + '\n')
        assert importlib.util.find_spec('distilled_rnns') is None
    return types.SimpleNamespace(returncode=0)
subprocess = types.SimpleNamespace(run=run)
sys.modules['torch'] = types.SimpleNamespace(
    __version__='test', cuda=types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda index: 'test GPU'))
if stale:
    sys.modules['distilled_rnns.config'] = types.SimpleNamespace(
        __file__=str(PROJECT.parent / 'old-project/src/distilled_rnns/config.py'))
IN_COLAB = True
try:
    exec(compile(setup, '<follow-up setup>', 'exec'))
except RuntimeError as error:
    assert stale and 'Restart' in str(error), str(error)
    assert calls == []
else:
    assert not stale
    import distilled_rnns.config as config_module
    assert Path(config_module.__file__).resolve() == (PROJECT / 'src/distilled_rnns/config.py').resolve()
    assert config_module.ExperimentConfig().n_subjects == 1000
    assert len(calls) == 2 and 'pip' in calls[0] and 'pytest' in calls[1]
print('Kernel source activation and revision guard passed')
'''
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(project), str(setup_file), str(stale_module)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Kernel source activation and revision guard passed" in result.stdout


def test_results_cell_handles_supported_and_unavailable_diagnostics(tmp_path, capsys):
    """Render the actual report cell without requiring a training run or display UI."""
    import types
    from unittest.mock import patch

    models = ("teacher", "solo", "distilled", "coinflip", "frequency", "solo_legacy",
              "solo_individual", "solo_population", "distilled_individual", "distilled_population")
    supported = dict(mean=.67, min=.66, max=.68, supported_seed_count=2, planned_seed_count=2)
    unavailable = dict(mean=None, min=None, max=None, supported_seed_count=0, planned_seed_count=2)
    comparisons = {metric: supported for metric in ("likelihood_gain", "fidelity_loss", "diagnostic_likelihood_gain")}
    group = {
        "models": {name: {metric: (unavailable if metric.startswith("diagnostic") else supported)
                           for metric in ("nll", "kl", "diagnostic_nll", "diagnostic_kl")} for name in models},
        "comparisons": {name: comparisons for name in ("solo", "solo_legacy", "coinflip", "frequency")},
        "matched_comparisons": {name: comparisons for name in ("individual", "population")},
        "per_seed": [dict(seed=71, diagnostic_count=0, diagnostic_subject_count=0),
                     dict(seed=81, diagnostic_count=0, diagnostic_subject_count=0)],
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
            (tmp_path / folder / filename).write_bytes(b"image placeholder for the mocked display")
    rendered = []
    display_module = types.SimpleNamespace(display=rendered.append, Markdown=lambda value: value,
                                           Image=lambda **kwargs: kwargs)
    results = [cell.source for cell in notebook().cells if cell.cell_type == "code"][4]
    with patch.dict(sys.modules, {"IPython.display": display_module}):
        exec(compile(results, "<baseline audit results>", "exec"), {"report": report, "Path": Path})
    text = "\n".join(value for value in rendered if isinstance(value, str))
    assert "Fixed 50/50" in text and "Training choice frequency" in text
    assert "Matched method" in text and "unavailable (0/2 seeds)" in text
    assert "Diagnostic KL" in text
    images = [value["filename"] for value in rendered if isinstance(value, dict)]
    assert images == [str(tmp_path / "calibration/selection_audit.png"),
                      str(tmp_path / "seed-71/selection_audit.png"),
                      str(tmp_path / "seed-71/convergence.png")]
    output = capsys.readouterr().out
    assert "Additional selection-audit figure: " + str(tmp_path / "seed-81/selection_audit.png") in output
    assert "Additional convergence figure: " + str(tmp_path / "calibration/convergence.png") in output
    assert "Additional convergence figure: " + str(tmp_path / "seed-81/convergence.png") in output
