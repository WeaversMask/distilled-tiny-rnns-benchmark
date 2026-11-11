import ast
import json
from pathlib import Path
import nbformat


def test_notebook_is_valid_clean_python():
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root / "notebooks/01_colab_pilot.ipynb", as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None
            assert cell.outputs == []


def test_notebook_pins_source_and_requires_tests_before_training():
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root / "notebooks/01_colab_pilot.ipynb", as_version=4)
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    settings = json.loads((root / "configs/colab_source.json").read_text())
    assert settings["source_sha256"] in code
    assert code.index('"pytest"') < code.index('smoke = run_experiment')
    assert 'if IN_COLAB and not torch.cuda.is_available()' in code


def test_public_launcher_embeds_checked_downloader_and_pins_revision():
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root / "notebooks/01_colab_pilot.ipynb", as_version=4)
    first = next(cell.source for cell in nb.cells if cell.cell_type == "code")
    settings = json.loads((root / "configs/colab_source.json").read_text())
    assert first.startswith((root / "scripts/colab_bootstrap.py").read_text())
    assert len(settings["commit"]) == 40
    int(settings["commit"], 16)
    assert f'PINNED_COMMIT = "{settings["commit"]}"' in first
    assert "PROJECT = fetch_public_project(GITHUB_REPOSITORY, PINNED_COMMIT, destination)" in first
    assert "userdata" not in first



def test_colab_setup_activates_new_editable_source_in_running_kernel(tmp_path):
    """A new .pth file must not require restarting the notebook kernel."""
    import shutil
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/01_colab_pilot.ipynb", as_version=4)
    setup = [cell.source for cell in notebook.cells if cell.cell_type == "code"][1]
    setup_file = tmp_path / "setup.py"
    setup_file.write_text(setup)
    project = tmp_path / "downloaded-project"
    package = project / "src/distilled_rnns"
    package.mkdir(parents=True)
    for filename in ("__init__.py", "config.py"):
        shutil.copyfile(root / "src/distilled_rnns" / filename, package / filename)
    script = r"""
from pathlib import Path
import importlib.util, sys, types
PROJECT = Path(sys.argv[1])
setup = Path(sys.argv[2]).read_text()
assert importlib.util.find_spec('distilled_rnns') is None
calls = []
def run(command, **kwargs):
    calls.append(command)
    if 'pip' in command:
        # pip writes this while the kernel is running. Python only processes it
        # on startup; the setup cell must expose the source explicitly now.
        (PROJECT / 'new-editable.pth').write_text(str(PROJECT / 'src') + '\n')
        assert importlib.util.find_spec('distilled_rnns') is None
    return types.SimpleNamespace(returncode=0)
subprocess = types.SimpleNamespace(run=run)
sys.modules['torch'] = types.SimpleNamespace(
    __version__='test', cuda=types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda index: 'test GPU'))
IN_COLAB = True
exec(compile(setup, '<notebook setup>', 'exec'))
from distilled_rnns.config import ExperimentConfig
import distilled_rnns.config as config_module
assert Path(config_module.__file__).resolve() == (PROJECT / 'src/distilled_rnns/config.py').resolve()
assert ExperimentConfig().n_subjects == 1000
assert len(calls) == 2 and 'pip' in calls[0] and 'pytest' in calls[1]
print('Running-kernel import regression passed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(project), str(setup_file)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Running-kernel import regression passed" in result.stdout
