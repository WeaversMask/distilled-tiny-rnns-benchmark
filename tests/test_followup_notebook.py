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
    return nbformat.read(ROOT / "notebooks/02_colab_followup.ipynb", as_version=4)


def test_followup_notebook_is_clean_and_pinned():
    nb = notebook()
    nbformat.validate(nb)
    cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert len(cells) == 5
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is None
        assert cell.outputs == []
    settings = json.loads((ROOT / "configs/colab_followup_source.json").read_text())
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
    assert code.index('"pytest"') < code.index("smoke = run_followup") < code.index("report = run_followup")
    assert 'if IN_COLAB and not torch.cuda.is_available()' in code


def test_builder_preserves_first_notebook_and_reproduces_second(tmp_path):
    for relative in ("notebooks/01_colab_pilot.ipynb", "configs/colab_followup_source.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    original = (tmp_path / "notebooks/01_colab_pilot.ipynb").read_bytes()
    spec = importlib.util.spec_from_file_location("build_followup_test", ROOT / "scripts/build_followup_colab.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    generated = builder.build(tmp_path)
    assert generated.read_bytes() == (ROOT / "notebooks/02_colab_followup.ipynb").read_bytes()
    assert (tmp_path / "notebooks/01_colab_pilot.ipynb").read_bytes() == original


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
