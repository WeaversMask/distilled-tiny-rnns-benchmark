import importlib.util
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

ROOT=Path(__file__).resolve().parents[1]


def cells():
    n=nbformat.read(ROOT/'notebooks/08_colab_rarity.ipynb',as_version=4)
    nbformat.validate(n)
    return [c for c in n.cells if c.cell_type=='code']


def test_reproducible_clean_pinned_builder_and_secure_downloader(tmp_path):
    spec=importlib.util.spec_from_file_location('rarity_builder',ROOT/'scripts/build_rarity_colab.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    for name in ('notebooks/07_colab_main_study.ipynb','scripts/colab_bootstrap.py','configs/colab_rarity_source.json'):
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    previous=(tmp_path/'notebooks/07_colab_main_study.ipynb').read_bytes()
    assert builder.build(tmp_path).read_bytes()==(ROOT/'notebooks/08_colab_rarity.ipynb').read_bytes()
    assert (tmp_path/'notebooks/07_colab_main_study.ipynb').read_bytes()==previous
    c=cells();assert len(c)==5 and all(x.execution_count is None and not x.outputs for x in c)
    assert c[0].source.startswith((ROOT/'scripts/colab_bootstrap.py').read_text())
    assert c[0].source.index('if IN_COLAB and not RARITY_REVISION_READY:')<c[0].source.index('PROJECT = fetch_public_project(')
    assert 'grid_baseline_reference_v1.json' in c[0].source
    assert '("grid_study", "grid_study")' in c[1].source


def test_release_guard_blocks_source_download():
    source=re.sub(r'^RARITY_REVISION_READY = .+$','RARITY_REVISION_READY = False',cells()[0].source,flags=re.MULTILINE)
    calls=[];colab=types.ModuleType('google.colab')
    colab.drive=types.SimpleNamespace(mount=lambda name:calls.append(name))
    google=types.ModuleType('google');google.colab=colab
    with patch.dict(sys.modules,{'google':google,'google.colab':colab}):
        with patch("urllib.request.build_opener") as opener:
            with pytest.raises(RuntimeError,match='awaiting its verified runtime'):
                exec(compile(source,'<bootstrap>','exec'),{})
        opener.assert_not_called()
    assert calls==[]


@pytest.mark.parametrize('changed',['grid_study','main_study','foundation','src/distilled_rnns','configs/grid_baseline_reference_v1.json'])
def test_bootstrap_binds_wrapper_engine_and_trusted_baseline(tmp_path,changed):
    project=tmp_path/'project'
    for folder in ('grid_study','main_study','foundation','src/distilled_rnns'):
        shutil.copytree(ROOT/folder,project/folder,ignore=shutil.ignore_patterns('__pycache__'))
    (project/'configs').mkdir()
    for file in ('main_study_v1.json','grid_baseline_reference_v1.json'):
        shutil.copyfile(ROOT/'configs'/file,project/'configs'/file)
    (project/'pyproject.toml').write_text('')
    target=project/changed
    if target.is_dir():target=next(target.glob('*.py'))
    with target.open('a') as f:f.write('\n# changed\n')
    output=tmp_path/'runs'
    result=subprocess.run([sys.executable,'-I','-S','-c',cells()[0].source],env=dict(os.environ,DISTILLED_PROJECT_ROOT=str(project),DISTILLED_OUTPUT_ROOT=str(output)),capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'Downloaded source differs' in result.stderr
    assert not output.exists()


def test_swapped_production_config_stops_before_training(tmp_path):
    namespace={'RUNS':tmp_path,'PROJECT':tmp_path,'IN_COLAB':True,
               'GridTrancheConfig':types.SimpleNamespace(from_file=lambda p:types.SimpleNamespace(mode='smoke')),
               'run_rarity_tranche':lambda *a,**k:pytest.fail('Wrong config reached training')}
    with pytest.raises(RuntimeError,match='wrong role'):
        exec(compile(cells()[3].source,'<production>','exec'),namespace)


def test_stale_grid_import_rejected_before_install_or_tests(tmp_path):
    module=types.ModuleType('grid_study');module.__file__=str(tmp_path/'other'/'__init__.py')
    with patch.dict(sys.modules,{'grid_study':module}):
        with pytest.raises(RuntimeError,match='Another project revision'):
            exec(compile(cells()[1].source,'<setup>','exec'),{'PROJECT':ROOT,'Path':Path,'sys':sys})
