import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
from unittest.mock import patch

import nbformat
import pytest

ROOT=Path(__file__).resolve().parents[1]


def cells():
    notebook=nbformat.read(ROOT/'notebooks/07_colab_main_study.ipynb',as_version=4)
    nbformat.validate(notebook)
    return [c for c in notebook.cells if c.cell_type=='code']


def test_pinned_clean_launcher_and_reproducible_builder(tmp_path):
    spec=importlib.util.spec_from_file_location('build_main',ROOT/'scripts/build_main_study_colab.py')
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    for name in ('scripts/colab_bootstrap.py','configs/colab_main_study_source.json'):
        target=tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    assert builder.build(tmp_path).read_bytes()==(ROOT/'notebooks/07_colab_main_study.ipynb').read_bytes()
    code=cells()
    assert len(code)==5 and all(c.execution_count is None and not c.outputs for c in code)
    assert code[0].source.startswith((ROOT/'scripts/colab_bootstrap.py').read_text())
    assert code[0].source.index('if IN_COLAB and not MAIN_STUDY_REVISION_READY')<code[0].source.index('PROJECT = fetch_public_project(')
    assert 'max_seconds' not in code[3].source


def test_release_guard_precedes_source_download():
    import re
    source=re.sub(r'^MAIN_STUDY_REVISION_READY = .+$','MAIN_STUDY_REVISION_READY = False',cells()[0].source,flags=re.MULTILINE)
    calls=[]
    colab=types.ModuleType('google.colab')
    colab.drive=types.SimpleNamespace(mount=lambda path:calls.append(path))
    google=types.ModuleType('google');google.colab=colab
    with patch.dict(sys.modules,{'google':google,'google.colab':colab}):
        with patch("urllib.request.build_opener") as opener:
            with pytest.raises(RuntimeError,match='awaiting its verified runtime'):
                exec(compile(source,'<bootstrap>','exec'),{})
        opener.assert_not_called()
    assert calls==[]


@pytest.mark.parametrize('changed',['main_study','foundation','src/distilled_rnns'])
def test_source_tampering_rejected_before_results_directory(tmp_path,changed):
    project=tmp_path/'project'
    for folder in ('main_study','foundation','src/distilled_rnns'):
        shutil.copytree(ROOT/folder,project/folder,ignore=shutil.ignore_patterns('__pycache__'))
    (project/'configs').mkdir()
    shutil.copyfile(ROOT/'configs/main_study_v1.json',project/'configs/main_study_v1.json')
    (project/'pyproject.toml').write_text('')
    with (next((project/changed).glob('*.py'))).open('a') as f:f.write('\n# changed\n')
    output=tmp_path/'runs'
    env=dict(os.environ,DISTILLED_PROJECT_ROOT=str(project),DISTILLED_OUTPUT_ROOT=str(output))
    result=subprocess.run([sys.executable,'-I','-S','-c',cells()[0].source],env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'Downloaded source differs' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('namespace',['main_study','foundation','distilled_rnns'])
def test_setup_rejects_stale_namespace_before_preflight(tmp_path,namespace):
    module=types.ModuleType(namespace);module.__file__=str(tmp_path/'other'/'module.py')
    with patch.dict(sys.modules,{namespace:module}):
        with pytest.raises(RuntimeError,match='Another project revision'):
            exec(compile(cells()[1].source,'<setup>','exec'),{'PROJECT':ROOT,'sys':sys,'Path':Path})


@pytest.mark.parametrize('cell_index,mode',[(2,'production'),(3,'smoke')])
def test_swapped_runtime_config_role_stops_before_training(cell_index,mode,tmp_path):
    config=types.SimpleNamespace(mode=mode)
    namespace={'RUNS':tmp_path,'PROJECT':tmp_path,'EXPECTED_SOURCE_SHA256':'a'*64,'IN_COLAB':True,
               'MainStudyConfig':types.SimpleNamespace(from_file=lambda path:config),
               'run_main_study':lambda *a,**k:pytest.fail('Wrong role reached training')}
    with pytest.raises(RuntimeError,match='smoke mode|wrong role'):
        exec(compile(cells()[cell_index].source,'<study role>','exec'),namespace)
