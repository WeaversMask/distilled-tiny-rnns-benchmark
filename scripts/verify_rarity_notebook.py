"""Execute notebook08's actual five cells with development-only local fixtures."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]


def archive_project(destination,settings):
    payload=subprocess.check_output(['git','archive','--format=zip','--prefix=project/',settings['commit']],cwd=ROOT)
    spec=importlib.util.spec_from_file_location('bootstrap',ROOT/'scripts/colab_bootstrap.py')
    bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
    return bootstrap._extract_project_archive(payload,destination)


def verify_summary(summary):
    if summary['status']!='complete' or summary['pending_cells'] or summary['completed_cells']!=summary['planned_cells']:
        raise AssertionError('Every scheduled cell must complete')
    if len(summary['cells'])!=3:
        raise AssertionError('Expected exactly three distinct rarity cells')
    baseline=summary['baseline_reference']['summary']
    if baseline['status']!='complete' or baseline['config']['rare_fraction']!=.2:
        raise AssertionError('Local baseline must use the unchanged tiny first-tranche mode')
    fractions=[]
    for cell in summary['cells'].values():
        if cell['status']!='complete' or cell['pending_populations']:
            raise AssertionError('Cell has unfinished populations')
        fractions.append(cell['config']['rare_fraction'])
        if cell['config']['mode']!='smoke':raise AssertionError('Local verification cannot evaluate production populations')
        for kind in cell['kinds'].values():
            if kind['planned_seeds']!=kind['completed_seeds']:
                raise AssertionError('Missing planned development seed')
            for group in kind['groups'].values():
                for mode in ('population','individual'):
                    for metric in ('prediction_gain','behavioral_distortion','quadrant_fraction'):
                        a=group['primary'][mode][metric]
                        if a['supported_seed_count']!=len(kind['planned_seeds']):
                            raise AssertionError('Missing primary support')
    if sorted(fractions)!=[.3,.4,.5]:raise AssertionError('Local cells must use only the three smoke rarities')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pinned-archive',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/08_colab_rarity.executed.ipynb')
    args=parser.parse_args(argv)
    settings=json.loads((ROOT/'configs/colab_rarity_source.json').read_text())
    with tempfile.TemporaryDirectory(prefix='rarity-notebook-') as folder:
        scratch=Path(folder).resolve()
        project=archive_project(scratch/'project',settings) if args.pinned_archive else ROOT
        kernel=scratch/'kernels/python3';kernel.mkdir(parents=True)
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Rarity verification','language':'python'}))
        env=dict(os.environ,DISTILLED_PROJECT_ROOT=str(project),DISTILLED_OUTPUT_ROOT=str(scratch/'runs'),
                 CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(scratch/'matplotlib'),IPYTHONDIR=str(scratch/'ipython'),
                 JUPYTER_RUNTIME_DIR=str(scratch/'jupyter'),PYTHONDONTWRITEBYTECODE='1')
        manager=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(scratch/'kernels')]))
        notebook=nbformat.read(ROOT/'notebooks/08_colab_rarity.ipynb',as_version=4)
        NotebookClient(notebook,km=manager,timeout=900,resources={'metadata':{'path':str(project)}}).execute(env=env)
        counts=[c.execution_count for c in notebook.cells if c.cell_type=='code']
        if counts!=[1,2,3,4,5]:raise AssertionError('All five shipped cells must execute')
        for name in ('rarity-smoke-'+settings['source_sha256'][:12],'main-study-rarity-v1'):
            verify_summary(json.loads((scratch/'runs'/name/'summary.json').read_text()))
        args.output.parent.mkdir(parents=True,exist_ok=True);nbformat.write(notebook,args.output)
        print(json.dumps({'output':str(args.output),'counts':counts,'runtime':'pinned archive' if args.pinned_archive else 'checkout','source_sha256':settings['source_sha256'],'all_smoke_cells_complete':True},indent=2))


if __name__=='__main__':main()
