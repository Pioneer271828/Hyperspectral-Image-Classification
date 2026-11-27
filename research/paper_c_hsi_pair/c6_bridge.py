"""Reuse the immutable C6 reader without changing its original authority."""
from pathlib import Path
import sys

C6=Path(__file__).resolve().parent.parent/'paper_c_hsi_admission'
sys.path.insert(0,str(C6))
from bank_reader import ReleasedBank,canonical,sha,pinned_json
from batch import patch_batch
from inspect_bank import guard

BASE='ed57e389fd83ca9693a973f97168106ca8534074'


def check_source(expected):
    import os
    import subprocess
    repo=Path(__file__).resolve().parents[2]
    if os.name!='posix' or subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=expected or subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Clean expected Linux source required')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,expected],cwd=repo,check=True,capture_output=True)
    if subprocess.check_output(['git','diff',BASE,expected,'--','research/paper_c_hsi_admission'],cwd=repo):raise ValueError('C6 dependency changed')


def frozen_source(expected,out):
    result=guard(expected,out);check_source(expected)
    return result
