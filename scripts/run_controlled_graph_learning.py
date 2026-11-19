"""Register, execute and audit one bounded synthetic regional graph campaign."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transfer.curriculum_cli import _code
from transfer.registered_controlled_graph import register,run_phase


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-commit",required=True)
    commands=p.add_subparsers(dest="action",required=True)
    r=commands.add_parser("register")
    r.add_argument("--out",required=True)
    donor = commands.add_parser("donor-register")
    donor.add_argument("--out",required=True,type=Path)
    execute = commands.add_parser("donor-run")
    execute.add_argument("--directory",required=True,type=Path)
    execute.add_argument("--registration-sha",required=True)
    execute.add_argument("--case",default="all")
    for key in ("stage","freeze","audit"):
        c=commands.add_parser(key)
        c.add_argument("--registration",nargs=2,required=True,metavar=("PATH","SHA256"))
        c.add_argument("--notice",nargs=2,required=True,metavar=("PATH","SHA256"))
        if key=="stage":
            c.add_argument("--group",choices=("main","fixed_final"),required=True)
            c.add_argument("--stage",choices=("full","prefix","resume","completed"),required=True)
        if key=="audit":
            c.add_argument("--freeze",nargs=2,required=True,metavar=("PATH","SHA256"))
    a=p.parse_args()
    torch.set_num_threads(1)
    code=_code(a.expected_commit)
    if a.action in ("donor-register","donor-run"):
        from scripts.run_frozen_band_region import external_directory
        from transfer.donor_retrieval_campaign import register as register_donor, run as run_donor
        if a.action=="donor-register":
            print(json.dumps(register_donor(external_directory(a.out),code),sort_keys=True))
        else:
            run_donor(external_directory(a.directory),code,a.registration_sha,a.case)
        return
    pin=lambda field:dict(zip(("path","sha256"),getattr(a,field),strict=True))
    if a.action=="register":
        out=register(a.out,code)
    elif a.action=="stage":
        out=run_phase(pin("registration"),pin("notice"),code,a.stage,a.group)
    else:
        from transfer.controlled_graph_audit import freeze,audit
        out=(freeze(pin("registration"),pin("notice"),code) if a.action=="freeze" else
             audit(pin("registration"),pin("notice"),pin("freeze"),code))
    print(json.dumps(out,sort_keys=True))


if __name__=="__main__":
    main()
