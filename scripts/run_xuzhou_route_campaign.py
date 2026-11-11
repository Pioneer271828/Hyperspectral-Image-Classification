"""Execute the frozen, bounded Xuzhou route campaign on its authorized LXC."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from research_runtime import code_snapshot, require_frozen_code
from transfer import route_campaign_allocation as allocation, xuzhou_route_campaign as campaign
from transfer.container_runtime import training_container_kind


def pin(value):
    path, checksum = value
    return {"path":str(Path(path).resolve(strict=True)),"sha256":checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit",required=True)
    commands = parser.add_subparsers(dest="action",required=True)
    prepare = commands.add_parser("register")
    prepare.add_argument("--design",nargs=2,required=True,metavar=("PATH","SHA256"))
    prepare.add_argument("--prefix-audit",nargs=2,required=True,metavar=("PATH","SHA256"))
    prepare.add_argument("--issue",type=int,required=True)
    for name in ("run-donor","audit-donor","match","run-arm","audit-arms"):
        command = commands.add_parser(name)
        command.add_argument("--campaign",nargs=2,required=True,metavar=("PATH","SHA256"))
        if name in ("run-arm","audit-arms"):
            command.add_argument("--allocation",nargs=2,required=True,metavar=("PATH","SHA256"))
        if name in ("run-donor","run-arm"):
            command.add_argument("--segment",required=True,choices=tuple(allocation.segments(256)))
            command.add_argument("--resume",nargs=2,metavar=("PATH","SHA256"))
        if name=="run-arm":
            command.add_argument("--arm",required=True,choices=allocation.ARM_NAMES)
    args = parser.parse_args()
    if not training_container_kind() or not __debug__:
        parser.error("Authorized Linux container with assertions required")
    torch.set_num_threads(1)
    code = code_snapshot()
    if code["commit"]!=args.expected_commit:
        parser.error("Clean exact consumer commit required")
    require_frozen_code(code)
    if args.action=="register":
        result = campaign.register(pin(args.design),pin(args.prefix_audit),code,issue=args.issue)
    else:
        run = pin(args.campaign)
        if args.action=="audit-donor":
            result = campaign.audit_donor(run,code)
        elif args.action=="match":
            result = campaign.match_donor(run,code)
        elif args.action=="audit-arms":
            result = campaign.audit_arms(run,pin(args.allocation),code)
        else:
            resume = None if args.resume is None else pin(args.resume)
            if args.action=="run-donor":
                root,config = campaign.load_campaign(run,code)
                result = allocation.execute(root,run,config,"donor",args.segment,code,resume=resume)
            else:
                arms = pin(args.allocation)
                root,config = campaign.load_arms(run,arms,code)
                result = allocation.execute(root,arms,config,args.arm,args.segment,code,resume=resume)
    print(json.dumps(result))


if __name__=="__main__":
    main()
