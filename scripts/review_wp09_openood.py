"""Run the optional OpenOOD oracle on a preauthorized synthetic E4c bundle."""
from argparse import ArgumentParser
import json
from pathlib import Path
from evaluation.wp09_openood import prepare, evaluate_approval


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--expected-commit", required=True)
    for name in ("authority","source-receipt","approval"):
        parser.add_argument("--"+name,nargs=2)
    parser.add_argument("--openood", type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--temperatures",nargs=3,type=float,default=(1.,2.,.5))
    parser.add_argument("--coverage",type=float,default=.95)
    args=parser.parse_args()

    def pin(values):
        if not values:
            parser.error("Missing independently approved input pin")
        return dict(zip(("path","sha256"),values,strict=True))

    authority=pin(args.authority)
    src=pin(args.source_receipt)
    if args.stage=="prepare":
        result=prepare(authority,src,args.openood,out=args.out,expected_commit=args.expected_commit,
                       temperature=list(args.temperatures),coverage=args.coverage)
    else:
        result=evaluate_approval(pin(args.approval),authority,src,args.openood,out=args.out,expected_commit=args.expected_commit)
    print(json.dumps(result))


if __name__=="__main__":
    main()
