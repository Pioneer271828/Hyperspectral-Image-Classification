"""Consume existing graph-edge results without replaying real model forwards."""

from collections import Counter

import torch

from evaluation.frozen_training_stages import HEADS, json_pin, pinned, require
from model.research_config import digest
from transfer import wp_cross_backend as backend

SOURCES = {
    "freeze": ("wp08-native-matrix01/freeze.json","faedf4ac600f976c07629f0f0bf604d2d936db719f5e9b253e6061b88c2794a1"),
    "audit": ("wp08-native-matrix01/audit-reviewed.json","0dde73772cf6eaf07291573feba12a00f18cf0f50a5dac12967366de9bc740a0"),
    "search": ("wp08-real-matched-support-20261118/real01/support-audit.json",
               "2acb265c5fa304101579c30bdcde8ba3eb17d6f2fdaa590ecbf66e8364e49e3c"),
    "xuzhou": ("wp08-refreshed-support-20261118/real01/xuzhou-evaluate.json",
               "4cb13fae2a51ced829557362c6f2bc3b338a0f4c6346d76c789ee76b57565f15"),
    "whu-honghu": ("wp08-refreshed-support-20261118/real01/whu-honghu-evaluate.json",
                   "4254c0125cfee5dcc612f4a08f3ac66f420d00954386d9cd872d08fb50f61302"),
}
CASES = ("wp08-nine8-64/joint","wp08-nine8-64/fixed","wp08-twentytwo13-64/random")
FOCUS = {"xuzhou":(5,6),"whu-honghu":(15,21)}


def archived_endpoints(root, freeze, receipt):
    restored = {}
    for case in CASES:
        restored[case] = {}
        for phase,pin in freeze["rows"][case]["endpoints"].items():
            reviewed = receipt["results"][case]["endpoints"][phase]
            require(reviewed["package"]==pin,"Archived tensor is outside its previously reviewed release")
            packet = torch.load(pinned(root,pin),map_location="cpu",weights_only=True)
            require(packet["version"]==2 and packet["registration"]==freeze["registration"]
                    and packet["endpoint"]==phase and packet["step"]==reviewed["step"],
                    "Historical stage-two endpoint differs from its original registration")
            model = backend.restore(packet["config"],"region_support",packet["state"],packet["state_hash"])
            require(model.export_config()==packet["config"],"Archive architecture differs after loading")
            restored[case][phase] = dict(package=pin,state=packet["state_hash"],step=packet["step"],
                tensors=len(packet["state"]),parameters=sum(p.numel() for p in model.parameters()),
                historical_view_contract=digest([packet["trial"],packet["development"],packet["bank"]]),
                forwarding_replayed=False)
    return restored


def class_counts(value, focus):
    rows = value["migrations"]
    result = {}
    for label in focus:
        same = [r for r in rows if r["truth"]==label]
        i = value["reference"]["class_ids"].index(label)
        a,b = (value[key]["confusion_matrix"][i] for key in ("reference","intervention"))
        require(len(same)==sum(a)==sum(b)
                and sum(r["reference"]==label for r in same)==a[i]
                and sum(r["intervention"]==label for r in same)==b[i],
                "Saved per-point transitions disagree with archived class confusion")
        gains = sum(r["reference"]!=label and r["intervention"]==label for r in same)
        losses = sum(r["reference"]==label and r["intervention"]!=label for r in same)
        result[str(label)] = dict(n=len(same),before=a[i],after=b[i],gains=gains,losses=losses)
    return result


def recorded_scene(document, search, scene):
    require(document["scene"]==scene and document["optimizer_updates"]==0
            and document["raw_gt"] is document["query"] is False,
            "Foreign historical graph evaluation")
    dataset = search["scenes"][scene]
    statuses = {role:Counter(r["status"] for r in dataset["rows"] if r["role"]==role)
                for role in ("devtrain","devval")}
    output = {}
    for phase,phase_rows in document["results"].items():
        totals,blocks = {},Counter()
        for block in phase_rows["rows"]:
            role = block["role"]
            blocks[role] += 1
            bank = totals.setdefault(role,{})
            for arm,row in block["released"].items():
                if row["count"]==0:
                    continue
                heads = bank.setdefault(arm,{})
                for head in HEADS:
                    measured = class_counts(row["heads"][head],FOCUS[scene])
                    for label,value in measured.items():
                        totals_by_class = heads.setdefault(head,{}).setdefault(label,Counter())
                        totals_by_class.update(value)
        output[phase] = dict(matched_blocks=phase_rows["matched_blocks"],by_role=blocks,focus=totals)
    return dict(search_status=statuses,original_scope=document["scope"],fresh_head=document["source_has_fresh_head"],
                role_support=document["authorized_per_class"],frozen_development=output,
                independent_accuracy=None,identified_true_signal_sources=None,
                source_limit="Authorized release has no causal emitting-source annotation")


def review_history(root):
    pins = {k:dict(path=str(root/path),sha256=sha) for k,(path,sha) in SOURCES.items()}
    docs = {name:json_pin(root,pin) for name,pin in pins.items()}
    freeze,audit = (docs[name] for name in ("freeze","audit"))
    require(audit["freeze"]==pins["freeze"] and audit["audit_updates"]==0
            and freeze["query"]==freeze["real_updates"]==0 and audit["updates"]==freeze["updates"],
            "Historical regional checkpoint release provenance differs")
    search = docs["search"]
    require(search["raw_gt"] is search["query"] is False and search["optimizer_updates"]==0
            and not search["matched_uses_labels"],"Foreign historical support search")
    return dict(schema="wp12_regional_history_v1",pins=pins,restored=archived_endpoints(root,freeze,audit),
                scenes={name:recorded_scene(docs[name],search,name) for name in FOCUS},
                old_endpoint_forwards=0,real_optimizer_updates=0,raw_gt_loaded=False,query_accessed=False,wp04=False)
