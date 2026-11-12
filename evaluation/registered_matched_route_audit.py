"""Independently accumulate matched-route development metrics and graph checks."""
from bisect import bisect_right
from collections import Counter, defaultdict
from fractions import Fraction
import math

import numpy as np

from evaluation.classification_metrics import metrics_from_confusion
from evaluation.matched_edge_review import coordinates
from loadData.spatial_tiles import tile_slice
from research_runtime import tensor_hash
from transfer.development_training import sparse_labels
from transfer.graph_route_context import _context


def support_ids(graph):
    nodes = graph.shape[0]
    return {r*nodes+c for r,c in zip(*graph.coalesce().indices().tolist(),strict=True)}


def signature(edges,nodes,xy,thresholds):
    rows,columns,bins = Counter(),Counter(),Counter()
    for edge in edges:
        r,c = divmod(edge,nodes)
        dy,dx = xy[r][0]-xy[c][0],xy[r][1]-xy[c][1]
        rows[r]+=1
        columns[c]+=1
        bins[bisect_right(thresholds,math.sqrt(dy*dy+dx*dx))]+=1
    return [len(edges),dict(rows),dict(columns),dict(bins)]


class Population:
    def __init__(self,run):
        self.run = run
        self.truth = sparse_labels(run.record,run.scene.plan,"validation_support")
        self.count = run.scene.classes
        self.tiles = [tile["id"] for tile in run.record["roles"] if tile["role"]=="devval"]
        self.matrices = defaultdict(lambda:np.zeros((3,self.count,self.count),dtype=np.int64))
        self.losses = defaultdict(lambda:[[],[],[]])
        self.traces = defaultdict(list)
        self.cnn = {}

    def observe(self,name,tile_id,outputs,graph,actual,removed,plan):
        n = graph.shape[0]
        original,changed = support_ids(graph),support_ids(actual)
        if changed!=original-set(removed) or len(changed)!=len(original)-len(removed):
            raise ValueError("Independent actual graph support differs from declared deletion")
        context = _context(self.run.scene,self.run.record,tile_id)
        xy = coordinates(context).tolist()
        counts = signature(removed,n,xy,plan["distance_bins"])
        if name!="none" and counts!=signature(plan["baseline"]["removed"],n,xy,plan["distance_bins"]):
            raise ValueError("Independent edge, degree or distance-bin constraint differs")
        labels = self.truth[tile_slice(self.run.scene.plan["tiles"][tile_id])].reshape(-1)
        ids = np.flatnonzero(labels)
        cnn = tensor_hash(outputs[2])
        if name=="none":
            self.cnn[tile_id] = cnn
        elif cnn!=self.cnn[tile_id]:
            raise ValueError("Independent CNN invariance check differs")
        for head,prob in enumerate(outputs):
            for i in ids:
                y = int(labels[i])-1
                row = prob[i]
                prediction = int(row.argmax())
                self.matrices[name][head,y,prediction]+=1
                self.losses[name][head].append(-math.log(max(float(row[y]),1e-8)))
        self.traces[name].append({"tile_id":tile_id,"original_nnz":len(original),"actual_nnz":len(changed),
                                  "removed":removed,"constraint_counts":counts,"labels":len(ids)})

    def finish(self,result):
        verified = {}
        for name,arm in result["controls"].items():
            if arm["status"]!="evaluated":
                if name in self.traces:
                    raise ValueError("Unavailable arm produced partial metrics")
                verified[name] = {"status":"unavailable","blocking_contexts":arm["blocking_contexts"]}
                continue
            traces = self.traces[name]
            if [row["tile_id"] for row in traces]!=self.tiles:
                raise ValueError("Independent validation population differs")
            if any(any(trace[k]!=other[k] for k in ("tile_id","original_nnz","actual_nnz","removed"))
                   for trace,other in zip(traces,arm["traces"],strict=True)):
                raise ValueError("Independent tile trace differs")
            original = arm["metrics"]
            result_heads = {}
            for i,head in enumerate(("main","graph","cnn")):
                cm = self.matrices[name][i]
                if cm.tolist()!=original["confusion"][i] or cm.sum(1).tolist()!=[5]*self.count:
                    raise ValueError("Independent class confusion or support differs")
                metrics = metrics_from_confusion(cm,range(1,self.count+1))
                nll = math.fsum(self.losses[name][i])/metrics["count"]
                # The original arm sums FP32 per tile. The separate FP64 sum checks
                # that output while the original baseline still requires exact NLL.
                if not math.isclose(nll,original["nll"][i],abs_tol=2e-6,rel_tol=0):
                    raise ValueError("Independent FP64 NLL differs from original block sums")
                baseline = self.matrices["none"][i]
                protections = [(Fraction(int(cm[j,j])-int(baseline[j,j]),5)) for j in range(self.count)]
                result_heads[head] = {"oa":metrics["oa"],"aa":metrics["aa"],"kappa":metrics["kappa"],
                    "macro_f1":metrics["macro_f1"],"nll":nll,"nll_difference":nll-original["nll"][i],
                    "confusion":metrics["confusion_matrix"],"per_class":metrics["per_class"],
                    "zero_recall_classes":[row["class_id"] for row in metrics["per_class"] if row["recall"]==0],
                    "class_protection":[{"class_id":j,"recall_delta_pp":float(100*v),
                                        "passes_2pp":v>=Fraction(-1,50)} for j,v in enumerate(protections,1)]}
            verified[name] = {"status":"evaluated","heads":result_heads,"traces":traces}
        return verified
