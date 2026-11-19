"""Equal-budget sparse message training and seed/graph scoped recovery."""

from collections import Counter, defaultdict
from copy import deepcopy
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from evaluation.classification_metrics import classification_metrics
from evaluation.graph_edge_intervention import delete_support
from model.VCASAHGCN import SharedHGCN
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.donor_retrieval_graph import BATCH, ORDERS, ROLES, UPDATES, observe, sample_graph
from transfer.g3_training import state_hash

HEADS = ("logit_local", "logit_graph", "logit_fused")
RATE = .0125
VALIDATION = (0, *range(10, UPDATES+1, 10))
EVAL_DRAWS = 4


class MessageClassifier(nn.Module):
    """Same parameter roster in every arm. No oracle features enter these layers."""
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.graph_path = SharedHGCN(3, 12, 2).eval()
            self.local_path = nn.Sequential(nn.Linear(3, 12), nn.ReLU(), nn.Linear(12, 2))
            self.fuse = nn.Linear(4, 2)
            self.route = nn.Parameter(torch.zeros(2))

    def forward(self, inputs, graph):
        if not graph.is_sparse or graph._nnz()<=0:
            raise ValueError("Actual positive sparse COO message adjacency required")
        local = self.local_path(inputs)
        graph_signal = self.graph_path.forward_stage2(self.graph_path.forward_stage1(inputs, graph), graph)
        fused = self.fuse(torch.cat((local, graph_signal), dim=-1))
        return dict(zip(HEADS, (local, graph_signal, fused), strict=True))


def blkdiag(graphs):
    n = graphs[0].shape[0]
    ij = torch.cat([graph.indices()+i*n for i,graph in enumerate(graphs)], dim=1)
    weights = torch.cat([graph.values() for graph in graphs])
    return torch.sparse_coo_tensor(ij, weights, (len(graphs)*n,)*2, is_coalesced=True)


def tensor_tree(value):
    """Hash model/optimizer/RNG-independent state using explicit tensor leaves."""
    if torch.is_tensor(value):
        return tensor_hash(value)
    if isinstance(value, dict):
        return {str(k): tensor_tree(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree(v) for v in value]
    return value


def scientific_state(value):
    return digest(tensor_tree({k:v for k,v in value.items() if k not in ("cost","state_sha256")}))


class Data:
    def __init__(self, seed, layout, direction):
        self.seed = seed
        self.rows = {role: tuple(sample_graph(seed, layout, direction, role, i) for i in range(size))
                     for role,(_,size) in ROLES.items()}
        self.identities = {role: [g.id for g,_ in rows] for role,rows in self.rows.items()}
        if len({g for rows in self.identities.values() for g in rows}) != sum(map(len,self.identities.values())):
            raise ValueError("Train/validation/confirmation graph identity overlaps")

    def entry(self, role, index, draw):
        graph, known = self.rows[role][index]
        x, labels, truth = observe(graph, known, seed=self.seed, role=role, index=index, draw=draw)
        return graph, x, labels, truth

    def released_training_row(self, index, draw):
        graph, x, labels, _ = self.entry("training", index, draw)
        return graph, x, labels

    def oracle_pair(self, role, index):
        return self.rows[role][index][1]


def choice(model, graph, arm, nonce, *, selected_model=None, oracle=None):
    if arm=="oracle":
        if oracle is None:
            raise ValueError("Only declared oracle control gets synthetic pairing")
        return ORDERS.index(oracle), None
    if arm=="local":
        return "local", None
    if arm=="random":
        return int(torch.randint(len(ORDERS),(),generator=torch.Generator().manual_seed(nonce))), None
    if arm=="spectral":
        return graph.spectral_choice, None
    if arm not in ("learned","fixed_final","fixed_cut"):
        raise ValueError("Unknown registered donor arm")
    scorer = model.route if arm=="learned" else selected_model
    if scorer is None:
        raise ValueError("Fixed reinitialization lacks its donor parent")
    options = graph.option_features.to(scorer.device)
    logits = (options / options.new_tensor([1., 4.])) @ scorer
    if arm!="learned" or nonce is None:
        return int(logits.argmax()), None
    p = logits.softmax(dim=0)
    selected = int(torch.multinomial(p.detach().cpu(),1,generator=torch.Generator().manual_seed(nonce)))
    return selected, logits.log_softmax(0)[selected]


def adjacency_for(graph, selected, arm):
    matrix = graph.adjacency(selected)
    if arm=="fixed_cut":
        n = matrix.shape[0]
        removed = [i*n+j for i,j in graph.pairs(selected) if i!=j]
        return delete_support(matrix, removed)
    return matrix


def role_nonce(seed, role, index, draw):
    band = {"training": 983, "validation": 1051, "confirmation": 1213}[role]
    return seed*10_000_000 + 100_000*band + 100*draw + index


def loadout(model, entries, selected, *, device):
    matrices = [matrix.to(device) for matrix in selected]
    adjacency = blkdiag(matrices)
    inputs = torch.cat([entry[1] for entry in entries]).to(device)
    labels = torch.cat([entry[2] for entry in entries]).to(device)
    targets = torch.cat([torch.as_tensor(entry[0].targets)+8*i for i,entry in enumerate(entries)]).to(device)
    return model(inputs, adjacency), labels, targets, adjacency._nnz()


class Run:
    def __init__(self, data, arm, *, scope, parent=None, device="cpu"):
        self.data, self.arm, self.device = data, arm, device
        self.model = MessageClassifier(data.seed).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=RATE)
        self.parent = parent
        if arm in ("fixed_final","fixed_cut") and (not isinstance(parent,dict) or len(parent)!=2):
            raise ValueError("A sealed learned final graph scorer is required")
        if arm not in ("fixed_final","fixed_cut") and parent is not None:
            raise ValueError("Dynamic arms accept no external predictor")
        self.scorer = None if parent is None else torch.as_tensor(parent["route"])
        self.initial = state_hash(self.model)
        self.contract = digest(dict(scope=scope, arm=arm, seed=data.seed, splits=data.identities, parent=parent,
                                    initial=self.initial, device=device))
        self.cursor = 0
        self.trajectory, self.validation = [], []
        self.selected = None
        self.cost = defaultdict(float)
        self.weight = torch.tensor([1.,3.],device=device)

    def route(self, graph, nonce, *, oracle=None):
        if self.arm!="oracle" and oracle is not None:
            raise ValueError("Truth cannot enter candidate selection")
        return choice(self.model, graph, self.arm, nonce, selected_model=self.scorer, oracle=oracle)

    def update(self):
        step = self.cursor+1
        if not 1<=step<=UPDATES:
            raise ValueError("Training update exceeds the registered budget")
        entries, masks, choices, log_probabilities = [], [], [], []
        started = perf_counter()
        for offset in range(BATCH):
            index = ((step-1)*BATCH+offset) % len(self.data.rows["training"])
            entry = self.data.released_training_row(index,step)
            graph = entry[0]
            oracle = self.data.oracle_pair("training",index) if self.arm=="oracle" else None
            selected, log_p = self.route(graph,role_nonce(self.data.seed,"training",index,step),oracle=oracle)
            entries.append(entry)
            masks.append(adjacency_for(graph, selected, self.arm))
            choices.append(selected)
            if log_p is not None:
                log_probabilities.append(log_p)
        self.cost["training_input_and_route_seconds"]+=perf_counter()-started
        started = perf_counter()
        outputs, labels, targets, nnz = loadout(self.model, entries, masks, device=self.device)
        self.cost["training_message_forward_seconds"]+=perf_counter()-started
        losses = [F.cross_entropy(pred[targets], labels, reduction="none", weight=self.weight) for pred in outputs.values()]
        classifier = torch.stack(losses).mean()
        policy = classifier.new_zeros(())
        if self.arm=="learned":
            costs = (losses[1]+losses[2]).detach().reshape(BATCH,4).mean(dim=1)
            # Leave-one-out baseline uses only training label loss, no donor IDs.
            advantage = costs-(costs.sum()-costs)/(BATCH-1)
            policy = (advantage * torch.stack(log_probabilities)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        started = perf_counter()
        (classifier+policy).backward()
        gradients = {name: float(sum(p.grad.detach().abs().sum() for p in module.parameters() if p.grad is not None))
                     for name,module in (("graph",self.model.graph_path),("local",self.model.local_path),("fusion",self.model.fuse))}
        gradients["selector"] = float(self.model.route.grad.abs().sum()) if self.model.route.grad is not None else 0.
        self.cost["backward_seconds"]+=perf_counter()-started
        started = perf_counter()
        self.optimizer.step()
        self.cost["optimizer_seconds"]+=perf_counter()-started
        self.cursor = step
        self.trajectory.append(dict(step=step, loss=float(classifier.detach()), policy=float(policy.detach()),
                                    gradients=gradients, selected=choices, nnz=nnz,
                                    scorer=self.model.route.detach().cpu().tolist()))

    def validate(self):
        if self.cursor not in VALIDATION or any(v["step"]==self.cursor for v in self.validation):
            raise ValueError("Duplicate or out-of-plan validation")
        value = self.evaluate("validation", include_predictions=False)
        row = dict(step=self.cursor, nll=value["heads"]["logit_fused"]["nll"], retrieval=value["retrieval"],
                   graph_aa=value["heads"]["logit_graph"]["aa"], fused_aa=value["heads"]["logit_fused"]["aa"])
        self.validation.append(row)
        if self.selected is None or row["nll"]<self.selected["nll"]:
            self.selected = dict(step=self.cursor, nll=row["nll"], model=deepcopy(self.model.state_dict()))

    @torch.no_grad()
    def evaluate(self, role, *, intervention=None, include_predictions=True):
        started = perf_counter()
        per_graph, predictions, labels, null_labels = [], defaultdict(list), [], []
        retrieval, selected_routes = 0, Counter()
        for index,(graph,known) in enumerate(self.data.rows[role]):
            rows = [self.data.entry(role,index,d) for d in range(EVAL_DRAWS)]
            graph_entries, masks, total = [], [], 0
            for draw,entry in enumerate(rows):
                truth = entry[3]
                nonce = None if self.arm=="learned" else role_nonce(self.data.seed,role,index,draw)
                option,_ = self.route(graph,nonce,oracle=known if self.arm=="oracle" else None)
                matrix = adjacency_for(graph, option, self.arm)
                if intervention is not None:
                    matrix = intervention(graph, known, option, matrix)
                edges = set(map(tuple, matrix.indices().T.tolist()))
                total += sum((int(graph.targets[i]),int(graph.donors[j])) in edges for i,j in enumerate(known))
                selected_routes["none" if isinstance(option,str) else str(option)] += 1
                graph_entries.append(entry)
                masks.append(matrix)
                labels.extend(entry[2].tolist())
                null_labels.extend([int(i==truth.null_positive) for i in range(4)])
            result, _, targets, nnz = loadout(self.model,graph_entries,masks,device=self.device)
            for head,values in result.items():
                predictions[head].extend(values[targets].softmax(dim=1).cpu().tolist())
            retrieval+=total
            per_graph.append(dict(id=graph.id, donor_hits=total, opportunities=4*EVAL_DRAWS, nnz=nnz,
                                  graph_hashes=[tensor_hash(m) for m in masks], candidates=len(graph.support.pool)))
        self.cost[role+"_seconds"]+=perf_counter()-started
        summary = summarize(predictions,labels)
        null = summarize(predictions,null_labels)
        value = dict(heads=summary, null_heads=null, retrieval=dict(hits=retrieval,total=len(labels)),
                     per_graph=per_graph, selection_counts=dict(selected_routes))
        if include_predictions:
            value.update(probabilities=dict(predictions), labels=labels, null_labels=null_labels)
        return value

    def pack(self):
        value = dict(contract=self.contract, cursor=self.cursor, model=deepcopy(self.model.state_dict()),
                     optimizer=deepcopy(self.optimizer.state_dict()), selected=deepcopy(self.selected),
                     validation=deepcopy(self.validation), trajectory=deepcopy(self.trajectory), cost=dict(self.cost))
        return dict(**value, state_sha256=scientific_state(value))

    def restore(self, value, *, expected_cursor):
        if (value["contract"]!=self.contract or value["cursor"]!=expected_cursor or self.cursor
                or scientific_state(value)!=value["state_sha256"] or len(value["trajectory"])!=expected_cursor
                or [row["step"] for row in value["validation"]]!=[step for step in VALIDATION if step<=expected_cursor]):
            raise ValueError("Donor checkpoint graph, initialization, seed, phase or cursor differs")
        self.model.load_state_dict(value["model"],strict=True)
        self.optimizer.load_state_dict(value["optimizer"])
        self.cursor,self.selected,self.trajectory,self.validation = (value[name] for name in
                                                                     ("cursor","selected","trajectory","validation"))


def summarize(predictions, labels):
    truth = torch.tensor(labels)
    results = {}
    for head,rows in predictions.items():
        p = torch.as_tensor(rows,dtype=torch.float64)
        # Class 0 is reserved for unlabeled pixels in the shared metrics API.
        metrics = classification_metrics((truth+1).tolist(),(p.argmax(1)+1).tolist(),[1,2])
        metrics["nll"] = float(-p[torch.arange(truth.numel()),truth].clamp(min=1e-20).log().mean())
        results[head] = metrics
    return results
