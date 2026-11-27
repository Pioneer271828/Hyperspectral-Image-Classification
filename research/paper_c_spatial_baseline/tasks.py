"""Only model adapters. Reuses the pinned C3 components, never an author trainer."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .protocol import AUTHOR, NATIVE, LICENSE, METHODS, MODEL, LOSSES, sha, require


def clean(path, expected):
    git = lambda *a: subprocess.check_output(['git', '-C', str(path), *a], text=True).strip()
    require(git('rev-parse', 'HEAD') == expected and not git('status', '--porcelain'), 'Expected clean full source SHA differs')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def dependencies(root, native, author):
    clean(native, NATIVE)
    require(sha(Path(author)/'LICENSE') == LICENSE, 'Author license bytes differ')
    adapter = module(Path(root)/'research/paper1_c/adapter.py', 'c13_prior_adapter')
    upstream = adapter.load_author_model(author)
    # The clean frozen native checkout has first import priority.
    require(not any(n == 'model' or n.startswith('model.') for n in sys.modules), 'Native model imported before dependency pin')
    sys.path.insert(0, str(Path(native).resolve()))
    return adapter, upstream


def centers_metric(outputs, labels, classes, *, logits=False):
    p = torch.cat(outputs); y = torch.cat(labels)
    if logits:
        nll = float(F.cross_entropy(p, y)); pred = p.argmax(1)
    else:
        require(p.shape == (len(y), classes) and torch.isfinite(p).all().item() and (p >= 0).all().item()
                and torch.allclose(p.sum(1), torch.ones(len(y)), atol=1e-5, rtol=1e-5), 'Invalid fused probabilities')
        nll = float(-p[torch.arange(len(y)), y].clamp_min(1e-8).log().mean()); pred = p.argmax(1)
    cm = torch.bincount(y*classes+pred, minlength=classes**2).reshape(classes, classes)
    support = cm.sum(1); require((support > 0).all().item(), 'Missing class support')
    return {'nll': nll, 'oa': float(cm.diag().sum()/len(y)), 'confusion': cm.tolist(),
            'support': support.tolist(), 'recall': (cm.diag().double()/support).tolist(),
            'prediction': pred.tolist(), 'points': len(y)}


class Task:
    def __init__(self, scene, data, method, adapter, author):
        require(method in METHODS, 'Unknown method')
        self.scene, self.data, self.method = scene, data, method
        self.adapter, self.author = adapter, author
        if method == 'dsnet':
            self.model = adapter.Adapter(author, data['bands'], data['classes'])
        else:
            from model.VCASAHGCN import MultiDatasetVCASAHGCN
            self.model = MultiDatasetVCASAHGCN({scene: data['classes']}, region_residual_mode=method, **MODEL)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                          weight_decay=0., amsgrad=False)
        yy, xx = np.mgrid[:7, :7]
        self.coordinates = torch.tensor(np.stack((yy/6, xx/6), -1), dtype=torch.float32)
        self.wavelengths = torch.tensor(data['wavelengths'])

    def inputs(self, role, i):
        """The SAME normalized spectral values; coordinates are a fixed grid."""
        return torch.from_numpy(self.data['roles'][role]['spectra'][i].copy())

    def forward(self, role, i):
        x = self.inputs(role, i)
        if self.method == 'dsnet':
            return self.model(x.permute(2, 0, 1).unsqueeze(0).contiguous())
        # Local protocol explicitly uses scale10 (~5 nodes in49 pixels).
        # This is not historical full-scene S2 or C3's large-tile graph recipe.
        return self.model(self.scene, torch.cat((x, self.coordinates), -1), wavelengths=self.wavelengths,
                          wavelength_unit='nm', superpixel_scale=10, return_graph_info=True)

    def loss(self, i):
        y = torch.tensor([self.data['roles']['train']['labels'][i]], dtype=torch.long)
        out = self.forward('train', i)
        if self.method == 'dsnet':
            x = self.inputs('train', i).permute(2, 0, 1).unsqueeze(0).contiguous()
            return self.adapter.objective(out, x, y)
        from model.task_losses import probability_loss_components
        mask = torch.zeros((49, self.data['classes'])); onehot = torch.zeros_like(mask)
        mask[24] = 1.; onehot[24] = F.one_hot(y, self.data['classes']).float()[0]
        return probability_loss_components(out, onehot, mask, **LOSSES)['total']

    def evaluate(self):
        outputs = []; labels = []
        for i, y in enumerate(self.data['roles']['devval']['labels']):
            out = self.forward('devval', i)
            outputs.append(out[2] if self.method == 'dsnet' else out[0][24:25])
            labels.append(torch.tensor([y], dtype=torch.long))
        return centers_metric(outputs, labels, self.data['classes'], logits=self.method == 'dsnet')


def parameters(task):
    return sum(p.numel() for p in task.model.parameters() if p.requires_grad)
