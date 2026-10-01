"""Reuse selected author functions without importing their image-data pipelines."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import numpy as np
from wisig512_frozen_study_common import module


def author_class(path, name, selected, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if selected is not None:
        node.bases = []
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    unit = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(unit, str(path), 'exec'), namespace)
    return namespace[name]


def acil(root, device, size=8192):
    root = Path(root) / 'ACIL'
    linear = module('author_acil_linear', root / 'AnalyticLinear.py')
    buffer = module('author_acil_buffer', root / 'Buffer.py')
    cls = author_class(root / 'ACIL.py', 'ACIL', None,
                       dict(torch=torch, RandomBuffer=buffer.RandomBuffer,
                            AnalyticLinear=linear.AnalyticLinear, RecursiveLinear=linear.RecursiveLinear))
    return cls(1024, buffer_size=size, gamma=.1, device=device, dtype=torch.double)


def fecam(root):
    cls = author_class(Path(root) / 'FeCAM/base.py', 'BaseLearner',
                       {'shrink_cov', '_tukeys_transform', '_mahalanobis'}, dict(torch=torch, F=F, np=np))
    obj = cls()
    obj.args = dict(alpha1=1., alpha2=1., beta=.5, tukey=False)
    obj._cur_task = 1
    obj._device = torch.device('cpu')
    obj._network = SimpleNamespace(feature_dim=1024)
    return obj


def translated_features(current, current_y, means, old_count):
    """Author compute_distances.py: nearest Euclidean mean and exact translation."""
    xs, ys = [current], [current_y]
    means = np.asarray(means)
    for c in range(old_count):
        d = np.sum((means[old_count:] - means[c]) ** 2, axis=1)
        nearest = old_count + int(d.argmin())
        translated = current[current_y == nearest] - means[nearest] + means[c]
        xs.append(translated)
        ys.append(np.full(len(translated), c, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def normalized_mahalanobis(query, mean, inverse):
    delta = F.normalize(query, dim=1) - F.normalize(mean, dim=0)
    return ((delta @ inverse) * delta).sum(1)
