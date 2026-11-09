"""Sparse candidate edge summaries, O(nodes+edges) storage and weak connectivity."""
import numpy as np

from .classification_metrics import integer_values


def graph_diagnostics(node_count, edge_indices, weights_before, weights_after=None):
    if type(node_count) is not int or node_count < 0:
        raise ValueError('node_count must be a nonnegative integer')
    indices = integer_values(edge_indices, 'Edge indices')
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError('edge_indices must be [2,E]')
    if indices.size and (indices >= node_count).any():
        raise ValueError('Edge endpoint outside node_count')
    if np.unique(indices, axis=1).shape[1] != indices.shape[1]:
        raise ValueError('Coalesce duplicate directed edges before diagnostics')
    def weights(value):
        value = np.asarray(value)
        if value.shape != (indices.shape[1],) or value.dtype.kind not in 'iuf' or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('Edge weights must be a matching finite nonnegative vector')
        return value.astype(float, copy=False)
    before = weights(weights_before)
    after = weights(weights_after) if weights_after is not None else before
    positive = after > 0
    rows, cols = indices[:, positive]
    degree = np.bincount(rows, minlength=node_count)
    mass = np.bincount(rows, weights=after[positive], minlength=node_count)
    nonself = rows != cols
    nonself_degree = np.bincount(np.concatenate((rows[nonself], cols[nonself])), minlength=node_count)
    parent = np.arange(node_count)
    size = np.ones(node_count, dtype=np.int64)
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for source, target in zip(rows[nonself], cols[nonself]):
        a, b = find(int(source)), find(int(target))
        if a != b:
            if size[a] < size[b]: a, b = b, a
            parent[b] = a; size[a] += size[b]
    roots = np.asarray([find(i) for i in range(node_count)], dtype=np.int64)
    components = np.bincount(roots, minlength=node_count)
    delta = np.abs(after - before)
    return dict(node_count=node_count, stored_edge_count=indices.shape[1], positive_edge_count=int(positive.sum()),
                self_loop_count=int((rows == cols).sum()), max_out_degree=int(degree.max()) if node_count else None,
                mean_out_degree=float(degree.mean()) if node_count else None,
                isolated_without_nonself_neighbors=int((nonself_degree == 0).sum()),
                weak_component_count=int((components > 0).sum()),
                largest_weak_component=int(components.max()) if node_count else 0,
                max_row_mass_error=float(np.abs(mass - 1).max()) if node_count else None,
                weight_change_mean=float(delta.mean()) if weights_after is not None and delta.size else None,
                weight_change_max=float(delta.max()) if weights_after is not None and delta.size else None,
                support_change_count=int(((before > 0) != (after > 0)).sum()) if weights_after is not None else None,
                comparison_available=weights_after is not None, connectivity='weak_undirected_projection')
