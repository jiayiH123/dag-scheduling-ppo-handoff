"""Optional Torch batch/loss helpers; no language-model weights or architecture."""
import torch
from .structured_dataset import numeric_view


def numeric_batch(rows, device='cpu'):
    views = [numeric_view(r['input']) for r in rows]
    size = max(v['position_count'] for v in views)
    types = torch.full((len(rows), size), 7, dtype=torch.long, device=device)
    ids = torch.full((len(rows), size, 2), -1, dtype=torch.long, device=device)
    values = torch.zeros((len(rows), size, 6), dtype=torch.float32, device=device)
    mask = torch.zeros((len(rows), size), dtype=torch.bool, device=device)
    for b, v in enumerate(views):
        length = v['position_count']; types[b, :length] = torch.tensor(v['types'], device=device)
        ids[b, :length] = torch.tensor(v['ids'], device=device)
        for j, payload in enumerate(v['payloads']):
            if payload: values[b, j, :len(payload)] = torch.tensor(payload, device=device)
        mask[b, :length] = True
    return dict(token_types=types, id_pairs=ids, numeric_features=values,
                attention_mask=mask, query_positions=[v['query_positions'] for v in views],
                prefix_lengths=[v['prefix_length'] for v in views], output_shapes=[v['output_shape'] for v in views])


def bc_loss(pair_coefficients, supervision):
    """Supervise pool choices under teacher prefixes. Coefficients are NOT GT."""
    from .module_lab.models import conditionals
    logs, _ = conditionals(pair_coefficients.reshape(-1, 2), supervision['decisions'])
    return -logs.mean()
