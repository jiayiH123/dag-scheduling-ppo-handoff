"""Physical-configuration view of the immutable structured-v1 raw records.

The generator draws one platform per base DAG and two budget conditions. Those
conditions can have equal integer limits. Keep the first serialized row for each
(base_id, budget limit), without comparing teacher quality or changing labels.
"""
from .structured_dataset import iter_rows


def configuration_key(row):
    return row['base_id'], row['input']['budget']['limit']


def iter_unique_rows(root, split=None, limit=None):
    """Stream distinct physical configurations; limit applies AFTER deduplication.

    Valid for this dataset's verified one-platform-per-base contract. Raw shards,
    split assignment and labels remain immutable. Use the same view for all
    architectures/baselines and retain grouping by base_id when subsampling.
    """
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive or None')
    seen = set()
    count = 0
    for row in iter_rows(root, split):
        key = configuration_key(row)
        if key in seen:
            continue
        seen.add(key)
        yield row
        count += 1
        if limit is not None and count >= limit:
            return
