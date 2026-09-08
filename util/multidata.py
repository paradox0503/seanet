"""Read headerless float32 datasets and append zeros for the 256-D baseline."""

from pathlib import Path

import numpy as np


def _positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError('%s must be a positive integer' % field)
    return value


def _record_count(path, dim, requested=None):
    if not path or not Path(path).is_file():
        raise ValueError('Dataset file does not exist: %s' % path)
    byte_size = Path(path).stat().st_size
    if byte_size == 0 or byte_size % (4 * dim):
        raise ValueError('%s is not a whole number of float32 records of dimension %d' % (path, dim))
    available = byte_size // (4 * dim)
    count = available if requested is None else _positive_int(requested, 'record count')
    if count > available:
        raise ValueError('%s contains %d records, requested %d' % (path, available, count))
    return count


def prepare_datasets(datasets, size_train, size_val, to_embed):
    """Validate explicit source dimensions; file size alone cannot identify them."""
    if not isinstance(datasets, list) or not datasets:
        raise ValueError('datasets must be a nonempty list')
    prepared = []
    names = set()
    for index, source in enumerate(datasets):
        entry = dict(source)
        dim = _positive_int(entry.get('dim_series'), 'dataset dim_series')
        if dim > 256:
            raise ValueError('dataset dim_series must not exceed 256 (no truncation)')
        name = entry.setdefault('dataset_name', 'dataset_%d' % index)
        if not isinstance(name, str) or not name or name in names or any(c in name for c in '/\\:') or name in ('.', '..'):
            raise ValueError('dataset_name must be a unique filename-safe name')
        names.add(name)
        entry['size_db'] = _record_count(entry.get('database_path'), dim, entry.get('size_db'))
        for field, default in [('size_train', size_train), ('size_val', size_val)]:
            entry[field] = _positive_int(entry.get(field, default), field)
        if to_embed:
            entry['size_query'] = _record_count(entry.get('query_path'), dim, entry.get('size_query'))
        prepared.append(entry)
    return prepared


def sample_datasets(datasets, seed):
    """Uniform sampling with replacement, matching the existing baseline sampler.

    Only sampled rows are materialized. Original files and their values are unchanged.
    """
    rng = np.random.RandomState(seed)
    splits = [[], []]
    for entry in datasets:
        dim = entry['dim_series']
        records = np.memmap(entry['database_path'], mode='r', dtype=np.float32,
                            shape=(entry['size_db'], dim))
        try:
            for split, field in zip(splits, ['size_train', 'size_val']):
                indices = rng.randint(0, entry['size_db'], size=entry[field])
                rows = records[indices]
                if not np.isfinite(rows).all():
                    raise ValueError('Non-finite sampled values in %s' % entry['database_path'])
                padded = np.zeros((len(rows), 1, 256), dtype=np.float32)
                padded[:, 0, :dim] = rows
                split.append(padded)
        finally:
            del records
    return tuple(np.concatenate(split, axis=0) for split in splits)
