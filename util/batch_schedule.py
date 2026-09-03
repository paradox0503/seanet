"""Index-only training schedules; randomness stays with the PyTorch caller."""


def batches_per_epoch(total_num_batches, divisor=10):
    """Return the fixed optimizer-update budget used by both schedules."""
    if type(total_num_batches) is not int or total_num_batches <= 0:
        raise ValueError('total_num_batches must be a positive integer')
    if type(divisor) is not int or divisor <= 0:
        raise ValueError('divisor must be a positive integer')
    return max(1, total_num_batches // divisor)


def candidate_batch_indices(schedule, ordered_indices, epoch, curriculum_epochs=80, stages=8):
    """Return eligible original batch IDs, preserving the supplied order.

    Curriculum callers supply loss-ranked IDs; baseline callers supply all IDs
    in original order. Epoch is zero-based, before the training loop increments it.
    """
    if schedule not in ('loss_curriculum', 'random_baseline'):
        raise ValueError('unknown training schedule: {}'.format(schedule))
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be a nonnegative integer')
    if (type(curriculum_epochs) is not int or curriculum_epochs <= 0
            or type(stages) is not int or stages <= 0
            or curriculum_epochs % stages != 0):
        raise ValueError('curriculum_epochs must be positive and divisible by positive stages')
    indices = list(ordered_indices)
    if not indices:
        raise ValueError('cannot select from an empty training set')
    if schedule == 'random_baseline' or epoch >= curriculum_epochs:
        return indices

    stage = min(epoch // (curriculum_epochs // stages), stages - 1)
    pool_end = (len(indices) * (stage + 1) + stages - 1) // stages
    return indices[:pool_end]


def select_batch_indices(candidate_indices, batch_count, shuffled_positions):
    """Map a random permutation (or its prefix) back to unique original IDs.

    Supply at least batch_count unique positions when the pool exceeds budget.
    No permutation is needed when the whole pool fits in the budget.
    """
    candidates = list(candidate_indices)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('candidate batch IDs must be nonempty and unique')
    if type(batch_count) is not int or batch_count <= 0:
        raise ValueError('batch_count must be a positive integer')
    if len(candidates) <= batch_count:
        return candidates

    positions = list(shuffled_positions)
    if len(positions) < batch_count:
        raise ValueError('not enough shuffled positions for the batch budget')
    if any(type(position) is not int or position < 0 or position >= len(candidates)
           for position in positions):
        raise ValueError('shuffled positions must be integer indices inside the candidate pool')
    if len(set(positions)) != len(positions):
        raise ValueError('shuffled positions must be unique for sampling without replacement')
    return [candidates[position] for position in positions[:batch_count]]
