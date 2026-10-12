"""Validation sequence policies; windows use available preceding context only."""
import torch

def validation_windows(loader, config, padding, coverage):
    policy = getattr(config, 'validation_sequence_policy', 'prefix')
    limit = getattr(config, 'validation_max_tokens', 10000)
    capacity = getattr(config, 'max_seq_length', limit)
    if policy not in ('prefix', 'full', 'sliding') or min(limit, capacity) < 2:
        raise ValueError('Validation requires a supported policy and window capacity >= 2.')
    limit = min(limit, capacity)
    max_batches = getattr(config, 'max_validation_batches', None)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        tokens, conts, mods, times = batch
        valid = mods != padding
        lengths = [int(torch.where(row)[0][-1]) + 1 if row.any() else 0 for row in valid]
        coverage['participants'] += len(lengths)
        coverage['eligible_targets'] += int(valid[:, 1:].sum())
        if policy == 'prefix':
            coverage['truncated_participants'] += sum(n > limit for n in lengths)
            if min(max(lengths, default=0), limit) < 2:
                continue
            yield (tokens[:, :limit], conts[:, :limit], mods[:, :limit], times[:, :limit]), 0
            continue
        for i, length in enumerate(lengths):
            if length < 2:
                continue
            if policy == 'full' and length > capacity:
                coverage['oversized_participants'] += 1
                continue
            first_unscored = 1
            while first_unscored < length:
                window_limit = capacity if policy == 'full' else limit
                start = max(0, first_unscored - max(1, window_limit // 2))
                end = min(length, start + window_limit)
                skip = first_unscored - start - 1
                yield (tokens[i:i+1, start:end], conts[i:i+1, start:end],
                       mods[i:i+1, start:end], times[i:i+1, start:end]), max(0, skip)
                first_unscored = end
