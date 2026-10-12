"""Aligned ELIS sequences shared by training and longitudinal validation."""
import torch
from torch.nn.utils.rnn import pad_sequence
from .visits import identify_visits, elis_modalities

CHANNELS = ('tokens', 'continuous_tokens', 'modalities', 'time_expanded')


def prepare_prompt(owner, tokens, continuous_tokens, modalities, time_expanded,
                   *, evaluation=False, coverage=None):
    if coverage is not None:
        coverage['participants'] += 1
    keep = modalities != owner.mask_modality_val
    channels = [value[keep] for value in (tokens, continuous_tokens, modalities, time_expanded)]
    if len(channels[0]) < 2:
        return None
    visits, _ = identify_visits(channels[2], channels[3], owner.visit_marking, owner.year_idx)
    if visits is None:
        return None
    first_end, second_start, second_end, _ = visits
    selected_mods = elis_modalities(owner.modality_name_to_number, evaluation=evaluation)
    future_mods = channels[2][second_start:second_end]
    selected = torch.isin(future_mods, torch.tensor(sorted(selected_mods),
                          dtype=future_mods.dtype, device=future_mods.device))
    eligible = int(selected.sum())
    count = min(eligible, max(0, (owner.config.max_seq_length - first_end + 1) // 2))
    if coverage is not None:
        coverage['eligible_targets'] += eligible
        coverage['truncated_participants'] += int(count < eligible)
    if count == 0:
        return None

    sequences, targets = [], []
    for channel in channels:
        prefix = channel[:first_end]
        future = channel[second_start:second_end][selected][:count]
        parts = [prefix]
        for i in range(count):
            parts.append(future[i:i + 1])
            if i < count - 1:
                parts.append(prefix[-1:])
        sequences.append(torch.cat(parts).to(owner.device))
        targets.append(future.to(owner.device))

    length = len(sequences[0]) - 1
    queries = list(range(first_end - 1, length, 2))
    # Both branches use the same mask. Future-value rows are self-only;
    # query rows cannot read them, including indirectly through another layer.
    mask = torch.eye(length, dtype=torch.bool, device=owner.device)
    mask[:first_end, :first_end] = torch.ones(first_end, first_end, dtype=torch.bool,
                                             device=owner.device).tril()
    for query in queries:
        mask[query, :first_end] = True
        mask[query, first_end:query] = False
        mask[query, query] = True
    return dict(zip(CHANNELS, sequences), attention_mask=mask,
                target_logit_indices=queries, target_tokens=targets[0],
                target_continuous=targets[1], target_modalities=targets[2])


def pad_prompts(samples, mask_modality_val):
    padded = []
    for key, padding in zip(CHANNELS, (0, 0.0, mask_modality_val, 0)):
        values = [sample[key] for sample in samples]
        if key in CHANNELS[:2]:
            values = [value[:-1] for value in values]
        if key in ('tokens', 'modalities'):
            values = [value.long() for value in values]
        value = pad_sequence(values, batch_first=True, padding_value=padding)
        padded.append(value)
    length = padded[0].shape[1]
    masks = [torch.nn.functional.pad(sample['attention_mask'],
             (0, length - len(sample['attention_mask']), 0, length - len(sample['attention_mask'])))
             for sample in samples]
    mask = torch.stack(masks).unsqueeze(1)
    batch, head, query = (~mask.any(dim=-1)).nonzero(as_tuple=True)
    mask[batch, head, query, query] = True
    return (*padded, mask)
