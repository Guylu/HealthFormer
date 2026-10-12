"""Explicit vocabulary normalization shared by training and validation."""
import torch
import torch.nn.functional as F

def vocabulary_cross_entropy(owner, logits, targets, modalities, soft_targets=None):
    if getattr(owner.config, 'loss_vocabulary', 'global') == 'global':
        if soft_targets is None:
            return F.cross_entropy(logits, targets, reduction='none')
        return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    loss_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    result = torch.zeros(targets.shape, device=logits.device, dtype=loss_dtype)
    covered = torch.zeros_like(targets, dtype=torch.bool)
    for name, values in owner.vals_per_modality.items():
        mod = owner.modality_name_to_number[name]
        rows = modalities == mod
        if not rows.any():
            continue
        ids = torch.tensor(sorted(v + owner.cum_base_value_per_modality[mod] for v in values),
                           device=logits.device, dtype=torch.long)
        selected = logits[rows].index_select(-1, ids).to(loss_dtype)
        matches = targets[rows, None] == ids[None, :]
        if not bool(matches.any(dim=-1).all()):
            raise ValueError(f'Target is outside declared vocabulary for {name}.')
        local_targets = matches.long().argmax(dim=-1)
        if soft_targets is None:
            result[rows] = F.cross_entropy(selected, local_targets, reduction='none')
        else:
            weights = soft_targets[rows].index_select(-1, ids)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            result[rows] = -(weights * F.log_softmax(selected, dim=-1)).sum(dim=-1)
        covered |= rows
    padding = modalities == owner.mask_modality_val
    if not bool((covered | padding).all()):
        raise ValueError('Unknown target modality in vocabulary loss.')
    return result
