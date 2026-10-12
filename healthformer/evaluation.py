"""Held-out evaluation and metric tracking."""
from .losses import vocabulary_cross_entropy
from .validation_windows import validation_windows

import numpy as np
import torch
from .tracking import tracker as wandb

METRIC_FIELDS = ('ce_sum', 'ce_count', 'count', 'rejected_ce_count')


class EvaluationMixin:
    @torch.no_grad()
    def evaluate(self):
        """Compute overall and per-modality CE with exact coverage accounting."""
        self.model.eval()

        # Global accumulators
        total_ce_sum = 0.0
        total_ce_count = 0
        total_count = 0
        total_rejected_ce_count = 0

        rank = self.fabric.global_rank

        unique_mods = sorted(self.number_to_modality_name)
        mod_metrics = {m: dict.fromkeys(METRIC_FIELDS, 0) for m in unique_mods}

        self.fabric.barrier()

        coverage = dict(participants=0, eligible_targets=0, truncated_participants=0,
                        oversized_participants=0)
        windows = validation_windows(self.validation_dataloader, self.config, self.mask_modality_val, coverage)
        for batch_idx, (batch, skip_targets) in enumerate(windows):
            tokens, conts, mods, time_exp = batch
            tokens = tokens.to(self.device)
            conts = conts.to(self.device)
            mods = mods.to(self.device)
            time_exp = time_exp.to(self.device)

            if self.config.reduce_context_in_train:
                T_orig = mods.shape[1]
                is_pad_mask = (mods == self.mask_modality_val)
                first_pad_indices = torch.argmax(is_pad_mask.int(), dim=1)
                effective_lengths = torch.where(is_pad_mask.any(dim=1), first_pad_indices, T_orig)
                local_max_len = effective_lengths.max().item()
                if local_max_len < T_orig:
                    tokens, conts, mods, time_exp = (
                        tokens[:, :local_max_len],
                        conts[:, :local_max_len],
                        mods[:, :local_max_len],
                        time_exp[:, :local_max_len]
                    )
                    tokens, conts, mods, time_exp = (
                        tokens.contiguous(), conts.contiguous(), mods.contiguous(), time_exp.contiguous()
                    )

            # For next-token prediction, shift tokens:
            # Input to model: tokens[:, :-1]
            # Targets: tokens[:, 1:]
            tokens_in = tokens[:, :-1].long().contiguous()
            conts_in = conts[:, :-1].contiguous()
            targets = tokens[:, 1:].long().contiguous()

            # Forward pass.
            with self.fabric.autocast():
                logits = self.model(tokens_in, conts_in, mods, time_exp)
            B, T_pred, vocab_size = logits.shape

            # Flatten predictions and targets to 2D vectors:
            logits_2d = logits.reshape(-1, vocab_size)  # [B*(T-1), vocab_size]
            targets_1d = targets.reshape(-1)  # [B*(T-1)]
            mods_1d = mods[:, 1:].reshape(-1)

            # Exclude padding targets.
            valid_mask = (mods_1d != self.mask_modality_val)
            if skip_targets:
                scored = torch.ones_like(targets, dtype=torch.bool)
                scored[:, :skip_targets] = False
                valid_mask &= scored.reshape(-1)
            logits_valid = logits_2d[valid_mask]
            targets_valid = targets_1d[valid_mask]
            mods_valid = mods_1d[valid_mask]

            # Compute per-token cross entropy.
            loss_per_token = vocabulary_cross_entropy(self, logits_valid, targets_valid, mods_valid)
            finite_loss_mask = torch.isfinite(loss_per_token)
            finite_loss_count = int(finite_loss_mask.sum().item())
            valid_target_count = int(logits_valid.shape[0])
            total_ce_sum += loss_per_token[finite_loss_mask].sum().item()
            total_ce_count += finite_loss_count
            total_count += valid_target_count
            total_rejected_ce_count += valid_target_count - finite_loss_count

            mods_np = mods_valid.detach().cpu().numpy()
            loss_np = loss_per_token.detach().cpu().numpy()

            # Per-modality CE and counts
            for m in np.unique(mods_np):
                mask = (mods_np == m)
                finite_mod_mask = mask & np.isfinite(loss_np)
                mod_count = int(mask.sum())
                finite_mod_count = int(finite_mod_mask.sum())
                mod_metrics[m]['ce_sum'] += float(loss_np[finite_mod_mask].sum())
                mod_metrics[m]['ce_count'] += finite_mod_count
                mod_metrics[m]['count'] += mod_count
                mod_metrics[m]['rejected_ce_count'] += mod_count - finite_mod_count
        self.fabric.barrier()

        dev = self.device
        if self.fabric.world_size > 1:

            totals = dict(zip(METRIC_FIELDS,
                          (total_ce_sum, total_ce_count, total_count, total_rejected_ce_count)))
            rows = [totals, *(mod_metrics[m] for m in unique_mods)]
            flat = torch.tensor([[row[field] for field in METRIC_FIELDS] for row in rows],
                                dtype=torch.float64, device=dev)
            self.fabric.barrier()
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
            for row, values in zip(rows, flat.tolist()):
                row.update({field: value if field == 'ce_sum' else int(value)
                            for field, value in zip(METRIC_FIELDS, values)})
            total_ce_sum, total_ce_count, total_count, total_rejected_ce_count = (
                totals[field] for field in METRIC_FIELDS)
        coverage_tensor = torch.tensor(list(coverage.values()), device=self.device, dtype=torch.long)
        self.fabric.all_reduce(coverage_tensor, reduce_op='sum')
        coverage = dict(zip(coverage, map(int, coverage_tensor.tolist())))
        coverage['scored_targets'] = total_count
        coverage['target_fraction'] = total_count / max(1, coverage['eligible_targets'])
        self.last_validation_summary = {
            'valid': False, 'rejected_count': total_rejected_ce_count, **coverage,
            'loss_vocabulary': getattr(self.config, 'loss_vocabulary', 'global'),
        }
        if coverage['oversized_participants']:
            self.model.train()
            raise RuntimeError('Full validation exceeds model capacity; select sliding explicitly. ' + str(coverage))
        if total_rejected_ce_count:
            self.model.train()
            raise RuntimeError(f'Validation has no finite score: rejected {total_rejected_ce_count} non-finite losses.')

        if total_ce_count == 0:
            self.model.train()
            raise RuntimeError("Evaluation found no valid target tokens.")

        summary = {'cross_entropy': total_ce_sum / total_ce_count,
                   'target_count': total_ce_count, 'rejected_count': total_rejected_ce_count,
                   'valid': True, **coverage,
                   'loss_vocabulary': getattr(self.config, 'loss_vocabulary', 'global')}
        self.last_validation_summary = summary
        self.best_val_loss = min(getattr(self, 'best_val_loss', float('inf')), summary['cross_entropy'])
        if rank != 0:
            self.model.train()
            return summary

        overall_avg_ce = summary['cross_entropy']
        print(
            f"Validation Overall cross-entropy: {overall_avg_ce:.4f} "
            f"(finite_count={total_ce_count}, rejected={total_rejected_ce_count})"
        )
        logged = {
            "Validation/CE Overall Loss": overall_avg_ce,
            "Validation/Best CE": self.best_val_loss,
            "Validation/CE Rejected Count": total_rejected_ce_count,
        }

        logged.update({f"Validation/coverage/{key}": value for key, value in coverage.items()})
        print(f"Validation coverage ({getattr(self.config, 'validation_sequence_policy', 'prefix')}): {coverage}")

        # Compute and log per-modality metrics.
        for m in unique_mods:
            info = mod_metrics[m]
            count = info['count']
            if count == 0:
                continue
            ce_count = info['ce_count']
            ce_mod = info['ce_sum'] / ce_count
            mod_name = self.number_to_modality_name[m]
            logged.update({
                f"Validation CE/ {mod_name}": ce_mod,
                f"Validation count/ {mod_name}": count,
                f"Validation CE rejected/ {mod_name}": info['rejected_ce_count'],
            })

        wandb.log(logged)
        self.model.train()
        return summary
