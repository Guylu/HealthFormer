"""Future-visit queries and longitudinal trajectory evaluation."""
from .losses import vocabulary_cross_entropy
from .prompts import prepare_prompt, pad_prompts

from .model import (
    _stable_softmax_over_token_ids,
)
from typing import Any
from typing import Dict
from typing import Optional
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import torch
from tqdm import tqdm
from .tracking import tracker as wandb

SCALAR_FIELDS = ('ce_sum', 'ce_count', 'count', 'correct_top1', 'correct_top5',
                 'correct_top10', 'nan_ce_inputs', 'nan_cont_preds')
CONTINUOUS_FIELDS = ('expected_vals', 'real_vals')


def merge_metrics(destination, source):
    # Retain batch-wise summation and participant order for numerical parity.
    for field, value in source.items():
        destination[field] += value


class LongitudinalMixin:
    def _create_default_metric_dict(self):
        return dict.fromkeys(SCALAR_FIELDS, 0) | {'ce_sum': 0.0, 'expected_vals': [], 'real_vals': []}

    def _prepare_longitudinal_prompt_sample(
            self,
            tokens,
            continuous_tokens,
            modalities,
            time_expanded,
            coverage=None,
    ) -> Optional[Dict[str, Any]]:
        return prepare_prompt(self, tokens.cpu(), continuous_tokens.cpu(), modalities.cpu(),
                              time_expanded.cpu(), evaluation=True, coverage=coverage)

    def _log_longitudinal_metrics(
            self,
            *,
            mode,
            metrics_by_mod,
            total_ce_sum,
            total_ce_count,
            total_count,
            correct_top1,
            correct_top5,
            correct_top10,
    ):
        """Report validated longitudinal results on rank zero."""
        top1_acc = correct_top1 / total_count
        top5_acc = correct_top5 / total_count
        top10_acc = correct_top10 / total_count
        ce_overall = total_ce_sum / total_ce_count
        logged = {
            f"Longitudinal_{mode}_Tops/Top1": top1_acc,
            f"Longitudinal_{mode}_Tops/Top5": top5_acc,
            f"Longitudinal_{mode}_Tops/Top10": top10_acc,
            f"Longitudinal_{mode}_Loss/": ce_overall,
        }
        print(f"[ELIS {mode}] total_count={total_count}, CE={ce_overall:.4f}, "
              f"Top1={top1_acc:.3f}, Top5={top5_acc:.3f}, Top10={top10_acc:.3f}")

        for m_log_key in sorted(metrics_by_mod):
            info = metrics_by_mod[m_log_key]
            c = info["count"]

            mod_name_log = self.number_to_modality_name.get(m_log_key, f"Mod{m_log_key}")
            t1_acc_mod = info["correct_top1"] / c
            t5_acc_mod = info["correct_top5"] / c
            t10_acc_mod = info["correct_top10"] / c

            ce_mod = info["ce_sum"] / info["ce_count"]
            logged.update({
                f"Longitudinal_{mode}_CE/{mod_name_log}": ce_mod,
                f"Longitudinal_{mode}_Top1/{mod_name_log}": t1_acc_mod,
                f"Longitudinal_{mode}_Top5/{mod_name_log}": t5_acc_mod,
                f"Longitudinal_{mode}_Top10/{mod_name_log}": t10_acc_mod,
                f"Longitudinal_{mode}_count/{mod_name_log}": c,
            })
            if info.get("nan_cont_preds", 0) > 0:
                logged[f"Longitudinal_{mode}_NaN_Cont_Preds/{mod_name_log}"] = info["nan_cont_preds"]

            # Real-valued stats for continuous modalities
            if info["expected_vals"]:
                exp_all_np = np.array(info["expected_vals"])
                real_all_np = np.array(info["real_vals"])

                valid_mask_mod = np.isfinite(exp_all_np) & np.isfinite(real_all_np)
                exp_all_valid = exp_all_np[valid_mask_mod]
                real_all_valid = real_all_np[valid_mask_mod]

                if len(exp_all_valid) > 0:  # Need at least one valid point for MSE/MAE
                    mse_mod = np.mean((exp_all_valid - real_all_valid) ** 2)
                    mae_mod = np.mean(np.abs(exp_all_valid - real_all_valid))
                    logged[f"Longitudinal_{mode}_MSE/{mod_name_log}"] = mse_mod
                    logged[f"Longitudinal_{mode}_MAE/{mod_name_log}"] = mae_mod

                    if len(exp_all_valid) > 1:  # Need at least two points for correlation
                        # Check for constant series to avoid Pearson/Spearman warning/error
                        if len(np.unique(real_all_valid)) > 1 and len(np.unique(exp_all_valid)) > 1:
                            # Descriptive event-pooled coefficients; observations within a participant
                            # are dependent, so ordinary correlation p-values are not reported.
                            pear_corr = pearsonr(real_all_valid, exp_all_valid)[0]
                            spear_corr = spearmanr(real_all_valid, exp_all_valid)[0]
                            logged[f"Longitudinal_{mode}_PearsonCorr/{mod_name_log}"] = pear_corr
                            logged[f"Longitudinal_{mode}_SpearmanCorr/{mod_name_log}"] = spear_corr
                    logged[f"Longitudinal_{mode}_ContinuousCount/{mod_name_log}"] = len(exp_all_valid)

        wandb.log(logged)
        return metrics_by_mod

    @torch.no_grad()
    def evaluate_longitudinal(self, loader, mode: str = "Validation"):
        """Evaluate ELIS queries for selected targets in the second marker year.

        Context includes retained history through the first marker year. Query
        rows see that context and themselves; they cannot see future values.
        Metrics are pooled across events and synchronized before reporting.
        """
        self.model.eval()
        device = self.device

        coverage = dict(participants=0, eligible_targets=0, truncated_participants=0)

        totals = dict.fromkeys(SCALAR_FIELDS[:-1], 0)
        totals['ce_sum'] = 0.0
        metrics_by_mod = defaultdict(self._create_default_metric_dict)

        for batch_idx, batch_data_tuple in enumerate(
                tqdm(loader, desc=f"ELIS {mode} Rank {self.fabric.global_rank}")):
            try:
                tokens_batch_orig, conts_batch_orig, mods_batch_orig, time_batch_orig = batch_data_tuple

                samples = []
                for i in range(tokens_batch_orig.shape[0]):
                    sample = self._prepare_longitudinal_prompt_sample(
                        tokens_batch_orig[i], conts_batch_orig[i], mods_batch_orig[i],
                        time_batch_orig[i], coverage=coverage)
                    if sample is not None:
                        samples.append(sample)
                if not samples:
                    continue
                tokens, continuous, modalities, times, mask = pad_prompts(
                    samples, self.mask_modality_val)
                with self.fabric.autocast():
                    logits_batch = self.model(tokens, continuous, modalities, times, mask=mask)

                batch_totals = dict.fromkeys(SCALAR_FIELDS[:-1], 0)
                batch_totals['ce_sum'] = 0.0
                batch_metrics = defaultdict(self._create_default_metric_dict)

                for sample_index, sample in enumerate(samples):
                    target_tokens = sample['target_tokens']
                    target_continuous = sample['target_continuous']
                    target_modalities = sample['target_modalities']
                    for target_index, query_index in enumerate(sample['target_logit_indices']):
                        target_logits = logits_batch[sample_index, query_index, :]  # [vocab_size]
                        target_token = target_tokens[target_index]
                        target_value = target_continuous[target_index]
                        target_modality = target_modalities[target_index]

                        current_mod_num = target_modality.item()
                        metrics = batch_metrics[current_mod_num]

                        # Cross-entropy
                        ce_loss_token = vocabulary_cross_entropy(
                            self, target_logits.unsqueeze(0), target_token.unsqueeze(0).long(),
                            target_modality.unsqueeze(0)).mean()
                        if torch.isfinite(ce_loss_token):
                            ce_loss_value = ce_loss_token.item()
                            batch_totals['ce_sum'] += ce_loss_value
                            batch_totals['ce_count'] += 1
                            metrics["ce_sum"] += ce_loss_value
                            metrics["ce_count"] += 1
                        else:
                            batch_totals['nan_ce_inputs'] += 1
                            metrics["nan_ce_inputs"] += 1

                        batch_totals['count'] += 1
                        metrics["count"] += 1

                        # Top-K Accuracy (on single logit vector)
                        pred_token_id_top1 = target_logits.argmax().item()
                        if pred_token_id_top1 == target_token.item():
                            batch_totals['correct_top1'] += 1
                            metrics["correct_top1"] += 1

                        _, topk_indices = torch.topk(target_logits, 10)
                        topk_indices_list = topk_indices.tolist()

                        if target_token.item() in topk_indices_list[:5]:
                            batch_totals['correct_top5'] += 1
                            metrics["correct_top5"] += 1
                        if target_token.item() in topk_indices_list:  # Covers top 10
                            batch_totals['correct_top10'] += 1
                            metrics["correct_top10"] += 1

                        # Continuous value prediction
                        mod_name_actual = self.number_to_modality_name.get(current_mod_num)
                        if mod_name_actual and mod_name_actual in self.mod2bininfo:
                            token_indices_np, bin_midpoints_np = self.mod2bininfo[mod_name_actual]
                            if len(token_indices_np) > 0 and len(bin_midpoints_np) > 0:
                                probs_bins = _stable_softmax_over_token_ids(target_logits, token_indices_np)
                                if probs_bins.numel() > 0 and len(probs_bins) == len(bin_midpoints_np):
                                    pred_cont_val = (probs_bins * torch.from_numpy(bin_midpoints_np).to(device=device,
                                                                                                        dtype=probs_bins.dtype)).sum().item()
                                    pred_cont_val = float(pred_cont_val)
                                    target_value = np.float64(target_value.item())
                                    if np.isfinite(pred_cont_val) and np.isfinite(target_value):
                                        metrics["expected_vals"].append(
                                            pred_cont_val)
                                        metrics["real_vals"].append(
                                            target_value.item())
                                    else:
                                        metrics["nan_cont_preds"] += 1

                merge_metrics(totals, batch_totals)
                for modality, metrics in batch_metrics.items():
                    merge_metrics(metrics_by_mod[modality], metrics)

            except Exception as e:
                raise RuntimeError(
                    f"ELIS validation failed at batch {batch_idx} "
                    f"on rank {self.fabric.global_rank}."
                ) from e


        if self.fabric.world_size > 1:
            rank = self.fabric.global_rank

            for field in totals:
                value = torch.tensor(totals[field], device=device,
                                     dtype=torch.float64 if field == 'ce_sum' else torch.long)
                self.fabric.all_reduce(value, reduce_op='sum')
                totals[field] = value.item()

            local_mod_keys = list(metrics_by_mod.keys())
            gathered_keys_list = [None] * self.fabric.world_size
            torch.distributed.all_gather_object(gathered_keys_list, local_mod_keys)

            sorted_mod_keys = sorted({key for keys in gathered_keys_list for key in keys})
            n_mods = len(sorted_mod_keys)
            all_mod_metrics = torch.zeros(n_mods, len(SCALAR_FIELDS), dtype=torch.float64, device=device)
            for i, m_key in enumerate(sorted_mod_keys):
                for j, sf in enumerate(SCALAR_FIELDS):
                    all_mod_metrics[i, j] = metrics_by_mod[m_key][sf]

            self.fabric.all_reduce(all_mod_metrics, reduce_op="sum")

            for i, m_key in enumerate(sorted_mod_keys):
                for j, sf in enumerate(SCALAR_FIELDS):
                    val = all_mod_metrics[i, j].item()
                    metrics_by_mod[m_key][sf] = val if sf == "ce_sum" else int(val)

            local_continuous = {m: {key: metrics_by_mod[m][key] for key in CONTINUOUS_FIELDS}
                                for m in sorted_mod_keys}
            gathered_continuous = [None] * self.fabric.world_size
            torch.distributed.all_gather_object(gathered_continuous, local_continuous)
            if rank == 0:
                for m in sorted_mod_keys:
                    for key in CONTINUOUS_FIELDS:
                        metrics_by_mod[m][key] = []
                for rank_data in gathered_continuous:
                    for m, metrics in rank_data.items():
                        merge_metrics(metrics_by_mod[m], metrics)

        coverage_tensor = torch.tensor(list(coverage.values()), dtype=torch.long, device=device)
        self.fabric.all_reduce(coverage_tensor, reduce_op="sum")
        coverage = dict(zip(coverage, map(int, coverage_tensor.tolist())))
        self.last_longitudinal_summary = {
            **coverage, 'scored_targets': totals['count'],
            'target_fraction': totals['count'] / max(1, coverage['eligible_targets']),
            'valid': totals['count'] > 0 and totals['nan_ce_inputs'] == 0,
        }
        if self.fabric.global_rank == 0:
            print(f"Longitudinal coverage: {self.last_longitudinal_summary}")
            wandb.log({f"Longitudinal_{mode}_coverage/{k}": v
                       for k, v in self.last_longitudinal_summary.items()})

        if totals['count'] == 0:
            self.model.train()
            raise RuntimeError('Longitudinal validation found no eligible targets within the context limit.')
        if totals['nan_ce_inputs']:
            self.model.train()
            raise RuntimeError(f"Longitudinal validation rejected {totals['nan_ce_inputs']} non-finite losses.")


        final_metrics_to_return = {}
        if self.fabric.global_rank == 0:
            final_metrics_to_return = self._log_longitudinal_metrics(
                mode=mode,
                metrics_by_mod=metrics_by_mod,
                total_ce_sum=totals['ce_sum'],
                total_ce_count=totals['ce_count'],
                total_count=totals['count'],
                correct_top1=totals['correct_top1'],
                correct_top5=totals['correct_top5'],
                correct_top10=totals['correct_top10'],
            )


        self.fabric.barrier()  # Ensure all processes finish before model mode changes
        self.model.train()
        return final_metrics_to_return if self.fabric.global_rank == 0 else {}
