"""Training objectives, gradient accumulation, and optimizer updates."""
from .prompts import prepare_prompt, pad_prompts

import math
import time
import torch
from .tracking import tracker as wandb

class OptimizationMixin:
    def get_lr(self, it):
        # Linear warmup.
        if it < self.warmup_steps:
            return self.max_lr * it / self.warmup_steps
        # A run whose last update is also its warmup endpoint has no cosine span.
        if self.max_steps == self.warmup_steps:
            return self.max_lr if it <= self.max_steps else self.min_lr
        # Hold the minimum rate after the planned updates.
        if it > self.max_steps:
            return self.min_lr
        # Cosine decay after warmup.
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def _normalize_accumulated_gradients(self, accumulated_microbatches: int):
        """Convert accumulated gradient sums to a mean before clipping."""
        accumulated_microbatches = self._validate_accumulate_grad_batches(accumulated_microbatches)
        if accumulated_microbatches == 1:
            return
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_microbatches)

    def _reset_pending_window_state(self, metric_state=None):
        """Reset only the state owned by the current accumulation window."""
        self.pending_microbatches = 0
        self.pending_window_loss = 0.0
        if metric_state is not None:
            metric_state.clear()

    def _log_completed_training_window(self, metric_state, loader_len, window_size, mean_weighted_loss, update_metrics):
        report_progress = self.it % self.config.log_wandb_metric_every_k == 0
        if not (self.config.allow_wandb or report_progress):
            metric_state.clear()
            return
        metrics = {"Training Loss/": mean_weighted_loss}
        metrics.update({f"Training Losses/{name}": value / window_size
                        for name, value in metric_state.items()})
        if self.fabric.world_size > 1:
            # Optional loss terms can be absent on individual workers. Average
            # their contributions over all workers, matching DDP's gradient mean.
            gathered = [None] * self.fabric.world_size
            torch.distributed.all_gather_object(gathered, metrics)
            names = sorted({name for worker in gathered for name in worker})
            metrics = {name: sum(worker.get(name, 0.0) for worker in gathered) / len(gathered)
                       for name in names}
        metrics['Training Accumulation/Window Size'] = window_size
        metrics.update(update_metrics)
        if self.fabric.global_rank == 0:
            if report_progress:
                print(f"Epoch {self.epoch + 1}, update {self.it}, batch {self.training_step + 1}/{loader_len}, "
                      f"training loss {metrics['Training Loss/']:.6f}", flush=True)
            wandb.log(metrics)
        metric_state.clear()

    def step_optimizer(self, accumulated_microbatches=1):
        self._normalize_accumulated_gradients(accumulated_microbatches)
        next_it = self.it + 1

        norm = self.fabric.clip_gradients(self.model, self.optimizer, max_norm=self.config.max_grad_norm, norm_type=2)
        lr = self.get_lr(next_it)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
        self.it = next_it
        self.optimizer.zero_grad(set_to_none=True)
        return {"misc/Gradient Norm": float(norm), "misc/Iteration": self.it, "misc/Learning rate": lr}

    def compute_standard_loss(
            self,
            tokens_batch_input: torch.Tensor,
            continuous_batch: torch.Tensor,
            targets_batch: torch.Tensor,
            modalities_batch_full: torch.Tensor,
            time_expanded_batch_full: torch.Tensor,
    ):
        """Return the standard training objective and its scalar components."""
        with self.fabric.autocast():
            # Modality and time channels include the target; the model aligns them.
            logits_standard = self.model(tokens_batch_input, continuous_batch,
                                         modalities_batch_full, time_expanded_batch_full)
            return self.model.calculate_loss(
                logits_standard, continuous_batch, targets_batch, modalities_batch_full,
                mode='Tr_NTP')

    def _all_min_bool(self, flag: bool) -> bool:
        """True only if ALL ranks passed True."""
        t = torch.tensor(1 if flag else 0, device=self.device)
        self.fabric.all_reduce(t, reduce_op="min")
        return bool(t.item())

    def compute_elis_loss(
            self,
            tokens_batch_input: torch.Tensor,
            continuous_batch: torch.Tensor,
            targets_batch: torch.Tensor,
            modalities_batch_full: torch.Tensor,
            time_expanded_batch_full: torch.Tensor,
    ):
        """
        Bounded ELIS augmentation.
        Collective eligibility reductions keep the workers on the same branch.
        Before any early exit, it checks with all other ranks.
        If any rank cannot proceed, all ranks will exit the function synchronously.
        """
        B_orig = tokens_batch_input.size(0)

        at_least_two_visits = False
        for b_idx in range(B_orig):
            if (modalities_batch_full[b_idx] == self.visit_marking).sum() >= 2:
                at_least_two_visits = True
                break

        do_elis_local = at_least_two_visits and tokens_batch_input.size(1) <= self.config.elis_max_input_length

        # A rank with no eligible sample makes every rank skip ELIS.
        do_elis_globally = self._all_min_bool(do_elis_local)

        if not do_elis_globally:
            return None, False

        with self.fabric.autocast():
            samples = []
            for i in range(B_orig):
                tokens = torch.cat((tokens_batch_input[i], targets_batch[i, -1:]))
                sample = prepare_prompt(self, tokens, continuous_batch[i],
                                        modalities_batch_full[i], time_expanded_batch_full[i])
                if sample is not None:
                    samples.append(sample)
            # Every rank must have a prompt before any model forward/backward.
            if not self._all_min_bool(bool(samples)):
                return None, False
            tokens, continuous, modalities, times, mask = pad_prompts(samples, self.mask_modality_val)
            logits = self.model(tokens, continuous, modalities, times, mask=mask)
            selected_logits = [logits[i, sample['target_logit_indices']]
                               for i, sample in enumerate(samples)]
            # Prompt construction guarantees one query for each retained target.
            logits_final = torch.cat(selected_logits)
            targets, continuous, modalities = [torch.cat([sample[key] for sample in samples])
                for key in ('target_tokens', 'target_continuous', 'target_modalities')]
            loss, _ = self.model.calculate_loss(logits_final, continuous, targets, modalities,
                                                       mode='Tr_ElisAug')
        return loss, True

    def train_epoch(self):
        def _reduce_any(flag_bool: bool) -> bool:
            """All-reduce MAX over a bool; returns True if any rank is True."""
            t = torch.tensor([1 if flag_bool else 0], device=self.device)
            self.fabric.all_reduce(t, reduce_op="max")
            return bool(t.item())

        train_cursor = self._active_train_cursor()
        train_cursor.set_epoch(self.epoch)
        self.current_batch_index = train_cursor.next_batch_index

        time_start_epoch = time.time()
        wandb.log({"Epoch": self.epoch + 1})

        if self.config.evaluate:
            self.evaluate()
        if getattr(self.config, 'longitudinal_evaluation', False):
            self.evaluate_longitudinal(
                self.validation_dataloader_with_morethan1_visit, 'Validation')

        if not self._accumulation_boundary_is_empty():
            raise RuntimeError("train_epoch must start at an empty gradient accumulation boundary.")
        updates_at_start = self.it
        metric_state = {}

        loader = self._active_train_dataloader()
        data_iter = iter(loader)
        loader_len = train_cursor.total_batches
        expected_remaining_batches = loader_len - train_cursor.next_batch_index
        if len(loader) != expected_remaining_batches:
            raise RuntimeError(
                f"Training loader exposes {len(loader)} remaining batches; expected {expected_remaining_batches}."
            )


        for self.training_step, (tokens, continuous_tokens, modalities, time_expanded) in (
                enumerate(data_iter, start=train_cursor.next_batch_index)
        ):


            if _reduce_any(not bool((modalities[:, 1:] != self.mask_modality_val).any())):
                raise RuntimeError('Training batch has no prediction targets on at least one rank.')

            if self.config.reduce_context_in_train:
                T_orig = modalities.shape[1]
                is_pad_mask = (modalities == self.mask_modality_val)
                first_pad_indices = torch.argmax(is_pad_mask.int(), dim=1)
                effective_lengths = torch.where(is_pad_mask.any(dim=1), first_pad_indices, T_orig)
                local_max_len = effective_lengths.max()
                self.fabric.all_reduce(local_max_len, reduce_op="max")
                global_max_len = int(local_max_len.item())
                tokens, continuous_tokens, modalities, time_expanded = (
                    tokens[:, :global_max_len],
                    continuous_tokens[:, :global_max_len],
                    modalities[:, :global_max_len],
                    time_expanded[:, :global_max_len],
                )
                tokens, continuous_tokens, modalities, time_expanded = (
                    tokens.contiguous(), continuous_tokens.contiguous(),
                    modalities.contiguous(), time_expanded.contiguous()
                )

            # Artifact validation guarantees that this removes padding only.
            limit = self.config.max_seq_length
            tokens, continuous_tokens, modalities, time_expanded = (
                tokens[:, :limit], continuous_tokens[:, :limit],
                modalities[:, :limit], time_expanded[:, :limit])

            tokens = tokens.to(self.device)
            modalities = modalities.to(self.device)
            time_expanded = time_expanded.to(self.device)

            targets = tokens[:, 1:].long().contiguous()
            tokens_in = tokens[:, :-1].long().contiguous()

            if not 0 <= self.pending_microbatches < self.config.accumulate_grad_batches:
                raise RuntimeError(
                    "Invalid pending microbatch count: "
                    f"{self.pending_microbatches} for accumulation "
                    f"{self.config.accumulate_grad_batches}."
                )
            final_physical_batch = self.training_step == loader_len - 1
            will_step_if_accepted = (
                self.pending_microbatches + 1 == self.config.accumulate_grad_batches
                or final_physical_batch
            )

            rand_pair = torch.zeros(2, device=self.device)
            if self.fabric.global_rank == 0:
                rand_pair = torch.rand(2, device=self.device)
            rand_pair = self.fabric.all_reduce(rand_pair, reduce_op='sum')
            want_elis = (rand_pair[0].item() < float(self.config.Elis_augmentation_prob))

            elis_loss = None
            elis_local_can_run = False
            elis_local_nonfinite = False
            elis_should_backward = False
            elis_loss_value = None

            # DDP reducer state is established during forward, so the forward and
            # backward must live in the same no-sync context.
            with self.fabric.no_backward_sync(self.model, enabled=True):
                # Even if disabled, every rank performs the same reductions in the same order.
                if want_elis and float(self.config.Elis_augmentation_prob) > 0 \
                        and float(self.config.elis_loss_weight) > 0:
                    elis_loss, elis_local_can_run = self.compute_elis_loss(
                        tokens_in, continuous_tokens, targets, modalities, time_expanded,
                    )
                    if elis_loss is None:
                        elis_local_can_run = False
                    elif not torch.isfinite(elis_loss):
                        elis_local_nonfinite = True
                        elis_local_can_run = False


                elis_nonfinite = _reduce_any(elis_local_nonfinite)
                if not elis_nonfinite:
                    run_elis_everywhere = self._all_min_bool(elis_local_can_run)
                    elis_should_backward = run_elis_everywhere and elis_loss is not None
                    if elis_should_backward:
                        elis_loss_value = float(elis_loss.detach().item())
                        self.fabric.backward(float(self.config.elis_loss_weight) * elis_loss)

            if elis_nonfinite:
                self.optimizer.zero_grad(set_to_none=True)
                self._reset_pending_window_state(metric_state)
                self._mark_active_train_batch_consumed(self.training_step + 1)
                print(
                    f"[rank {self.fabric.global_rank}] batch {self.training_step + 1}: "
                    "Non-finite ELIS loss; discarded accumulation window.",
                    flush=True,
                )
                wandb.log({"nonfinite_loss/ELIS": 1})
                continue

            with self.fabric.no_backward_sync(self.model, enabled=not will_step_if_accepted):
                std_loss, losses = \
                    self.compute_standard_loss(tokens_in, continuous_tokens, targets, modalities, time_expanded)
                standard_nonfinite = _reduce_any(not torch.isfinite(std_loss))
                if not standard_nonfinite:
                    self.fabric.backward(float(self.config.not_elis_loss_weight) * std_loss)

            if standard_nonfinite:
                self.optimizer.zero_grad(set_to_none=True)
                self._reset_pending_window_state(metric_state)
                self._mark_active_train_batch_consumed(self.training_step + 1)
                print(
                    f"[rank {self.fabric.global_rank}] batch {self.training_step + 1}: "
                    "Non-finite standard loss; discarded accumulation window.",
                    flush=True,
                )
                wandb.log({"nonfinite_loss/STANDARD": 1})
                continue

            # Only a fully accepted standard backward advances the logical window.
            self.pending_microbatches += 1
            standard_loss_value = float(std_loss.detach().item())
            weighted_objective = float(self.config.not_elis_loss_weight) * standard_loss_value
            if elis_should_backward:
                weighted_objective += float(self.config.elis_loss_weight) * elis_loss_value
            self.pending_window_loss += weighted_objective

            contributions = {
                'Standard weighted contribution': float(self.config.not_elis_loss_weight) * standard_loss_value,
                'ELIS weighted contribution': (float(self.config.elis_loss_weight) * elis_loss_value
                                                if elis_should_backward else 0.0),
            }
            for name, value in contributions.items():
                metric_state[name] = metric_state.get(name, 0.0) + value

            for name, value in losses.items():
                metric_state[name] = metric_state.get(name, 0.0) + float(value.detach())

            if will_step_if_accepted:
                window_size = self.pending_microbatches
                mean_weighted_loss = self.pending_window_loss / window_size
                update_metrics = self.step_optimizer(
                    accumulated_microbatches=window_size,
                )
                self._reset_pending_window_state()
                # The optimizer update is the durable state transition. Commit its
                # input batch before nonessential reporting so a logging failure
                # cannot cause the updated batch to be replayed after resume.
                self._mark_active_train_batch_consumed(self.training_step + 1)
                self._log_completed_training_window(
                    metric_state,
                    loader_len,
                    window_size,
                    mean_weighted_loss,
                    update_metrics,
                )

            else:
                self._mark_active_train_batch_consumed(self.training_step + 1)

        if not self._accumulation_boundary_is_empty():
            raise RuntimeError("Training epoch ended with an unstepped gradient accumulation window.")
        if self.it == updates_at_start and (expected_remaining_batches or loader_len == 0):
            raise RuntimeError('Training epoch accepted no optimizer updates.')
        train_cursor.finish_epoch()
        self.current_batch_index = train_cursor.next_batch_index
        time_taken_for_epoch_in_mins = (time.time() - time_start_epoch) / 60
        if self.fabric.global_rank == 0:
            print(f"Epoch {self.epoch + 1} completed in {time_taken_for_epoch_in_mins:.2f} minutes.")
        wandb.log({"misc/Time taken for epoch (mins)": time_taken_for_epoch_in_mins})

        # A completed epoch resumes at the next epoch, not at batch zero of this one.
        self.epoch += 1
        train_cursor.set_epoch(self.epoch)

    def train(self):
        epochs = self.config.epochs
        for self.epoch in range(self.epoch, epochs):

            if self.config.stop_early_now is not None and self.epoch > self.config.stop_early_now:
                if self.fabric.global_rank == 0:
                    print(f"Reached configured epoch-index cutoff {self.config.stop_early_now}.")
                break

            self.train_epoch()
            cutoff = self.config.stop_early_now
            more_epochs = self.epoch < epochs and (cutoff is None or self.epoch <= cutoff)
            if self.config.save and more_epochs:
                self.save_model(id='last')

        if self.config.evaluate:
            self.evaluate()
        if getattr(self.config, 'longitudinal_evaluation', False):
            self.evaluate_longitudinal(
                self.validation_dataloader_with_morethan1_visit, 'Validation')
        if self.config.save:
            self.save_model(id='last')
        return 0
