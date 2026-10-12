"""Multimodal sequences and dataset augmentations."""

from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Tuple
from typing import Union
import random
import torch


class CustomDataset(Dataset):
    """Aligned value, modality, and time sequences with optional augmentation."""

    def __init__(
        self,
        tokens,
        continuous_tokens,
        bins,
        modalities,
        time_expanded,
        modality_name_to_number,
        number_of_modalities,
        cat_cols,
        index,
        mod2bininfo,
        visit_marking,
        augmentation_chance=0.5,
        augmentation_rate=0.1,
        random_removal_rate=0.1,
        random_removal_chance=0.1,
        random_block_removal_rate=0.1,
        random_block_removal_chance=0.1,
        random_block_removal_number=5,
        augment_all=True,
        config=None,
    ):
        self.tokens = tokens
        self.continuous_tokens = continuous_tokens
        self.bins = bins
        self.modalities = modalities
        self.time_expanded = time_expanded
        self.modality_name_to_number = modality_name_to_number
        self.number_of_modalities = number_of_modalities
        self.cat_cols = cat_cols
        self.index = index
        self.augmentation_chance = augmentation_chance
        self.augmentation_rate = augmentation_rate
        self.random_removal_rate = random_removal_rate
        self.random_removal_chance = random_removal_chance
        self.random_block_removal_rate = random_block_removal_rate
        self.random_block_removal_chance = random_block_removal_chance
        self.random_block_removal_number = random_block_removal_number
        self.random_modality_subset_chance = config.random_modality_subset_chance
        self.random_modality_exclusion_chance = config.random_modality_exclusion_chance
        self.chance_to_remove_time_series_data = config.chance_to_remove_time_series_data

        self.augment_all = augment_all
        self.mod2bininfo = mod2bininfo
        self.visit_marking = visit_marking

        self.config = config

        self.mask_modality_val = number_of_modalities

    def __len__(self):
        return len(self.tokens)

    def augment_noise(self, modalities, tokens, continuous_tokens):
        """Perturb selected observed values using their declared ordered bin IDs."""
        for name, mod in self.modality_name_to_number.items():
            if name in self.cat_cols or name not in self.mod2bininfo:
                continue
            ids, _ = self.mod2bininfo[name]
            positions = torch.where((modalities == mod) & torch.isfinite(continuous_tokens))[0]
            count = int(len(positions) * self.augmentation_rate)
            if count == 0:
                continue
            positions = positions[torch.randperm(len(positions), device=positions.device)[:count]]
            ids = torch.as_tensor(ids, device=tokens.device, dtype=torch.long)
            indices = torch.searchsorted(ids, tokens[positions].long())
            amplitude = 1 if torch.rand(1).item() > 0.33 else 2
            noise = torch.randint(-amplitude, amplitude + 1, (count,), device=tokens.device)
            indices = (indices + noise).clamp(0, len(ids) - 1)
            tokens[positions] = ids[indices].to(tokens.dtype)
            edges = torch.as_tensor(self.bins[name], device=continuous_tokens.device,
                                    dtype=continuous_tokens.dtype)
            left, right = edges[indices], edges[indices + 1]
            values = torch.normal(left / 2 + right / 2, (right - left) / 6)
            continuous_tokens[positions] = values.clamp(min=left, max=right)
        return tokens, continuous_tokens

    def _filter_and_pad(
            self,
            tokens: torch.Tensor,
            continuous_tokens: torch.Tensor,
            modalities: torch.Tensor,
            time_expanded: torch.Tensor,
            keep_mask: torch.Tensor,
            original_length: int,
            continuous_padding_value: Union[int, float] = 0,
            skip_padding_when_unchanged: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply an existing keep mask and restore the caller's target length."""
        tokens_filtered = tokens[keep_mask]
        continuous_tokens_filtered = continuous_tokens[keep_mask]
        modalities_filtered = modalities[keep_mask]
        time_expanded_filtered = time_expanded[keep_mask]

        pad_len = original_length - len(tokens_filtered)
        if skip_padding_when_unchanged and pad_len == 0:
            return (
                tokens_filtered,
                continuous_tokens_filtered,
                modalities_filtered,
                time_expanded_filtered,
            )

        tokens_padded = F.pad(tokens_filtered, (0, pad_len), mode='constant', value=0)
        continuous_tokens_padded = F.pad(
            continuous_tokens_filtered,
            (0, pad_len),
            mode='constant',
            value=continuous_padding_value,
        )
        modalities_padded = F.pad(
            modalities_filtered,
            (0, pad_len),
            mode='constant',
            value=self.mask_modality_val,
        )
        time_expanded_padded = F.pad(
            time_expanded_filtered,
            (0, 0, 0, pad_len),
            mode='constant',
            value=0,
        )

        return tokens_padded, continuous_tokens_padded, modalities_padded, time_expanded_padded

    def remove_random_tokens_and_pad(self, tokens, continuous_tokens, modalities, time_expanded):
        """Remove random non-marker tokens and pad all aligned channels to the original length."""

        eligible = torch.where((modalities != self.mask_modality_val)
                               & (modalities != self.visit_marking))[0]
        num_to_remove = int(len(eligible) * self.random_removal_rate)
        if num_to_remove == 0:
            return tokens, continuous_tokens, modalities, time_expanded
        indices_to_remove = eligible[torch.randperm(len(eligible), device=tokens.device)[:num_to_remove]]

        keep_mask = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)
        keep_mask[indices_to_remove] = False

        original_length = len(tokens)
        return self._filter_and_pad(
            tokens,
            continuous_tokens,
            modalities,
            time_expanded,
            keep_mask,
            original_length,
        )

    def remove_random_token_block_and_pad(self, tokens, continuous_tokens, modalities, time_expanded):
        """Remove blocks from real events, protect visit markers, and restore padding."""
        if self.random_block_removal_number == 0:
            return tokens, continuous_tokens, modalities, time_expanded
        original_length = len(tokens)
        real_length = int((modalities != self.mask_modality_val).sum())
        if real_length == 0:
            return tokens, continuous_tokens, modalities, time_expanded

        # Retain the original count jitter for nonzero settings.
        num = int(torch.randint(self.random_block_removal_number - 3, self.random_block_removal_number + 3, (1,)))

        num_blocks_to_remove = min(max(num, 1), real_length)

        block_starts = torch.randperm(real_length, device=tokens.device)[:num_blocks_to_remove]

        # Retain the original +/-1% relative rate jitter.
        perct = self.random_block_removal_rate * 0.01
        rate = torch.rand(1) * (2 * perct) + self.random_block_removal_rate - perct

        block_size = int(rate * real_length)

        keep_mask = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)

        for start in block_starts:
            end = min(start + block_size, real_length)
            keep_mask[start:end] = False

        # Protect visit markings from removal
        visit_positions = (modalities == self.visit_marking)
        keep_mask[visit_positions] = True

        return self._filter_and_pad(
            tokens,
            continuous_tokens,
            modalities,
            time_expanded,
            keep_mask,
            original_length,
        )

    def random_modality_subset_and_pad(
            self,
            tokens: torch.Tensor,
            continuous_tokens: torch.Tensor,
            modalities: torch.Tensor,
            time_expanded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep a random modality subset and visit markers, then restore the original length."""

        original_length = len(tokens)

        num_to_keep = int(self.number_of_modalities * self.config.random_modality_subset_fraction)
        if num_to_keep < 1:
            num_to_keep = 1
        elif num_to_keep > self.number_of_modalities:
            num_to_keep = self.number_of_modalities

        # Sample non-marker modalities; retain visit markers separately.
        all_possible_mods = list(range(self.number_of_modalities))
        if self.visit_marking in all_possible_mods:
            all_possible_mods.remove(self.visit_marking)
        num_to_keep = min(num_to_keep, len(all_possible_mods))

        chosen_mods = random.sample(all_possible_mods, num_to_keep)

        keep_mask = (modalities == self.visit_marking)
        for m in chosen_mods:
            keep_mask |= (modalities == m)

        return self._filter_and_pad(
            tokens,
            continuous_tokens,
            modalities,
            time_expanded,
            keep_mask,
            original_length,
        )

    def random_modality_exclusion_and_pad(
            self,
            tokens: torch.Tensor,
            continuous_tokens: torch.Tensor,
            modalities: torch.Tensor,
            time_expanded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exclude selected modalities while preserving visit markers and aligned padding."""

        original_length = tokens.shape[0]

        exclusion_fraction = 0.1
        num_to_exclude = int(self.number_of_modalities * exclusion_fraction)
        if num_to_exclude < 1:
            num_to_exclude = 1
        elif num_to_exclude > self.number_of_modalities - 1:
            num_to_exclude = self.number_of_modalities - 1

        # Exclude only non-marker modalities.
        all_possible_mods = list(range(self.number_of_modalities))
        if self.visit_marking in all_possible_mods:
            all_possible_mods.remove(self.visit_marking)

        if num_to_exclude > len(all_possible_mods):
            num_to_exclude = len(all_possible_mods)

        excluded_mods = random.sample(all_possible_mods, num_to_exclude)

        keep_mask = (modalities != self.mask_modality_val)
        for m in excluded_mods:
            keep_mask &= (modalities != m)

        return self._filter_and_pad(
            tokens,
            continuous_tokens,
            modalities,
            time_expanded,
            keep_mask,
            original_length,
        )

    def remove_time_series_data(
            self,
            tokens: torch.Tensor,
            continuous_tokens: torch.Tensor,
            modalities: torch.Tensor,
            time_expanded: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove the configured time-series modality groups and restore padding."""
        original_length = len(tokens)
        if original_length == 0:  # Handle empty input case
            return tokens, continuous_tokens, modalities, time_expanded

        exact_names = {"GlucoseValue", "sleep_event"}
        suffixes = ("_apple_google", "_species", "_gene", "_questionires")
        removed_modalities = {number for name, number in self.modality_name_to_number.items()
                              if name in exact_names or name.endswith(suffixes)}
        keep_mask = torch.ones_like(modalities, dtype=torch.bool)
        for number in removed_modalities:
            keep_mask[modalities == number] = False

        return self._filter_and_pad(
            tokens, continuous_tokens, modalities, time_expanded, keep_mask, original_length,
            continuous_padding_value=0.0, skip_padding_when_unchanged=True,
        )

    def __getitem__(self, idx, augment=True):
        tokens, continuous_tokens, modalities, time_expanded = (
            self.tokens[idx], self.continuous_tokens[idx],
            self.modalities[idx], self.time_expanded[idx],
        )
        # The tokenizer stores modality IDs compactly as int16, while embedding
        # and indexed-lookup operations require int32 or int64 IDs.
        modalities = modalities.int()
        augment = augment and self.augment_all
        if augment:
            tokens = tokens.clone()
            continuous_tokens = continuous_tokens.clone()
            modalities = modalities.clone()
            time_expanded = time_expanded.clone()
            if torch.rand(1) < self.augmentation_chance:
                tokens, continuous_tokens = self.augment_noise(
                    modalities, tokens, continuous_tokens,
                )

            if torch.rand(1) < self.random_removal_chance:
                tokens, continuous_tokens, modalities, time_expanded = (
                    self.remove_random_tokens_and_pad(
                        tokens, continuous_tokens, modalities, time_expanded,
                    )
                )
            if torch.rand(1) < self.random_block_removal_chance:
                tokens, continuous_tokens, modalities, time_expanded = (
                    self.remove_random_token_block_and_pad(
                        tokens, continuous_tokens, modalities, time_expanded,
                    )
                )
            if torch.rand(1) < self.random_modality_subset_chance:
                tokens, continuous_tokens, modalities, time_expanded = (
                    self.random_modality_subset_and_pad(
                        tokens, continuous_tokens, modalities, time_expanded,
                    )
                )

            if torch.rand(1) < self.random_modality_exclusion_chance:
                tokens, continuous_tokens, modalities, time_expanded = (
                    self.random_modality_exclusion_and_pad(
                        tokens, continuous_tokens, modalities, time_expanded,
                    )
                )
            if torch.rand(1) < self.chance_to_remove_time_series_data:
                tokens, continuous_tokens, modalities, time_expanded = (
                    self.remove_time_series_data(
                        tokens, continuous_tokens, modalities, time_expanded,
                    )
                )

        # A rejected augmentation restores every channel, including values changed by noise.
        if augment and int((modalities != self.mask_modality_val).sum()) < 2:
            tokens = self.tokens[idx].clone()
            continuous_tokens = self.continuous_tokens[idx].clone()
            modalities = self.modalities[idx].int().clone()
            time_expanded = self.time_expanded[idx].clone()

        return tokens, continuous_tokens, modalities, time_expanded
