"""Distributed sampling and resumable batch cursors."""

from typing import Any
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset
from typing import Dict
from torch.utils.data import DistributedSampler
from typing import Iterator
from typing import List
from typing import Optional
from torch.utils.data import Sampler
from torch.utils.data import Subset
import hashlib
from itertools import islice
import numpy as np
import torch

class ResumableBatchSampler:
    """Batch-level cursor that skips consumed index batches before dataset access."""

    STATE_VERSION = 1

    def __init__(self, sampler, batch_size: int, drop_last: bool):
        self._batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
        self._epoch = int(getattr(sampler, "epoch", 0))
        self.next_batch_index = 0

        dataset = getattr(sampler, "dataset", None)
        if dataset is None:
            raise TypeError("ResumableBatchSampler requires a sampler with a dataset attribute.")

        sampler_type = type(sampler)
        self._sampler_class = f"{sampler_type.__module__}.{sampler_type.__qualname__}"
        self._sampler_seed = int(getattr(sampler, "seed", 0))
        self._dataset_length = len(dataset)
        self._dataset_identity = self._build_dataset_identity(dataset)

    @staticmethod
    def _fingerprint_ordered_integers(values) -> Optional[str]:
        if values is None:
            return None
        digest = hashlib.sha256()
        count = 0
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"Dataset identity values must be integers, got {value!r}."
                )
            encoded = str(int(value)).encode("ascii")
            digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
            digest.update(encoded)
            count += 1
        digest.update(count.to_bytes(8, byteorder="big", signed=False))
        return digest.hexdigest()

    @classmethod
    def _build_dataset_identity(cls, dataset) -> Dict[str, Optional[str]]:
        if isinstance(dataset, Subset):
            view_indices = list(dataset.indices)
            backing_index = getattr(dataset.dataset, "index", None)
            participant_ids = getattr(dataset, "index", None)
            if participant_ids is None and backing_index is not None:
                participant_ids = [backing_index[int(index)] for index in view_indices]
        else:
            view_indices = list(range(len(dataset)))
            backing_index = getattr(dataset, "index", None)
            participant_ids = backing_index

        return {
            "dataset_view_indices_sha256": cls._fingerprint_ordered_integers(view_indices),
            "dataset_participant_ids_sha256": cls._fingerprint_ordered_integers(participant_ids),
            "backing_participant_ids_sha256": cls._fingerprint_ordered_integers(backing_index),
        }

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def total_batches(self) -> int:
        return len(self._batch_sampler)

    def __iter__(self):
        if not 0 <= self.next_batch_index <= self.total_batches:
            raise RuntimeError(
                f"Invalid batch cursor {self.next_batch_index}; expected 0..{self.total_batches}."
            )
        return islice(iter(self._batch_sampler), self.next_batch_index, None)

    def __len__(self) -> int:
        return self.total_batches - self.next_batch_index

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}.")
        if epoch != self._epoch:
            if self.next_batch_index != 0:
                raise RuntimeError(
                    "Cannot change sampler epoch while a nonzero batch cursor is active."
                )
            self._epoch = epoch

        underlying_sampler = self._batch_sampler.sampler
        if hasattr(underlying_sampler, "set_epoch"):
            underlying_sampler.set_epoch(epoch)

    def mark_consumed(self, expected_absolute_index: int):
        expected_absolute_index = int(expected_absolute_index)
        required_index = self.next_batch_index + 1
        if expected_absolute_index != required_index:
            raise RuntimeError(
                "Batch cursor must advance by exactly one: "
                f"expected {required_index}, got {expected_absolute_index}."
            )
        if expected_absolute_index > self.total_batches:
            raise RuntimeError(
                f"Batch cursor {expected_absolute_index} exceeds total batches {self.total_batches}."
            )
        self.next_batch_index = expected_absolute_index

    def finish_epoch(self):
        if self.next_batch_index != self.total_batches:
            raise RuntimeError(
                "Cannot finish an incomplete sampler epoch: "
                f"consumed {self.next_batch_index} of {self.total_batches} batches."
            )
        self.next_batch_index = 0

    def state_dict(self) -> Dict[str, int]:
        return {
            "version": self.STATE_VERSION,
            "epoch": self.epoch,
            "next_batch_index": self.next_batch_index,
        }

    def load_state_dict(self, state: Dict[str, int]):
        if not isinstance(state, dict):
            raise TypeError("Batch cursor state must be a dictionary.")
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise RuntimeError(
                f"Unsupported batch cursor version {state.get('version')!r}; "
                f"expected {self.STATE_VERSION}."
            )
        if self.next_batch_index != 0:
            raise RuntimeError("Cannot restore a batch cursor over nonzero in-memory progress.")

        restored_epoch = int(state["epoch"])
        restored_next_batch = int(state["next_batch_index"])
        if not 0 <= restored_next_batch <= self.total_batches:
            raise RuntimeError(
                f"Restored batch cursor {restored_next_batch} is outside 0..{self.total_batches}."
            )

        self.set_epoch(restored_epoch)
        self.next_batch_index = restored_next_batch

    def structural_metadata(self) -> Dict[str, Any]:
        metadata = {
            "sampler_class": self._sampler_class,
            "sampler_seed": self._sampler_seed,
            "dataset_length": self._dataset_length,
            "batch_size": self._batch_sampler.batch_size,
            "batch_drop_last": self._batch_sampler.drop_last,
            "total_batches": self.total_batches,
        }
        metadata.update(self._dataset_identity)
        return metadata

class EvaluationSubset(Subset):
    """A dataset subset that always retrieves samples without augmentation."""

    def __getitem__(self, index):
        if isinstance(index, list):
            return [
                self.dataset.__getitem__(self.indices[item], augment=False)
                for item in index
            ]
        return self.dataset.__getitem__(self.indices[index], augment=False)

    def __getitems__(self, indices):
        return [
            self.dataset.__getitem__(self.indices[index], augment=False)
            for index in indices
        ]

class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices across ranks without padding or duplication."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        if isinstance(num_replicas, bool) or int(num_replicas) <= 0:
            raise ValueError("num_replicas must be a positive integer.")
        if isinstance(rank, bool) or not 0 <= int(rank) < int(num_replicas):
            raise ValueError("rank must be in [0, num_replicas).")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


class EliFriendlyDistributedSampler(DistributedSampler):
    """Shuffle multi-visit groups and balance each global batch by sequence length.

    Metadata is indexed within the supplied dataset view. The training path
    always shuffles and drops the distributed remainder without duplicating rows.
    """

    def __init__(self, dataset: Dataset, batch_size: int, num_replicas: int = None,
                 rank: int = None, seed: int = 0, *, lengths, multi_visit_rows):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=True,
                         seed=seed, drop_last=True)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if len(lengths) != len(dataset) or not set(multi_visit_rows).issubset(range(len(dataset))):
            raise ValueError("Sampler metadata must match the dataset view.")
        self.batch_size = batch_size
        self.local_idx_to_len = list(lengths)
        self.multi_visit_indices = [i for i in range(len(dataset)) if i in multi_visit_rows]
        self.single_visit_indices = [i for i in range(len(dataset)) if i not in multi_visit_rows]

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Shuffle indices within each group for the new epoch.
        multi_visit_shuffled = [self.multi_visit_indices[i] for i in
                                torch.randperm(len(self.multi_visit_indices), generator=g)]
        single_visit_shuffled = [self.single_visit_indices[i] for i in
                                 torch.randperm(len(self.single_visit_indices), generator=g)]

        # Create a combined list of indices, prioritizing multi-visit batches.
        global_batch_size = self.batch_size * self.num_replicas

        num_full_multi_visit_batches = len(multi_visit_shuffled) // global_batch_size

        # Start with full batches of only multi-visit samples
        epoch_indices = multi_visit_shuffled[:num_full_multi_visit_batches * global_batch_size]

        # Combine remaining multi-visit samples with all single-visit samples
        remaining_indices = multi_visit_shuffled[
                                num_full_multi_visit_batches * global_batch_size:] + single_visit_shuffled

        # Shuffle the combined remainder to mix them up
        remaining_shuffled = [remaining_indices[i] for i in torch.randperm(len(remaining_indices), generator=g)]

        epoch_indices.extend(remaining_shuffled)

        # Truncate to total_size to handle drop_last=True.
        effective_epoch_indices = epoch_indices[:self.total_size]

        # Distribute using snake pairing.
        my_rank_indices_this_epoch: List[int] = []
        rank_counts = [0] * self.num_replicas

        for i_global_step in range(0, self.total_size, global_batch_size):
            global_chunk_indices = effective_epoch_indices[i_global_step: i_global_step + global_batch_size]
            if not global_chunk_indices: continue

            # Sort this chunk by pre-calculated length
            global_chunk_indices.sort(key=lambda idx: self.local_idx_to_len[idx], reverse=True)

            indices_for_ranks_this_step: List[List[int]] = [[] for _ in range(self.num_replicas)]

            for i_item, local_idx in enumerate(global_chunk_indices):
                row_num = i_item // self.num_replicas
                col_num = i_item % self.num_replicas

                assigned_rank = col_num if row_num % 2 == 0 else self.num_replicas - 1 - col_num

                if len(indices_for_ranks_this_step[assigned_rank]) < self.batch_size:
                    indices_for_ranks_this_step[assigned_rank].append(local_idx)

            for rank, indices in enumerate(indices_for_ranks_this_step):
                rank_counts[rank] += len(indices)
            my_rank_indices_this_epoch.extend(indices_for_ranks_this_step[self.rank])

        # Every rank checks every count before yielding a sample. An internal
        # partitioning error must stop the run, not silently duplicate participants.
        if any(count != self.num_samples for count in rank_counts):
            raise RuntimeError(f"Sampler produced unequal or unexpected rank lengths: {rank_counts}; "
                               f"expected {self.num_samples} per rank.")

        return iter(my_rank_indices_this_epoch)

    def __len__(self) -> int:
        return self.num_samples
