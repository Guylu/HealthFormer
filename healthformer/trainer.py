"""Trainer construction and composition of training components."""
from collections import Counter
from .visits import identify_visits

from .checkpointing import (
    CheckpointMixin,
)
from .data import (
    CustomDataset,
)
from .evaluation import (
    EvaluationMixin,
)
from .inputs import (
    InputMixin,
)
from .longitudinal import (
    LongitudinalMixin,
)
from .model import (
    MultiModalTransformer,
)
from .training import (
    OptimizationMixin,
)
from omegaconf import OmegaConf
import numpy as np
import torch
from tqdm import tqdm
from .tracking import tracker as wandb

class Trainer(InputMixin, OptimizationMixin, EvaluationMixin, LongitudinalMixin, CheckpointMixin):
    TRAIN_CURSOR_CHECKPOINT_VERSION = 2

    ACCUMULATION_CHECKPOINT_VERSION = 1

    @staticmethod
    def _validate_accumulate_grad_batches(value) -> int:
        """Return a valid microbatch accumulation count."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("accumulate_grad_batches must be an integer greater than or equal to 1.")
        value = int(value)
        if value < 1:
            raise ValueError("accumulate_grad_batches must be greater than or equal to 1.")
        return value

    @staticmethod
    def _optimizer_steps_per_epoch(total_batches: int, accumulation: int) -> int:
        """Return the number of optimizer updates, including a short epoch tail."""
        if isinstance(total_batches, (bool, np.bool_)) or not isinstance(total_batches, (int, np.integer)):
            raise TypeError("total_batches must be a non-negative integer.")
        total_batches = int(total_batches)
        if total_batches < 0:
            raise ValueError("total_batches must be non-negative.")
        accumulation = Trainer._validate_accumulate_grad_batches(accumulation)
        return (total_batches + accumulation - 1) // accumulation

    def __init__(self, config, fabric, augment_all=True):

        self.augment_all = augment_all


        self.fabric = fabric

        from .config import normalize_config
        self.config = OmegaConf.create(normalize_config(config))
        self.config.accumulate_grad_batches = self._validate_accumulate_grad_batches(
            self.config.accumulate_grad_batches
        )
        from .config import validate_methodology_config, validate_distributed_config
        validate_methodology_config(self.config)
        validate_distributed_config(
            self.config,
            self.fabric.world_size,
        )
        self.pending_microbatches = 0
        self.pending_window_loss = 0.0

        self.device = self.fabric.device

        self.epoch = 0
        self.best_val_loss = float('inf')


        data = self._load_training_artifact()
        self._validate_training_artifact(data)
        self.preprocessing_provenance = data.get('preprocessing_provenance', {'status': 'unknown'})

        self.padded_tokens_tensor, self.padded_modality_indicators_tensor, self.padded_time_expanded_tensor \
            = data['tokens'], data['modalities'], data['time_expanded'].int()
        self.cum_base_value_per_modality = data['cum_base_value_per_modality']
        self.modality_name_to_number, self.number_to_modality_name, self.vals_per_modality \
            = data['modality_name_to_number'], data['number_to_modality_name'], data['vals_per_modality']
        self.vocab_size, self.temporal_vocab_size, self.time_dim_order \
            = data['vocab_size'], data['temporal_vocab_size'], data['time_dim_order']
        self.number_of_modalities = data['number_of_modalities']
        self.cat_cols = data['cat_cols']

        self.year_idx = self.time_dim_order.index('year')

        self.continuous_tokens = data['continuous_tokens']
        self.bins = data['bins']
        self.index = data['index']


        self.mod2bininfo = {}
        for mod_name in tqdm(self.bins.keys(), "mod2bininfo"):
            bin_edges = self.bins[mod_name]  # length N+1
            bin_midpoints = [
                float(bin_edges[i]) / 2.0 + float(bin_edges[i + 1]) / 2.0
                for i in range(len(bin_edges) - 1)
            ]
            bin_midpoints = np.array(bin_midpoints, dtype=np.float32)

            # Store absolute token IDs in the unified vocabulary.
            mod_num = self.modality_name_to_number[mod_name]
            token_indices = np.array(
                sorted(self.vals_per_modality[mod_name]), dtype=np.int64
            ) + self.cum_base_value_per_modality[mod_num]

            self.mod2bininfo[mod_name] = (token_indices, bin_midpoints)

        self._select_participant_rows(self.index)
        if not self.train_dataset_index:
            raise ValueError("No training participants remain after selection.")
        training_counts = torch.zeros(self.number_of_modalities + 1, dtype=torch.long)
        for start in range(0, len(self.train_dataset_index), 32):
            rows = self.train_dataset_index[start:start + 32]
            lengths = (self.padded_modality_indicators_tensor[rows]
                       != self.number_of_modalities).sum(dim=1)
            if (lengths < 2).any():
                raise ValueError('Training records must contain at least two real tokens.')
            if (lengths > self.config.max_seq_length).any():
                raise ValueError('Training record exceeds max_seq_length; prepare bounded records explicitly.')
            training_counts += torch.bincount(
                self.padded_modality_indicators_tensor[rows].flatten().long(),
                minlength=self.number_of_modalities + 1)

        # Select the marker using training counts only; preserve mapping order for ties.
        dexa_modalities = [mod for name, mod in self.modality_name_to_number.items()
                          if name.split('_')[-1] == 'DEXALoader']
        self.visit_marking = max(dexa_modalities, key=lambda mod: int(training_counts[mod]))

        visit_exclusions = Counter()
        self.sequence_lengths = []
        self.multi_visit_rows = set()
        for i, row in enumerate(self.padded_modality_indicators_tensor):
            valid = row != self.number_of_modalities
            self.sequence_lengths.append(int(valid.sum()))
            visits, reason = identify_visits(row[valid], self.padded_time_expanded_tensor[i][valid],
                                            self.visit_marking, self.year_idx)
            if reason is not None:
                visit_exclusions[reason] += 1
            if visits is not None:
                self.multi_visit_rows.add(i)

        if self.fabric.global_rank == 0:
            print(f"Artifact distinct-year exclusions: {dict(visit_exclusions)}")

        self.dataset = CustomDataset(
            tokens=self.padded_tokens_tensor, continuous_tokens=self.continuous_tokens,
            bins=self.bins, modalities=self.padded_modality_indicators_tensor,
            time_expanded=self.padded_time_expanded_tensor,
            modality_name_to_number=self.modality_name_to_number,
            number_of_modalities=self.number_of_modalities, cat_cols=self.cat_cols,
            index=self.index, mod2bininfo=self.mod2bininfo, visit_marking=self.visit_marking,
            augmentation_chance=self.config.augmentation_chance,
            augmentation_rate=self.config.augmentation_rate,
            random_removal_rate=self.config.random_removal_rate,
            random_removal_chance=self.config.random_removal_chance,
            random_block_removal_rate=self.config.random_block_removal_rate,
            random_block_removal_chance=self.config.random_block_removal_chance,
            random_block_removal_number=self.config.random_block_removal_number,
            augment_all=self.augment_all, config=self.config,
        )

        self._build_dataset_views()



        self.init_dataloaders()
        if self.train_batch_sampler.total_batches == 0:
            raise ValueError('No usable training batches; reduce batch_per_gpu/devices or provide more records.')

        device = self.fabric.device
        torch.set_float32_matmul_precision('high')
        self.mask_modality_val = self.number_of_modalities
        self.continuous_min_max_stats = None
        self.mod2stats = None

        # Compute cohort statistics once and share them with every worker.
        if self.fabric.global_rank == 0:
            self.mod2stats, self.continuous_min_max_stats = self.compute_training_statistics()

        self.fabric.barrier()
        self.continuous_min_max_stats = self.fabric.broadcast(self.continuous_min_max_stats, src=0)
        self.mod2stats = self.fabric.broadcast(self.mod2stats, src=0)

        self.fabric.barrier()

        with self.fabric.init_module():
            self.model = MultiModalTransformer(
                num_modalities=self.number_of_modalities, vocab_size=self.vocab_size,
                mask_modality_val=self.mask_modality_val, temporal_vocab_size=self.temporal_vocab_size,
                config=self.config, vals_per_modality=self.vals_per_modality,
                modality_name_to_number=self.modality_name_to_number,
                cum_base_value_per_modality=self.cum_base_value_per_modality,
                cat_cols=self.cat_cols, mod2bininfo=self.mod2bininfo, mod2stats=self.mod2stats,
                continuous_min_max_stats=self.continuous_min_max_stats,
            )

        self.max_lr = float(self.config.lr)
        self.min_lr = self.max_lr * self.config.gamma
        self.warmup_steps = self.config.warmup_steps

        self.optimizer = self.model.configure_optimizers(float(self.config.weight_decay), float(self.config.lr), device)
        self.it = 0
        self.current_batch_index = 0

        self.fabric.barrier()
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        self.train_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader,
            use_distributed_sampler=False,
        )
        self.validation_dataloader = self.fabric.setup_dataloaders(
            self.validation_dataloader,
            use_distributed_sampler=False,
        )

        self.max_steps = self._optimizer_steps_per_epoch(
            self._active_train_cursor().total_batches,
            self.config.accumulate_grad_batches,
        ) * self.config.epochs

        from .identity import experiment_identity
        self.experiment_identity = experiment_identity(self.config, self.split_manifest)
        self.maybe_resume()


        self.model.train()
