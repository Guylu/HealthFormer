"""Input validation, dataset views, statistics, and data loaders."""
from .serialization import load_artifact, artifact_path
from .config import is_validation_id

from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import torch
from .tracking import tracker as wandb
from .samplers import (
    EliFriendlyDistributedSampler,
    EvaluationSubset,
    ExactDistributedEvalSampler,
    ResumableBatchSampler,
)

class InputMixin:
    def _load_training_artifact(self):
        """Resolve and load the configured training artifact without changing path semantics."""
        path = artifact_path(self.config)
        return load_artifact(path)

    def _validate_training_artifact(self, data):
        """Validate the active serialized-data contract before any field is consumed."""
        self._validate_training_artifact_structure(data)
        self._validate_training_artifact_time(data)
        self._validate_training_artifact_id_ranges(data)
        self._validate_training_artifact_mapping(data)
        self._validate_training_artifact_bins(data)

    def _validate_training_artifact_mapping(self, data):
        names = data['modality_name_to_number']
        count = data['number_of_modalities']
        if (set(names.values()) != set(range(count)) or len(names) != count
                or data['number_to_modality_name'] != {v: k for k, v in names.items()}):
            raise ValueError('Modality mappings must be bijective and cover the declared IDs.')
        if set(data['vals_per_modality']) != set(names):
            raise ValueError('Every modality must declare its token IDs.')
        if not torch.is_floating_point(data['continuous_tokens']):
            raise TypeError('continuous_tokens must be floating point.')
        padding = data['modalities'] == count
        if (padding[:, :-1] & ~padding[:, 1:]).any():
            raise ValueError('Padding must be trailing: real events cannot follow padding.')
        for name, mod in names.items():
            values = list(data['vals_per_modality'][name])
            offset = data['cum_base_value_per_modality'][mod]
            if isinstance(offset, (bool, np.bool_)) or not isinstance(offset, (int, np.integer)) or offset < 0:
                raise ValueError(f'Invalid token offset for {name!r}.')
            if (not values or len(set(values)) != len(values) or any(
                    isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                    or v < 0 or not 0 <= v + offset < data['vocab_size'] for v in values)):
                raise ValueError(f'Invalid declared token IDs for {name!r}.')
            ids = torch.tensor([v + offset for v in values], dtype=torch.long)
            observed = data['tokens'][data['modalities'] == mod].long()
            if not torch.isin(observed, ids).all():
                raise ValueError(f'Token/modality mismatch for {name!r}.')

    def _validate_training_artifact_bins(self, data):
        """Reject ambiguous bin-to-token mappings before distributed training starts."""
        for name, raw_edges in data['bins'].items():
            if name not in data['modality_name_to_number'] or name not in data['vals_per_modality']:
                raise ValueError(f"Unknown bin modality {name!r}.")
            edges = np.asarray(raw_edges, dtype=np.float64)
            ids = list(data['vals_per_modality'][name])
            mod = data['modality_name_to_number'][name]
            offset = data['cum_base_value_per_modality'][mod]
            if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                    or not (np.diff(edges) > 0).all()):
                raise ValueError(f"Bin edges for {name!r} must be finite and strictly increasing.")
            if len(edges) - 1 != len(ids):
                raise ValueError(f"Bin/token count mismatch for {name!r}.")
            if (len(set(ids)) != len(ids) or any(
                    isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                    or i < 0 or not 0 <= i + offset < data['vocab_size'] for i in ids)):
                raise ValueError(f"Invalid bin token IDs for {name!r}.")
            midpoints = edges[:-1] / 2 + edges[1:] / 2
            if (np.abs(midpoints) > np.finfo(np.float32).max).any():
                raise ValueError(f"Bin midpoints for {name!r} exceed float32 range.")

    def _validate_training_artifact_structure(self, data):
        """Validate required fields, tensor types, shapes, and sample indexing."""
        required_fields = (
            'tokens',
            'continuous_tokens',
            'modalities',
            'time_expanded',
            'index',
            'cum_base_value_per_modality',
            'modality_name_to_number',
            'number_to_modality_name',
            'vals_per_modality',
            'vocab_size',
            'temporal_vocab_size',
            'time_dim_order',
            'number_of_modalities',
            'cat_cols',
            'bins',
        )

        if not hasattr(data, 'keys'):
            raise TypeError(
                f"Training artifact has invalid type: observed {type(data).__name__}, "
                "expected a mapping containing the required fields."
            )

        for field in required_fields:
            if field not in data:
                raise KeyError(
                    f"Training artifact field '{field}' is missing: observed missing, "
                    "expected the field to be present."
                )

        tensor_fields = ('tokens', 'continuous_tokens', 'modalities', 'time_expanded')
        for field in tensor_fields:
            if not isinstance(data[field], torch.Tensor):
                raise TypeError(
                    f"Training artifact field '{field}' has invalid type: "
                    f"observed {type(data[field]).__name__}, expected torch.Tensor."
                )

        for field in ('tokens', 'continuous_tokens', 'modalities'):
            observed_shape = tuple(data[field].shape)
            if data[field].ndim != 2:
                raise ValueError(
                    f"Training artifact field '{field}' has invalid shape: "
                    f"observed {observed_shape}, expected a two-dimensional tensor."
                )

        token_shape = tuple(data['tokens'].shape)
        for field in ('continuous_tokens', 'modalities'):
            observed_shape = tuple(data[field].shape)
            if observed_shape != token_shape:
                raise ValueError(
                    f"Training artifact field '{field}' has mismatched shape: "
                    f"observed {observed_shape}, expected {token_shape} to match 'tokens'."
                )

        time_shape = tuple(data['time_expanded'].shape)
        if data['time_expanded'].ndim != 3:
            raise ValueError(
                f"Training artifact field 'time_expanded' has invalid shape: "
                f"observed {time_shape}, expected a three-dimensional tensor."
            )
        expected_time_prefix = token_shape
        if tuple(data['time_expanded'].shape[:2]) != expected_time_prefix:
            raise ValueError(
                f"Training artifact field 'time_expanded' has mismatched leading shape: "
                f"observed {tuple(data['time_expanded'].shape[:2])}, "
                f"expected {expected_time_prefix} to match 'tokens'."
            )
        if data['time_expanded'].shape[2] != 7:
            raise ValueError(
                f"Training artifact field 'time_expanded' has invalid feature shape: "
                f"observed {data['time_expanded'].shape[2]} features, expected 7."
            )

        try:
            index_length = len(data['index'])
        except TypeError as exc:
            raise TypeError(
                f"Training artifact field 'index' has invalid shape: observed no length, "
                f"expected length {token_shape[0]}."
            ) from exc
        if index_length != token_shape[0]:
            raise ValueError(
                f"Training artifact field 'index' has mismatched length: "
                f"observed {index_length}, expected {token_shape[0]}."
            )
        for position, participant_id in enumerate(data['index']):
            if isinstance(participant_id, (bool, np.bool_)) or not isinstance(
                    participant_id, (int, np.integer)):
                raise TypeError(
                    "Training artifact field 'index' has invalid participant ID "
                    f"at position {position}: observed {participant_id!r}, "
                    "expected a Python or NumPy integer."
                )

    def _validate_training_artifact_time(self, data):
        """Validate temporal metadata, dtype, and per-feature value ranges."""
        expected_time_order = [
            'day_of_week',
            'hour',
            'minute',
            'month',
            'year',
            'day_of_month',
            'sleep',
        ]
        try:
            observed_time_order = list(data['time_dim_order'])
        except TypeError as exc:
            raise TypeError(
                f"Training artifact field 'time_dim_order' has invalid order: "
                f"observed {data['time_dim_order']!r}, expected {expected_time_order}."
            ) from exc
        if observed_time_order != expected_time_order:
            raise ValueError(
                f"Training artifact field 'time_dim_order' has invalid order: "
                f"observed {observed_time_order}, expected {expected_time_order}."
            )

        try:
            temporal_vocab_length = len(data['temporal_vocab_size'])
        except TypeError as exc:
            raise TypeError(
                f"Training artifact field 'temporal_vocab_size' has invalid shape: "
                "observed no length, expected 7 entries."
            ) from exc
        if temporal_vocab_length != 7:
            raise ValueError(
                f"Training artifact field 'temporal_vocab_size' has invalid shape: "
                f"observed {temporal_vocab_length} entries, expected 7."
            )

        temporal_vocab_sizes = list(data['temporal_vocab_size'])
        for feature_name, feature_vocab_size in zip(expected_time_order, temporal_vocab_sizes):
            if isinstance(feature_vocab_size, (bool, np.bool_)) or not isinstance(
                    feature_vocab_size, (int, np.integer)):
                raise TypeError(
                    "Training artifact field 'temporal_vocab_size' has an invalid entry for "
                    f"'{feature_name}': observed {feature_vocab_size!r}, expected a positive integer."
                )
            if feature_vocab_size <= 0:
                raise ValueError(
                    "Training artifact field 'temporal_vocab_size' has an invalid entry for "
                    f"'{feature_name}': observed {feature_vocab_size}, expected a positive integer."
                )

        time_tensor = data['time_expanded']
        if time_tensor.dtype == torch.bool or time_tensor.is_complex():
            raise TypeError(
                "Training artifact field 'time_expanded' has invalid feature dtype: "
                f"observed {time_tensor.dtype}, expected integer-valued numeric features."
            )
        if torch.is_floating_point(time_tensor):
            if not torch.isfinite(time_tensor).all():
                raise ValueError(
                    "Training artifact field 'time_expanded' contains non-finite values; "
                    "expected finite integer-valued features."
                )
            if not torch.equal(time_tensor, torch.round(time_tensor)):
                raise ValueError(
                    "Training artifact field 'time_expanded' contains fractional values; "
                    "expected integer-valued features."
                )
        else:
            try:
                torch.iinfo(time_tensor.dtype)
            except TypeError as exc:
                raise TypeError(
                    "Training artifact field 'time_expanded' has invalid feature dtype: "
                    f"observed {time_tensor.dtype}, expected integer-valued numeric features."
                ) from exc

        for feature_index, (feature_name, feature_vocab_size) in enumerate(
                zip(expected_time_order, temporal_vocab_sizes)):
            feature_values = time_tensor[:, :, feature_index]
            if feature_values.numel() == 0:
                continue
            observed_min = feature_values.min().item()
            observed_max = feature_values.max().item()
            if observed_min < 0 or observed_max >= feature_vocab_size:
                raise ValueError(
                    "Training artifact field 'time_expanded' has values outside the declared range "
                    f"for '{feature_name}': observed range [{observed_min}, {observed_max}], "
                    f"expected [0, {feature_vocab_size})."
                )

    def _validate_training_artifact_id_ranges(self, data):
        """Validate token and modality ID range declarations and values."""
        vocab_size = data['vocab_size']
        if isinstance(vocab_size, (bool, np.bool_)) or not isinstance(vocab_size, (int, np.integer)):
            raise TypeError(
                f"Training artifact field 'vocab_size' has invalid range declaration: "
                f"observed {vocab_size!r}, expected a positive integer."
            )
        if vocab_size <= 0:
            raise ValueError(
                f"Training artifact field 'vocab_size' has invalid range declaration: "
                f"observed {vocab_size}, expected a positive integer."
            )

        number_of_modalities = data['number_of_modalities']
        if isinstance(number_of_modalities, (bool, np.bool_)) or not isinstance(
                number_of_modalities, (int, np.integer)):
            raise TypeError(
                f"Training artifact field 'number_of_modalities' has invalid range declaration: "
                f"observed {number_of_modalities!r}, expected a non-negative integer."
            )
        if number_of_modalities < 0:
            raise ValueError(
                f"Training artifact field 'number_of_modalities' has invalid range declaration: "
                f"observed {number_of_modalities}, expected a non-negative integer."
            )

        id_ranges = (
            ('tokens', vocab_size, f"[0, {vocab_size})"),
            ('modalities', number_of_modalities + 1, f"[0, {number_of_modalities}]"),
        )
        for field, upper_bound, expected_range in id_ranges:
            tensor = data[field]
            try:
                torch.iinfo(tensor.dtype)
            except TypeError as exc:
                raise TypeError(
                    f"Training artifact field '{field}' has invalid ID dtype: "
                    f"observed {tensor.dtype}, expected an integer tensor with values in {expected_range}."
                ) from exc

            if tensor.numel() == 0:
                continue
            observed_min = tensor.min().item()
            observed_max = tensor.max().item()
            if observed_min < 0 or observed_max >= upper_bound:
                raise ValueError(
                    f"Training artifact field '{field}' has IDs outside its declared range: "
                    f"observed range [{observed_min}, {observed_max}], expected {expected_range}."
                )

    def _select_participant_rows(self, participant_ids):
        self.train_dataset_index = [i for i, pid in enumerate(participant_ids)
                                    if not is_validation_id(pid, self.config)]
        self.validation_dataset_index = [i for i, pid in enumerate(participant_ids)
                                         if is_validation_id(pid, self.config)]

    def _build_dataset_views(self):
        """Build ordered train/validation views while preserving their backing datasets."""
        percent_validation = self.config.percent_validation
        wandb.log({"misc/Percent validation": percent_validation})

        if not hasattr(self, "train_dataset_index"):
            self._select_participant_rows(self.dataset.index)


        validation_dataset_with_morethan1_visit = [
            i for i in self.validation_dataset_index if i in self.multi_visit_rows]

        # Training clones samples before augmentation; validation retrieves unaugmented views.

        # The held-out split is validation; no separate test split is constructed here.

        # Train - Subset of the dataset
        self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_dataset_index)
        self.train_dataset.index = [self.dataset.index[i] for i in self.train_dataset_index]

        # Validation - Subset of the dataset
        self.validation_dataset = EvaluationSubset(self.dataset, self.validation_dataset_index)
        self.validation_dataset.index = [self.dataset.index[i] for i in self.validation_dataset_index]

        # Validation - Subset of the dataset with more than 1 visit
        self.validation_dataset_with_morethan1_visit = EvaluationSubset(
            self.dataset,
            validation_dataset_with_morethan1_visit,
        )
        self.validation_dataset_with_morethan1_visit.index = [self.dataset.index[i] for i in
                                                        validation_dataset_with_morethan1_visit]

        if set(self.train_dataset.index).intersection(self.validation_dataset.index):
            raise ValueError('Training and validation participant IDs overlap.')

        self.split_manifest = {
            'policy': getattr(self.config, 'split_policy', 'historical_modulo_inclusive'),
            'threshold': self.config.percent_validation,
            'train_ids': [int(x) for x in self.train_dataset.index],
            'validation_ids': [int(x) for x in self.validation_dataset.index],
        }

        if self.fabric.global_rank == 0:
            print(f"Participants: training={len(self.train_dataset)}, validation={len(self.validation_dataset)}; "
                  f"multi-visit validation={len(self.validation_dataset_with_morethan1_visit)}")
        wandb.log({"misc/Train size": len(self.train_dataset)}, )
        wandb.log({"misc/Validation size": len(self.validation_dataset)}, )

    def compute_training_statistics(self):
        """Collect finite training values by modality, preserving participant order."""
        stats_names = list(self.bins)
        range_names = ([name for name in self.modality_name_to_number if name not in self.cat_cols]
                       if self.config.continuous_value_embeddings else [])
        values = defaultdict(list)
        names = list(dict.fromkeys(stats_names + range_names))
        for index in self.train_dataset.indices:
            conts = self.dataset.continuous_tokens[index].cpu()
            grouping_ids = self.dataset.modalities[index].cpu()
            for name in names:
                selected = conts[grouping_ids == self.modality_name_to_number[name]]
                values[name].extend(selected[torch.isfinite(selected)].tolist())

        stats = {}
        for name in stats_names:
            if len(values[name]) < 50:
                continue
            array = np.asarray(values[name], dtype=np.float64)
            stats[name] = {'mean': float(array.mean()), 'std': max(float(array.std(ddof=1)), 1e-8)}
        ranges = ({name: {'min': min(values[name]), 'max': max(values[name])}
                   for name in range_names if values[name]} if self.config.continuous_value_embeddings else None)
        return stats, ranges

    def init_dataloaders(self):
        loader_options = dict(pin_memory=True, num_workers=0)

        # Create generators for DataLoaders
        train_generator = torch.Generator()
        train_generator.manual_seed(self.config.seed)
        validation_generator = torch.Generator()
        validation_generator.manual_seed(self.config.seed)

        if self.config.train_sampler == 'eli_friendly':
            self.train_sampler = EliFriendlyDistributedSampler(
                self.train_dataset,
                batch_size=self.config.batch_per_gpu,
                num_replicas=self.fabric.world_size,
                rank=self.fabric.global_rank,
                seed=self.config.seed,
                lengths=[self.sequence_lengths[i] for i in self.train_dataset_index],
                multi_visit_rows={local for local, i in enumerate(self.train_dataset_index)
                                  if i in self.multi_visit_rows},
            )

        elif self.config.train_sampler == 'distributed':
            self.train_sampler = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=self.fabric.world_size,
                rank=self.fabric.global_rank,
                shuffle=True,
                seed=self.config.seed,
                drop_last=True,
            )
        else:
            raise ValueError(f"Unknown train_sampler type: {self.config['train_sampler']}")


        self.train_batch_sampler = ResumableBatchSampler(
            self.train_sampler,
            batch_size=self.config.batch_per_gpu,
            drop_last=True,
        )

        # Initialize Training DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            **loader_options,
            generator=train_generator
        )
        self.validation_sampler = ExactDistributedEvalSampler(
            self.validation_dataset,
            num_replicas=self.fabric.world_size,
            rank=self.fabric.global_rank,
        )
        # Initialize Validation DataLoader
        self.validation_dataloader = DataLoader(
            self.validation_dataset,
            batch_size=self.config.batch_per_gpu,
            sampler=self.validation_sampler,
            shuffle=False,
            drop_last=False,
            **loader_options,
            generator=validation_generator,
        )


        self.validation_morethan1_visit_sampler = ExactDistributedEvalSampler(
            self.validation_dataset_with_morethan1_visit,
            num_replicas=self.fabric.world_size,
            rank=self.fabric.global_rank,
        )
        # Initialize Validation DataLoader with more than 1 visit
        self.validation_dataloader_with_morethan1_visit = DataLoader(
            self.validation_dataset_with_morethan1_visit,
            batch_size=self.config.batch_per_gpu,
            sampler=self.validation_morethan1_visit_sampler,
            shuffle=False,
            drop_last=False,
            **loader_options,
            generator=validation_generator,
        )

        # wrap in fabric.setup_dataloaders
        self.validation_dataloader_with_morethan1_visit = self.fabric.setup_dataloaders(
            self.validation_dataloader_with_morethan1_visit,
            use_distributed_sampler=False,
        )

        wandb.log({"misc/Validation size with more than 1 visit": len(self.validation_dataset_with_morethan1_visit)}, )
