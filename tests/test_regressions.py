"""Focused regressions for the HealthFormer training implementation."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, Subset


ROOT = Path(__file__).resolve().parents[1]


from tests._helpers import load_implementation


implementation = load_implementation()


def make_dataset(**overrides):
    config = implementation.OmegaConf.create(
        {
            "random_modality_subset_chance": 0.0,
            "random_modality_exclusion_chance": 0.0,
            "chance_to_remove_time_series_data": 0.0,
            "random_modality_subset_fraction": 0.5,

            "max_seq_length": 8,

        }
    )
    kwargs = {
        "tokens": torch.tensor([[1, 2, 0]], dtype=torch.int16),
        "continuous_tokens": torch.zeros(1, 3),
        "bins": {},
        "modalities": torch.tensor([[0, 1, 2]], dtype=torch.int16),
        "time_expanded": torch.zeros(1, 3, 7, dtype=torch.int32),
        "modality_name_to_number": {"m0": 0, "m1": 1},
        "number_of_modalities": 2,
        "cat_cols": [],
        "index": [101],

        "mod2bininfo": {},
        "visit_marking": 1,
        "augmentation_chance": 0.0,
        "augmentation_rate": 0.0,
        "random_removal_rate": 0.0,
        "random_removal_chance": 0.0,
        "random_block_removal_rate": 0.0,
        "random_block_removal_chance": 0.0,
        "random_block_removal_number": 0,

        "augment_all": False,
        "config": config,
    }
    kwargs.update(overrides)
    return implementation.CustomDataset(**kwargs)


class ArtifactAndDatasetTests(unittest.TestCase):
    @staticmethod
    def valid_artifact():
        return {
            "tokens": torch.tensor([[0, 1], [1, 2]], dtype=torch.int16),
            "continuous_tokens": torch.zeros(2, 2),
            "modalities": torch.tensor([[0, 2], [1, 2]], dtype=torch.int16),
            "time_expanded": torch.zeros(2, 2, 7, dtype=torch.int16),
            "index": [101, np.int64(102)],
            "base_value_per_modality": {0: 1, 1: 1},
            "cum_base_value_per_modality": {0: 0, 1: 1},
            "modality_name_to_number": {"m0": 0, "m1": 1},
            "number_to_modality_name": {0: "m0", 1: "m1"},
            "vals_per_modality": {"m0": [0, 1], "m1": [0, 1]},
            "vocab_size": 3,
            "temporal_vocab_size": [8, 25, 61, 13, 147, 32, 2],
            "time_dim_order": [
                "day_of_week", "hour", "minute", "month", "year", "day_of_month", "sleep"
            ],
            "number_of_modalities": 2,
            "cat_cols": [],
            "bins": {},
        }

    def test_int16_modality_backing_is_preserved_but_rows_are_int32(self):
        dataset = make_dataset()
        _, _, returned_modalities, _ = dataset.__getitem__(0, augment=False)
        self.assertEqual(dataset.modalities.dtype, torch.int16)
        self.assertEqual(returned_modalities.dtype, torch.int32)
        self.assertEqual(returned_modalities.tolist(), [0, 1, 2])
        embedding = torch.nn.Embedding(3, 2)
        self.assertEqual(tuple(embedding(returned_modalities).shape), (3, 2))

    def test_int16_modalities_are_cast_before_indexed_augmentation(self):
        dataset = make_dataset(
            augment_all=True,

        )
        _, _, returned_modalities, _ = dataset.__getitem__(0, augment=True)
        self.assertEqual(dataset.modalities.dtype, torch.int16)
        self.assertEqual(returned_modalities.dtype, torch.int32)

    def test_participant_ids_must_be_integers(self):
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        trainer._validate_training_artifact(self.valid_artifact())
        for invalid_id in ("101", 101.0, True, np.bool_(False)):
            with self.subTest(invalid_id=invalid_id):
                artifact = self.valid_artifact()
                artifact["index"] = [invalid_id, 102]
                with self.assertRaisesRegex(TypeError, "index.*participant ID"):
                    trainer._validate_training_artifact(artifact)

class GenderTests(unittest.TestCase):
    def make_model(self, mapping, cumulative):
        model = implementation.MultiModalTransformer.__new__(implementation.MultiModalTransformer)
        torch.nn.Module.__init__(model)
        model.modality_name_to_number = mapping
        model.cum_base_value_per_modality = cumulative
        model.fabric = SimpleNamespace(global_rank=0)
        model.gender_positional_encoding = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
        model.gender_learned = torch.nn.Parameter(torch.zeros(2))
        model.age_positional_encoding = torch.zeros(1000, 2)
        model.age_learned = torch.nn.Parameter(torch.zeros(2))
        return model

    def assert_one_sided_gender_works(self, name):
        mapping = {"birth_event": 0, name: 1}
        model = self.make_model(mapping, {0: 0, 1: 10})
        tokens = torch.tensor([[1, 11]])
        modalities = torch.tensor([[0, 1, 2]])
        times = torch.zeros(1, 3, 7, dtype=torch.long)
        result = model._build_demographic_embedding(
            tokens, modalities, times, 2, torch.zeros(1, 2, 2)
        )
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.equal(result, torch.zeros_like(result)))

    def test_gender_only_schema(self):
        self.assert_one_sided_gender_works("gender")

    def test_gender_at_birth_only_schema(self):
        self.assert_one_sided_gender_works("gender_at_birth")


class IndexedDataset(Dataset):
    def __init__(self, participant_ids):
        self.index = list(participant_ids)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        return int(index)


def make_indexed_subset(participant_ids, indices):
    dataset = IndexedDataset(participant_ids)
    subset = Subset(dataset, list(indices))
    subset.index = [dataset.index[index] for index in subset.indices]
    return subset


class ModalityBoundaryTests(unittest.TestCase):
    def test_noise_keeps_shared_inclusive_endpoint(self):
        for min_token, max_token in ((10, 12), (32766, 32767)):
            with self.subTest(max_token=max_token):
                dataset = make_dataset(
                    tokens=torch.tensor([[max_token]], dtype=torch.int16),
                    continuous_tokens=torch.zeros(1, 1),
                    modalities=torch.zeros(1, 1, dtype=torch.int16),
                    time_expanded=torch.zeros(1, 1, 7, dtype=torch.int32),
                    modality_name_to_number={"m": 0},
                    number_of_modalities=1,
                    index=[101],
                    visit_marking=0,
                    augmentation_rate=1.0,
                    bins={"m": list(range(max_token - min_token + 2))},
                    mod2bininfo={"m": (list(range(min_token, max_token + 1)), [])},
                )
                tokens = dataset.tokens[0].clone()
                with mock.patch.object(implementation.torch, "randperm", return_value=torch.tensor([0])), \
                        mock.patch.object(implementation.torch, "rand", return_value=torch.ones(1)), \
                        mock.patch.object(
                            implementation.torch,
                            "randint",
                            return_value=torch.ones(1, dtype=torch.long),
                        ) as randint:
                    augmented, _ = dataset.augment_noise(
                        dataset.modalities[0].int(),
                        tokens,
                        dataset.continuous_tokens[0].clone(),
                    )
                self.assertEqual(augmented.item(), max_token)
                self.assertNotIn("dtype", randint.call_args.kwargs)

    def make_metric_model(self):
        model = implementation.MultiModalTransformer.__new__(implementation.MultiModalTransformer)
        torch.nn.Module.__init__(model)
        model.vals_per_modality = {"m0": [0, 1], "m1": [0, 1]}
        model.modality_name_to_number = {"m0": 0, "m1": 1}
        model.cum_base_value_per_modality = {0: 0, 1: 1}
        model.mask_modality_val = 2
        return model


class AugmentationAndScheduleTests(unittest.TestCase):
    def test_uniform_modality_subset_is_unique_and_excludes_visit_marker(self):
        config = implementation.OmegaConf.create(
            {
                "random_modality_subset_chance": 1.0,
                "random_modality_exclusion_chance": 0.0,
                "chance_to_remove_time_series_data": 0.0,
                "random_modality_subset_fraction": 0.67,

                "max_seq_length": 4,

            }
        )
        dataset = make_dataset(
            tokens=torch.tensor([[1, 2, 3, 0]], dtype=torch.int16),
            continuous_tokens=torch.zeros(1, 4),
            modalities=torch.tensor([[0, 1, 2, 3]], dtype=torch.int16),
            time_expanded=torch.zeros(1, 4, 7, dtype=torch.int32),
            modality_name_to_number={"m0": 0, "visit": 1, "m2": 2},
            number_of_modalities=3,
            index=[101],

            visit_marking=1,
            config=config,
        )
        with mock.patch.object(implementation.random, "sample", wraps=__import__('random').sample) as choose:
            _, _, modalities, _ = dataset.random_modality_subset_and_pad(
                dataset.tokens[0].clone(), dataset.continuous_tokens[0].clone(),
                dataset.modalities[0].int().clone(), dataset.time_expanded[0].clone())
        self.assertEqual(choose.call_args.args, ([0, 2], 2))
        self.assertEqual(set(modalities[modalities != 3].tolist()), {0, 1, 2})

    def test_first_warmup_update_and_endpoint_use_expected_learning_rates(self):
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        trainer.max_lr = 1.0
        trainer.min_lr = 0.1
        trainer.warmup_steps = 4
        trainer.max_steps = 10
        self.assertEqual(trainer.get_lr(1), 0.25)
        self.assertEqual(trainer.get_lr(4), 1.0)
        self.assertEqual(trainer.get_lr(10), 0.1)


class EvaluationTests(unittest.TestCase):
    def test_exact_sampler_covers_each_sample_once_even_with_empty_rank(self):
        for size, world_size in ((5, 2), (6, 4), (1, 2)):
            with self.subTest(size=size, world_size=world_size):
                dataset = IndexedDataset(range(100, 100 + size))
                rank_indices = [
                    list(implementation.ExactDistributedEvalSampler(dataset, world_size, rank))
                    for rank in range(world_size)
                ]
                flattened = [index for indices in rank_indices for index in indices]
                self.assertEqual(sorted(flattened), list(range(size)))
                self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(
            len(implementation.ExactDistributedEvalSampler(IndexedDataset([100]), 2, 1)),
            0,
        )

    def test_validation_rename_preserves_historical_boundary_membership(self):
        ids = [100, 119, 120, 121, 199, 220]
        dataset = make_dataset(
            tokens=torch.tensor([[1, 2, 0]] * len(ids), dtype=torch.int16),
            continuous_tokens=torch.zeros(len(ids), 3),
            modalities=torch.tensor([[1, 1, 2]] * len(ids), dtype=torch.int16),
            time_expanded=torch.zeros(len(ids), 3, 7, dtype=torch.int32),
            index=ids, augment_all=True, visit_marking=1,
        )
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        trainer.config = SimpleNamespace(percent_validation=20)
        trainer.fabric = SimpleNamespace(global_rank=0)
        trainer.dataset = dataset
        trainer.multi_visit_rows = set(range(len(ids)))
        trainer.visit_marking = 1
        with mock.patch.object(implementation.wandb, "log"), \
                contextlib.redirect_stdout(io.StringIO()):
            trainer._build_dataset_views()
        self.assertEqual(trainer.validation_dataset.index, [100, 119, 120, 220])
        self.assertEqual(trainer.train_dataset.index, [121, 199])
        self.assertEqual(trainer.validation_dataset_with_morethan1_visit.index, [100, 119, 120, 220])

    def test_eval_dataset_and_dedicated_loader_generator_preserve_global_rng(self):
        dataset = make_dataset(
            tokens=torch.tensor([[1, 2, 0], [1, 2, 0]], dtype=torch.int16),
            continuous_tokens=torch.zeros(2, 3),
            modalities=torch.tensor([[1, 1, 2], [1, 1, 2]], dtype=torch.int16),
            time_expanded=torch.zeros(2, 3, 7, dtype=torch.int32),
            index=[101, 150],
            augment_all=True,
            visit_marking=1,
        )
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        trainer.config = SimpleNamespace(percent_validation=20)
        trainer.fabric = SimpleNamespace(global_rank=0)
        trainer.dataset = dataset
        trainer.multi_visit_rows = {0, 1}
        trainer.visit_marking = 1
        with mock.patch.object(implementation.wandb, "log"), \
                contextlib.redirect_stdout(io.StringIO()):
            trainer._build_dataset_views()
        self.assertTrue(trainer.train_dataset.dataset.augment_all)
        self.assertTrue(trainer.validation_dataset.dataset.augment_all)
        self.assertTrue(trainer.validation_dataset_with_morethan1_visit.dataset.augment_all)
        self.assertIsInstance(trainer.validation_dataset, implementation.EvaluationSubset)
        self.assertIsInstance(
            trainer.validation_dataset_with_morethan1_visit,
            implementation.EvaluationSubset,
        )
        self.assertIs(
            trainer.validation_dataset_with_morethan1_visit.dataset,
            trainer.validation_dataset.dataset,
        )

        torch.manual_seed(123)
        before_sample = torch.random.get_rng_state().clone()
        trainer.validation_dataset[0]
        self.assertTrue(torch.equal(before_sample, torch.random.get_rng_state()))

        self.assertIs(trainer.train_dataset.dataset, dataset)
        self.assertIs(trainer.validation_dataset.dataset, dataset)
        original = [getattr(dataset, field).clone() for field in
                    ('tokens', 'continuous_tokens', 'modalities', 'time_expanded')]
        before_training_sample = torch.random.get_rng_state().clone()
        dataset.augmentation_chance = 1.0
        def mutate_noise(modalities, tokens, continuous):
            tokens.fill_(3); continuous.fill_(99.)
            return tokens, continuous
        with mock.patch.object(dataset, 'augment_noise', side_effect=mutate_noise):
            augmented = trainer.train_dataset[0]
        self.assertTrue((augmented[0] == 3).all())
        self.assertTrue((augmented[1] == 99.).all())
        self.assertFalse(
            torch.equal(before_training_sample, torch.random.get_rng_state())
        )

        for field, expected in zip(('tokens', 'continuous_tokens', 'modalities', 'time_expanded'), original):
            torch.testing.assert_close(getattr(dataset, field), expected, rtol=0, atol=0)

        loader_generator = torch.Generator().manual_seed(42)
        loader = torch.utils.data.DataLoader(
            IndexedDataset([100, 101]),
            batch_size=1,
            sampler=implementation.ExactDistributedEvalSampler(IndexedDataset([100, 101]), 1, 0),
            generator=loader_generator,
        )
        before_loader = torch.random.get_rng_state().clone()
        next(iter(loader))
        self.assertTrue(torch.equal(before_loader, torch.random.get_rng_state()))

    def test_validation_loaders_are_wired_for_exact_non_dropping_evaluation(self):
        class RecordingFabric:
            world_size = 1
            global_rank = 0

            def __init__(self):
                self.setup_calls = []

            def setup_dataloaders(self, loader, use_distributed_sampler=True):
                self.setup_calls.append((loader, use_distributed_sampler))
                return loader

        participant_ids = list(range(100, 108))
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        trainer.config = implementation.OmegaConf.create(
            {"train_sampler": "distributed", "batch_per_gpu": 2,  "seed": 42}
        )
        trainer.fabric = RecordingFabric()
        trainer.visit_marking = 1
        trainer.train_dataset = make_indexed_subset(participant_ids, [0, 1, 2, 3])
        trainer.validation_dataset = make_indexed_subset(participant_ids, [4, 5, 6])
        trainer.validation_dataset_with_morethan1_visit = make_indexed_subset(
            participant_ids, [4, 5]
        )

        with mock.patch.object(implementation.wandb, "log"), contextlib.redirect_stdout(io.StringIO()):
            trainer.init_dataloaders()

        self.assertIsInstance(trainer.validation_dataloader.sampler, implementation.ExactDistributedEvalSampler)
        self.assertIsInstance(
            trainer.validation_dataloader_with_morethan1_visit.sampler,
            implementation.ExactDistributedEvalSampler,
        )
        self.assertFalse(trainer.validation_dataloader.drop_last)
        self.assertFalse(trainer.validation_dataloader_with_morethan1_visit.drop_last)
        self.assertIsNotNone(trainer.validation_dataloader.generator)
        self.assertIsNotNone(trainer.validation_dataloader_with_morethan1_visit.generator)
        self.assertFalse(trainer.fabric.setup_calls[-1][1])

        source = (ROOT / 'healthformer' / 'trainer.py').read_text()
        regular_setup = source[source.index("self.train_dataloader = self.fabric.setup_dataloaders("):]
        regular_setup = regular_setup[:regular_setup.index("self.max_steps =")]
        self.assertIn("self.validation_dataloader", regular_setup)
        self.assertIn("use_distributed_sampler=False", regular_setup)

    def test_distributed_training_sampler_uses_fabric_rank(self):
        participant_ids = list(range(100, 108))
        orders = []
        for rank in (0, 1):
            trainer = implementation.Trainer.__new__(implementation.Trainer)
            trainer.config = implementation.OmegaConf.create(
                {'train_sampler': 'distributed', 'batch_per_gpu': 2, 'seed': 42})
            trainer.fabric = SimpleNamespace(world_size=2, global_rank=rank,
                                            setup_dataloaders=lambda loader, **kwargs: loader)
            dataset = make_indexed_subset(participant_ids, list(range(8)))
            trainer.train_dataset = trainer.validation_dataset = dataset
            trainer.validation_dataset_with_morethan1_visit = dataset
            with mock.patch.object(implementation.wandb, 'log'):
                trainer.init_dataloaders()
            self.assertEqual(trainer.train_sampler.num_replicas, 2)
            self.assertEqual(trainer.train_sampler.rank, rank)
            trainer.epoch = 0
            trainer.current_batch_index = 0
            saved_cursor = trainer._active_train_cursor_checkpoint_state()
            self.assertEqual(saved_cursor['signature']['distributed_partitioning_version'], 2)
            for previous_version in (None, 1):
                old_signature = dict(saved_cursor['signature'])
                if previous_version is None:
                    del old_signature['distributed_partitioning_version']
                else:
                    old_signature['distributed_partitioning_version'] = previous_version
                with self.assertRaisesRegex(RuntimeError, 'distributed_partitioning_version'):
                    trainer._restore_active_train_cursor({'current_batch_index': 0,
                        'train_cursor_state': {**saved_cursor, 'signature': old_signature}})
                self.assertEqual(trainer.train_batch_sampler.next_batch_index, 0)
            trainer._restore_active_train_cursor({'current_batch_index': 0,
                                                  'train_cursor_state': saved_cursor})
            self.assertIs(trainer.validation_dataloader.generator,
                          trainer.validation_dataloader_with_morethan1_visit.generator)
            self.assertIsNot(trainer.train_dataloader.generator, trainer.validation_dataloader.generator)
            orders.append(list(trainer.train_sampler))
        self.assertFalse(set(orders[0]) & set(orders[1]))
        self.assertEqual(sorted(orders[0] + orders[1]), list(range(8)))

    def test_distributed_sampler_drops_remainder_without_repeating_participants(self):
        for population in (1, 2, 5, 7, 8, 11, 17):
            for world_size in (1, 2, 3, 4):
                for batch_size in (1, 2, 3):
                    with self.subTest(population=population, world=world_size, batch=batch_size):
                        dataset = make_indexed_subset(list(range(100, 100 + population)),
                                                      list(range(population)))
                        for epoch in (0, 1):
                            orders = []
                            for rank in range(world_size):
                                trainer = implementation.Trainer.__new__(implementation.Trainer)
                                trainer.config = implementation.OmegaConf.create({
                                    'train_sampler': 'distributed', 'batch_per_gpu': batch_size, 'seed': 42})
                                trainer.fabric = SimpleNamespace(world_size=world_size, global_rank=rank,
                                    setup_dataloaders=lambda loader, **kwargs: loader)
                                trainer.train_dataset = trainer.validation_dataset = dataset
                                trainer.validation_dataset_with_morethan1_visit = dataset
                                with mock.patch.object(implementation.wandb, 'log'):
                                    trainer.init_dataloaders()
                                cursor = trainer.train_batch_sampler
                                cursor.set_epoch(epoch)
                                batches = list(cursor)
                                self.assertTrue(all(len(batch) == batch_size for batch in batches))
                                order = [index for batch in batches for index in batch]
                                self.assertEqual(len(order), population // world_size // batch_size * batch_size)
                                if batches:
                                    cursor.mark_consumed(1)
                                    saved = cursor.state_dict()
                                    fresh = implementation.ResumableBatchSampler(trainer.train_sampler,
                                                                                 batch_size, drop_last=True)
                                    fresh.load_state_dict(saved)
                                    self.assertEqual(list(fresh), batches[1:])
                                orders.extend(order)
                            self.assertEqual(len(orders), len(set(orders)))
                            self.assertTrue(set(orders).issubset(range(population)))

    def test_ddp_configuration_disables_buffer_broadcasts(self):
        source = (ROOT / 'healthformer' / 'launch.py').read_text()
        self.assertIn("broadcast_buffers=False", source)


class CursorIdentityTests(unittest.TestCase):
    @staticmethod
    def make_trainer(subset, *, rank=0, world_size=1):
        cursor = implementation.ResumableBatchSampler(
            DistributedSampler(
                subset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=23,
                drop_last=False,
            ),
            batch_size=2,
            drop_last=False,
        )
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        trainer.config = {}
        trainer.device = torch.device("cpu")
        trainer.fabric = SimpleNamespace(world_size=world_size)
        trainer.epoch = 3
        trainer.current_batch_index = 0
        trainer.train_batch_sampler = cursor
        return trainer, cursor

    def assert_restore_rejected(self, saved, subset):
        trainer, _ = self.make_trainer(subset)
        with self.assertRaisesRegex(RuntimeError, "structurally incompatible"):
            trainer._restore_active_train_cursor(
                {"current_batch_index": 0, "train_cursor_state": saved}
            )

    def test_same_length_different_subset_or_participants_is_rejected(self):
        participant_ids = list(range(100, 108))
        source_subset = make_indexed_subset(participant_ids, [0, 1, 2, 3])
        source_trainer, source_cursor = self.make_trainer(source_subset)
        source_cursor.set_epoch(3)
        saved = source_trainer._active_train_cursor_checkpoint_state()

        self.assert_restore_rejected(
            saved, make_indexed_subset(participant_ids, [4, 5, 6, 7])
        )
        self.assert_restore_rejected(
            saved, make_indexed_subset(list(range(200, 208)), [0, 1, 2, 3])
        )

    def test_identical_dataset_identity_restores_across_ranks(self):
        subset = make_indexed_subset(list(range(100, 108)), [0, 1, 2, 3, 4])
        saving_trainer, saving_cursor = self.make_trainer(subset, rank=0, world_size=2)
        saving_cursor.set_epoch(3)
        saving_cursor.mark_consumed(1)
        saving_trainer.current_batch_index = 1
        saved = saving_trainer._active_train_cursor_checkpoint_state()

        restoring_trainer, restoring_cursor = self.make_trainer(
            make_indexed_subset(list(range(100, 108)), [0, 1, 2, 3, 4]),
            rank=1,
            world_size=2,
        )
        restoring_trainer._restore_active_train_cursor(
            {"current_batch_index": 1, "train_cursor_state": saved}
        )
        self.assertEqual(restoring_cursor.next_batch_index, 1)


class CurrentCheckpointMetadataTests(unittest.TestCase):
    def test_old_or_missing_cursor_is_rejected(self):
        subset = make_indexed_subset(list(range(100, 108)), [0, 1, 2, 3])
        trainer, cursor = CursorIdentityTests.make_trainer(subset)
        cursor.set_epoch(3)
        saved = trainer._active_train_cursor_checkpoint_state()
        saved['version'] = 1
        with self.assertRaisesRegex(RuntimeError, 'Unsupported train cursor'):
            trainer._restore_active_train_cursor({'current_batch_index': 0, 'train_cursor_state': saved})
        with self.assertRaisesRegex(TypeError, 'train_cursor_state'):
            trainer._restore_active_train_cursor({'current_batch_index': 0})

    def test_missing_accumulation_metadata_is_rejected(self):
        trainer = CheckpointBoundaryTests.make_trainer(accumulation=1)
        with self.assertRaisesRegex(RuntimeError, 'requires accumulation_state'):
            trainer._validate_accumulation_resume_state({})


class _StrictCursor:
    STATE_VERSION = 1

    def __init__(self, total_batches):
        self.total_batches = total_batches
        self.next_batch_index = 0
        self.epoch = 0
        self.consumed = []

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def mark_consumed(self, expected_absolute_index):
        expected_absolute_index = int(expected_absolute_index)
        if expected_absolute_index != self.next_batch_index + 1:
            raise RuntimeError("cursor did not advance by exactly one")
        self.next_batch_index = expected_absolute_index
        self.consumed.append(expected_absolute_index)

    def finish_epoch(self):
        if self.next_batch_index != self.total_batches:
            raise RuntimeError("incomplete cursor")
        self.next_batch_index = 0

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "epoch": self.epoch,
            "next_batch_index": self.next_batch_index,
        }

    def load_state_dict(self, state):
        if self.next_batch_index != 0:
            raise RuntimeError("cannot restore over in-memory progress")
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise RuntimeError("unsupported cursor version")
        restored_index = int(state["next_batch_index"])
        if not 0 <= restored_index <= self.total_batches:
            raise RuntimeError("cursor outside valid range")
        self.set_epoch(state["epoch"])
        self.next_batch_index = restored_index

    def structural_metadata(self):
        return {
            "sampler_class": "test.StrictCursor",
            "sampler_seed": 0,
            "dataset_length": self.total_batches,
            "batch_size": 1,
            "batch_drop_last": False,
            "total_batches": self.total_batches,
            "dataset_view_indices_sha256": "test-view",
            "dataset_participant_ids_sha256": "test-participants",
            "backing_participant_ids_sha256": "test-backing",
        }


class _TinyBranchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(1.0))
        self.elis_only = torch.nn.Parameter(torch.tensor(1.0))
        self.standard_only = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, scale, stage):
        if stage == "ELIS":
            return scale * (self.shared + self.elis_only)
        if stage == "STANDARD":
            return scale * (2.0 * self.shared + self.standard_only)
        raise ValueError(stage)


class _RecordingFabric:
    def autocast(self):
        return contextlib.nullcontext()

    global_rank = 0
    world_size = 1

    def __init__(self):
        self.events = []
        self.pending_stage = None
        self.clip_gradients_seen = []
        self.saved = []

    def all_reduce(self, tensor, reduce_op):
        return tensor

    def broadcast(self, value, src):
        return value

    @contextlib.contextmanager
    def no_backward_sync(self, model, enabled):
        self.events.append(("enter_no_sync", bool(enabled)))
        try:
            yield
        finally:
            self.events.append(("exit_no_sync", bool(enabled)))

    def backward(self, loss):
        self.events.append((f"{self.pending_stage}_backward", None))
        loss.backward()

    def clip_gradients(self, model, optimizer, max_norm, norm_type):
        self.clip_gradients_seen.append([
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ])
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=norm_type)

    def barrier(self):
        return None

    def save(self, filename, state):
        self.saved.append((filename, state))


class _FileRecordingFabric(_RecordingFabric):
    def save(self, filename, state):
        super().save(filename, state)
        torch.save(state, filename)


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.gradient_windows = []
        self.parameter_snapshots = []
        self.learning_rates = []

    def step(self, closure=None):
        self.learning_rates.append(self.param_groups[0]["lr"])
        self.gradient_windows.append([
            None if parameter.grad is None else parameter.grad.detach().clone()
            for group in self.param_groups for parameter in group["params"]
        ])
        result = super().step(closure)
        self.parameter_snapshots.append([
            parameter.detach().clone()
            for group in self.param_groups for parameter in group["params"]
        ])
        return result


def _training_batches(count):
    batches = []
    for batch_index in range(count):
        batches.append((
            torch.tensor([[batch_index + 1, batch_index + 2]], dtype=torch.int64),
            torch.zeros(1, 2),
            torch.zeros(1, 2, dtype=torch.int64),
            torch.zeros(1, 2, 7, dtype=torch.int64),
        ))
    return batches


def _make_training_harness(
        batch_count,
        accumulation,
        *,
        run_elis=True,
        standard_nonfinite_step=None,
        elis_oom_step=None,
):
    trainer = implementation.Trainer.__new__(implementation.Trainer)
    trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
    trainer.fabric = _RecordingFabric()
    trainer.device = torch.device("cpu")
    trainer.model = _TinyBranchModel()
    trainer.optimizer = _RecordingSGD(trainer.model.parameters(), lr=0.1)
    trainer.config = SimpleNamespace(
        evaluate=False,
        precision='32-true',
            max_seq_length=64,
        reduce_context_in_train=False,

        accumulate_grad_batches=accumulation,
        Elis_augmentation_prob=1.0 if run_elis else 0.0,
        elis_loss_weight=0.01,
        not_elis_loss_weight=1.0,

        max_grad_norm=1e9,
        log_wandb_metric_every_k=1000,
        allow_wandb=True,

    )
    trainer.pending_microbatches = 0
    trainer.pending_window_loss = 0.0
    trainer.epoch = 0
    trainer.it = 0
    trainer.current_batch_index = 0
    trainer.mask_modality_val = 99
    trainer.vals_per_modality = {}
    trainer.max_lr = 0.1
    trainer.min_lr = 0.1
    trainer.warmup_steps = 0
    trainer.max_steps = max(batch_count, 1)

    cursor = _StrictCursor(batch_count)
    loader = _training_batches(batch_count)
    trainer._active_train_cursor = lambda: cursor
    trainer._active_train_dataloader = lambda: loader[cursor.next_batch_index:]

    def compute_elis_loss(self, *args, **kwargs):
        if self.training_step == elis_oom_step:
            raise torch.OutOfMemoryError("CUDA out of memory")
        scale = torch.tensor(float(self.training_step + 1))
        self.fabric.events.append(("ELIS_forward", None))
        self.fabric.pending_stage = "ELIS"
        loss = self.model(scale, stage="ELIS")
        return loss, True

    def compute_standard_loss(self, *args, **kwargs):
        scale = torch.tensor(float(self.training_step + 1))
        self.fabric.events.append(("STANDARD_forward", None))
        self.fabric.pending_stage = "STANDARD"
        loss = self.model(scale, stage="STANDARD")
        if self.training_step == standard_nonfinite_step:
            loss = loss * torch.tensor(float("nan"))
        return loss, {}

    trainer.compute_elis_loss = MethodType(compute_elis_loss, trainer)
    trainer.compute_standard_loss = MethodType(compute_standard_loss, trainer)
    return trainer, cursor


class AccumulationTests(unittest.TestCase):
    def run_epoch(self, *args, **kwargs):
        trainer, cursor = _make_training_harness(*args, **kwargs)
        logs = []
        with mock.patch.object(implementation.wandb, "log", side_effect=logs.append), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            trainer.train_epoch()
        return trainer, cursor, logs

    def test_unrelated_runtime_errors_are_not_treated_as_oom(self):
        for stage in ('compute_elis_loss', 'compute_standard_loss'):
            for message in ('allocation metadata mismatch', 'invalid shape 481', 'CUDA out of memory'):
                trainer, _ = _make_training_harness(1, 1)
                with mock.patch.object(trainer, stage, side_effect=RuntimeError(message)), \
                        mock.patch.object(implementation.wandb, 'log'), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, message):
                        trainer.train_epoch()
                self.assertEqual(len(trainer.optimizer.gradient_windows), 0)

    def test_accumulation_configuration_validation(self):
        for valid in (1, 4, np.int64(8)):
            self.assertEqual(implementation.Trainer._validate_accumulate_grad_batches(valid), int(valid))
        for invalid in (0, -1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "accumulate_grad_batches"):
                implementation.Trainer._validate_accumulate_grad_batches(invalid)
        for invalid in (True, False, 1.0, 4.0, "4", None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(TypeError, "accumulate_grad_batches"):
                implementation.Trainer._validate_accumulate_grad_batches(invalid)

    def test_optimizer_step_count_uses_epoch_ceiling(self):
        cases = {
            (5, 4): 2,
            (8, 4): 2,
            (3, 8): 1,
            (1, 1): 1,
            (0, 4): 0,
        }
        for arguments, expected in cases.items():
            with self.subTest(arguments=arguments):
                self.assertEqual(implementation.Trainer._optimizer_steps_per_epoch(*arguments), expected)
        self.assertEqual(implementation.Trainer._optimizer_steps_per_epoch(5, 4) * 2, 4)

    def test_equal_warmup_and_horizon_has_defined_endpoint(self):
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        trainer.max_lr = 0.2
        trainer.min_lr = 0.02
        trainer.warmup_steps = 4
        trainer.max_steps = 4
        self.assertAlmostEqual(trainer.get_lr(3), 0.15)
        self.assertAlmostEqual(trainer.get_lr(4), 0.2)
        self.assertAlmostEqual(trainer.get_lr(5), 0.02)

    def test_gradient_normalization_uses_actual_count_and_skips_none(self):
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        trainer.model = torch.nn.Module()
        trainer.model.first = torch.nn.Linear(1, 1, bias=False)
        trainer.model.second = torch.nn.Linear(1, 1, bias=False)
        trainer.model.first.weight.grad = torch.tensor([[6.0]])
        trainer.model.second.weight.grad = None
        trainer.config = SimpleNamespace()
        trainer._normalize_accumulated_gradients(3)
        self.assertEqual(trainer.model.first.weight.grad.item(), 2.0)
        self.assertIsNone(trainer.model.second.weight.grad)
        with self.assertRaises(ValueError):
            trainer._normalize_accumulated_gradients(0)

    def test_full_and_tail_windows_use_mean_gradients(self):
        trainer, cursor, logs = self.run_epoch(5, 4)
        self.assertEqual(trainer.it, 2)
        self.assertEqual(cursor.consumed, [1, 2, 3, 4, 5])
        self.assertEqual(trainer.pending_microbatches, 0)
        self.assertEqual(len(trainer.optimizer.gradient_windows), 2)
        self.assertEqual(len(trainer.optimizer.learning_rates), 2)

        first, tail = trainer.optimizer.gradient_windows
        self.assertAlmostEqual(first[0].item(), 2.01 * 2.5, places=6)
        self.assertAlmostEqual(first[1].item(), 0.01 * 2.5, places=6)
        self.assertAlmostEqual(first[2].item(), 2.5, places=6)
        self.assertAlmostEqual(tail[0].item(), 2.01 * 5.0, places=6)
        self.assertAlmostEqual(tail[1].item(), 0.01 * 5.0, places=6)
        self.assertAlmostEqual(tail[2].item(), 5.0, places=6)
        for before_clip, at_step in zip(trainer.fabric.clip_gradients_seen,
                                        trainer.optimizer.gradient_windows):
            for clipped, stepped in zip(before_clip, at_step):
                self.assertTrue(torch.equal(clipped, stepped))

        window_sizes = [
            log["Training Accumulation/Window Size"]
            for log in logs if "Training Accumulation/Window Size" in log
        ]
        self.assertEqual(window_sizes, [4, 1])
        self.assertFalse(any(
            any(key.endswith("loss scaled") for key in log)
            for log in logs
        ))

    def test_short_epoch_and_exact_windows_step_correctly(self):
        short, _, short_logs = self.run_epoch(3, 8)
        exact, _, exact_logs = self.run_epoch(8, 4)
        self.assertEqual(short.it, 1)
        self.assertEqual(exact.it, 2)
        self.assertEqual(
            [log["Training Accumulation/Window Size"] for log in short_logs
             if "Training Accumulation/Window Size" in log],
            [3],
        )
        self.assertEqual(
            [log["Training Accumulation/Window Size"] for log in exact_logs
             if "Training Accumulation/Window Size" in log],
            [4, 4],
        )

    def test_empty_epoch_fails_without_optimizer_updates(self):
        with self.assertRaisesRegex(RuntimeError, 'no optimizer updates'):
            self.run_epoch(0, 4)

    def test_elis_and_standard_are_forward_backward_paired(self):
        trainer, _, _ = self.run_epoch(1, 1)
        self.assertEqual(trainer.fabric.events[:8], [
            ("enter_no_sync", True),
            ("ELIS_forward", None),
            ("ELIS_backward", None),
            ("exit_no_sync", True),
            ("enter_no_sync", False),
            ("STANDARD_forward", None),
            ("STANDARD_backward", None),
            ("exit_no_sync", False),
        ])
        gradients = trainer.optimizer.gradient_windows[0]
        self.assertAlmostEqual(gradients[0].item(), 2.01, places=6)
        self.assertAlmostEqual(gradients[1].item(), 0.01, places=6)
        self.assertAlmostEqual(gradients[2].item(), 1.0, places=6)

    def test_a1_standard_only_uses_the_unscaled_standard_gradient(self):
        trainer, _, _ = self.run_epoch(1, 1, run_elis=False)
        gradients = trainer.optimizer.gradient_windows[0]
        self.assertAlmostEqual(gradients[0].item(), 2.0, places=6)
        self.assertIsNone(gradients[1])
        self.assertAlmostEqual(gradients[2].item(), 1.0, places=6)
        event_names = [event[0] for event in trainer.fabric.events]
        self.assertNotIn("ELIS_forward", event_names)
        self.assertNotIn("ELIS_backward", event_names)

    def test_train_epoch_preserves_next_token_and_metadata_alignment_inputs(self):
        trainer, cursor = _make_training_harness(1, 1, run_elis=False)
        tokens = torch.tensor([[10, 20, 30, 40]], dtype=torch.int64)
        continuous = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        modalities = torch.tensor([[5, 6, 7, 8]], dtype=torch.int64)
        times = torch.arange(28, dtype=torch.int64).reshape(1, 4, 7)
        batch = (tokens, continuous, modalities, times)
        captured = {}
        trainer._active_train_dataloader = lambda: [batch]

        def capture_standard(self, tokens_in, continuous_tokens, targets,
                             modality_data, time_data):
            captured.update({
                "tokens_in": tokens_in.detach().clone(),
                "continuous": continuous_tokens.detach().clone(),
                "targets": targets.detach().clone(),
                "modalities": modality_data.detach().clone(),
                "times": time_data.detach().clone(),
            })
            self.fabric.events.append(("STANDARD_forward", None))
            self.fabric.pending_stage = "STANDARD"
            return self.model(torch.tensor(1.0), stage="STANDARD"), {}

        trainer.compute_standard_loss = MethodType(capture_standard, trainer)
        with mock.patch.object(implementation.wandb, "log"), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            trainer.train_epoch()

        self.assertEqual(cursor.consumed, [1])
        self.assertTrue(torch.equal(captured["tokens_in"], tokens[:, :-1]))
        self.assertTrue(torch.equal(captured["targets"], tokens[:, 1:]))
        self.assertTrue(torch.equal(captured["continuous"], continuous))
        self.assertTrue(torch.equal(captured["modalities"], modalities))
        self.assertTrue(torch.equal(captured["times"], times))

    def test_standard_rejection_discards_partial_window(self):
        trainer, cursor, logs = self.run_epoch(5, 4, standard_nonfinite_step=2)
        self.assertEqual(cursor.consumed, [1, 2, 3, 4, 5])
        self.assertEqual(trainer.it, 1)
        self.assertEqual(trainer.pending_microbatches, 0)
        self.assertEqual(len(trainer.optimizer.gradient_windows), 1)
        gradients = trainer.optimizer.gradient_windows[0]
        self.assertAlmostEqual(gradients[0].item(), 2.01 * 4.5, places=6)
        self.assertAlmostEqual(gradients[1].item(), 0.01 * 4.5, places=6)
        self.assertAlmostEqual(gradients[2].item(), 4.5, places=6)
        self.assertEqual(
            [log["Training Accumulation/Window Size"] for log in logs
             if "Training Accumulation/Window Size" in log],
            [2],
        )
        logged_mean = [
            log["Training Loss/"] for log in logs if "Training Loss/" in log
        ]
        self.assertEqual(len(logged_mean), 1)
        self.assertAlmostEqual(logged_mean[0], 3.02 * 4.5, places=6)

    def test_elis_oom_propagates_without_an_optimizer_step(self):
        trainer, cursor = _make_training_harness(5, 4, elis_oom_step=2)
        with self.assertRaises(torch.OutOfMemoryError), mock.patch.object(implementation.wandb, 'log'):
            trainer.train_epoch()
        self.assertEqual(cursor.consumed, [1, 2])
        self.assertEqual(trainer.it, 0)

    def test_rejected_tail_reports_no_accepted_updates(self):
        trainer, cursor = _make_training_harness(3, 4, standard_nonfinite_step=2)
        with self.assertRaisesRegex(RuntimeError, 'no optimizer updates'), mock.patch.object(implementation.wandb, 'log'):
            trainer.train_epoch()
        self.assertEqual(cursor.consumed, [1, 2, 3])
        self.assertEqual(trainer.it, 0)
        self.assertEqual(trainer.pending_microbatches, 0)
        self.assertEqual(len(trainer.optimizer.gradient_windows), 0)
        self.assertTrue(all(parameter.grad is None for parameter in trainer.model.parameters()))

    def test_boundary_resume_matches_uninterrupted_training(self):
        uninterrupted, _, _ = self.run_epoch(5, 4)

        interrupted, interrupted_cursor = _make_training_harness(5, 4)

        class BoundaryReached(Exception):
            pass

        commit_batch = interrupted._mark_active_train_batch_consumed

        def stop_at_first_boundary(next_batch):
            commit_batch(next_batch)
            if interrupted_cursor.next_batch_index == 4:
                raise BoundaryReached

        interrupted._mark_active_train_batch_consumed = stop_at_first_boundary
        with mock.patch.object(implementation.wandb, "log"), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(BoundaryReached):
            interrupted.train_epoch()
        self.assertEqual(interrupted_cursor.next_batch_index, 4)
        self.assertEqual(interrupted.pending_microbatches, 0)
        self.assertTrue(all(parameter.grad is None for parameter in interrupted.model.parameters()))

        resumed, resumed_cursor = _make_training_harness(5, 4)
        resumed.model.load_state_dict(interrupted.model.state_dict())
        resumed.optimizer.load_state_dict(interrupted.optimizer.state_dict())
        resumed.it = interrupted.it
        resumed_cursor.next_batch_index = 4
        resumed.current_batch_index = 4
        logs = []
        with mock.patch.object(implementation.wandb, "log", side_effect=logs.append), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            resumed.train_epoch()

        self.assertEqual(resumed.it, uninterrupted.it)
        for resumed_parameter, uninterrupted_parameter in zip(
                resumed.model.parameters(), uninterrupted.model.parameters()):
            self.assertTrue(torch.allclose(
                resumed_parameter.detach(),
                uninterrupted_parameter.detach(),
                rtol=0,
                atol=1e-7,
            ))
        self.assertEqual(
            resumed.optimizer.param_groups[0]["lr"],
            uninterrupted.optimizer.param_groups[0]["lr"],
        )

    def test_real_boundary_checkpoint_resume_matches_momentum_training(self):
        def prepare_trainer():
            trainer, cursor = _make_training_harness(5, 4)
            trainer.fabric = _FileRecordingFabric()
            trainer.optimizer = torch.optim.SGD(
                trainer.model.parameters(),
                lr=0.1,
                momentum=0.9,
            )
            trainer.config = implementation.OmegaConf.create({
                **vars(trainer.config),


            })
            trainer.train_sampler = SimpleNamespace()
            trainer.best_val_loss = 1.0
            return trainer, cursor

        uninterrupted, uninterrupted_cursor = prepare_trainer()
        with mock.patch.object(implementation.wandb, "log"), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            uninterrupted.train_epoch()

        interrupted, interrupted_cursor = prepare_trainer()

        class BoundarySaved(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            interrupted.config.checkpoint_dir = directory
            checkpoint_path = Path(directory) / "model_boundary.pt"
            fake_run = SimpleNamespace(dir=directory)

            commit_batch = interrupted._mark_active_train_batch_consumed

            def save_at_first_boundary(next_batch):
                commit_batch(next_batch)
                if interrupted_cursor.next_batch_index == 4:
                    saved_path = interrupted.save_model(
                        id="boundary",
                    )
                    self.assertEqual(Path(saved_path), checkpoint_path)
                    raise BoundarySaved

            interrupted._mark_active_train_batch_consumed = save_at_first_boundary
            with mock.patch.object(implementation.wandb, "run", fake_run), \
                    mock.patch.object(implementation.wandb, "log"), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(BoundarySaved):
                interrupted.train_epoch()

            resumed, resumed_cursor = prepare_trainer()
            with mock.patch.object(implementation.wandb, "log"), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                resumed.load_model(checkpoint_path)
                resumed.train_epoch()

        self.assertEqual(interrupted_cursor.next_batch_index, 4)
        self.assertEqual(resumed_cursor.next_batch_index, uninterrupted_cursor.next_batch_index)
        self.assertEqual(resumed.it, uninterrupted.it)
        self.assertEqual(
            resumed.optimizer.param_groups[0]["lr"],
            uninterrupted.optimizer.param_groups[0]["lr"],
        )
        for resumed_parameter, uninterrupted_parameter in zip(
                resumed.model.parameters(), uninterrupted.model.parameters()):
            self.assertTrue(torch.allclose(
                resumed_parameter.detach(),
                uninterrupted_parameter.detach(),
                rtol=0,
                atol=1e-7,
            ))

        resumed_state = resumed.optimizer.state_dict()
        uninterrupted_state = uninterrupted.optimizer.state_dict()
        self.assertEqual(resumed_state["param_groups"], uninterrupted_state["param_groups"])
        self.assertEqual(set(resumed_state["state"]), set(uninterrupted_state["state"]))
        for parameter_id in resumed_state["state"]:
            resumed_momentum = resumed_state["state"][parameter_id]["momentum_buffer"]
            uninterrupted_momentum = uninterrupted_state["state"][parameter_id]["momentum_buffer"]
            self.assertTrue(torch.allclose(
                resumed_momentum,
                uninterrupted_momentum,
                rtol=0,
                atol=1e-7,
            ))


class CheckpointBoundaryTests(unittest.TestCase):
    @staticmethod
    def make_trainer(accumulation=4, total_batches=10):
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.experiment_identity = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        trainer.device = torch.device("cpu")
        trainer.config = SimpleNamespace(accumulate_grad_batches=accumulation)
        trainer.pending_microbatches = 0
        trainer.pending_window_loss = 0.0
        trainer.model = torch.nn.Linear(1, 1, bias=False)
        cursor = SimpleNamespace(total_batches=total_batches)
        trainer._active_train_cursor = lambda: cursor
        return trainer

    def test_empty_boundary_guard_checks_counter_and_parameter_gradients(self):
        trainer = self.make_trainer()
        trainer.fabric = _RecordingFabric()
        trainer.pending_microbatches = 1
        with self.assertRaisesRegex(RuntimeError, "non-empty"):
            trainer._accumulation_checkpoint_state()
        with self.assertRaisesRegex(RuntimeError, "non-empty"):
            trainer.save_model()
        self.assertEqual(trainer.fabric.saved, [])
        trainer.pending_microbatches = 0
        trainer.model.weight.grad = torch.zeros_like(trainer.model.weight)
        with self.assertRaisesRegex(RuntimeError, "gradients_present=1"):
            trainer._accumulation_checkpoint_state()
        with self.assertRaisesRegex(RuntimeError, "gradients_present=1"):
            trainer.save_model()
        self.assertEqual(trainer.fabric.saved, [])
        trainer.model.weight.grad = None
        self.assertEqual(trainer._accumulation_checkpoint_state(), {
            "version": 1,
            "accumulate_grad_batches": 4,
            "pending_microbatches": 0,
        })

    def test_resume_cannot_overwrite_live_cursor_progress(self):
        trainer = self.make_trainer()
        cursor = SimpleNamespace(total_batches=10, next_batch_index=1)
        trainer._active_train_cursor = lambda: cursor
        trainer.epoch = 0
        with self.assertRaisesRegex(RuntimeError, "nonzero in-memory progress"):
            trainer._validate_active_train_cursor_checkpoint_state({
                "current_batch_index": 0,
            })

    def test_resume_state_rejects_mismatch_and_pending_gradients(self):
        trainer = self.make_trainer(accumulation=4)
        valid = {
            "accumulation_state": {
                "version": 1,
                "accumulate_grad_batches": 4,
                "pending_microbatches": 0,
            }
        }
        trainer._validate_accumulation_resume_state(valid)
        for change, pattern in (
                ({"accumulate_grad_batches": 2}, "does not match"),
                ({"pending_microbatches": 1}, "non-empty"),
                ({"version": 9}, "Unsupported")):
            state = dict(valid["accumulation_state"])
            state.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, pattern):
                trainer._validate_accumulation_resume_state({"accumulation_state": state})

    def test_boundary_save_records_versioned_empty_accumulation_state(self):
        trainer = self.make_trainer(accumulation=4)
        trainer.fabric = _RecordingFabric()
        trainer.config = implementation.OmegaConf.create({
            "accumulate_grad_batches": 4,


        })
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.train_sampler = SimpleNamespace(state_dict=lambda: {"sampler": 1})
        trainer._active_train_cursor_checkpoint_state = lambda: {"cursor": "state"}
        trainer.it = 2
        trainer.epoch = 0
        trainer.current_batch_index = 4
        trainer.best_val_loss = 1.0
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer.config.checkpoint_dir = directory
            saved_state = implementation.load_artifact(trainer.save_model(id="boundary"))
        self.assertEqual(saved_state["accumulation_state"], {
            "version": 1,
            "accumulate_grad_batches": 4,
            "pending_microbatches": 0,
        })


    def test_load_preflights_accumulation_and_clears_stale_gradients(self):
        trainer = self.make_trainer(accumulation=1)
        trainer.fabric = _RecordingFabric()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.pending_microbatches = 0
        trainer.epoch = 0
        trainer.it = 0
        trainer.current_batch_index = 0
        trainer.best_val_loss = float("inf")
        _, cursor = CursorIdentityTests.make_trainer(
            make_indexed_subset(list(range(100, 120)), list(range(20))))
        cursor.set_epoch(0)
        trainer._active_train_cursor = lambda: cursor
        trainer.model.weight.grad = torch.ones_like(trainer.model.weight)

        checkpoint = {
            "split_manifest": None,
            "train_cursor_state": trainer._active_train_cursor_checkpoint_state(),
            'format_version': 4,
            'rng_state_cuda': None,
            'experiment_identity': trainer.experiment_identity,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "python_rng_state": __import__("random").getstate(),
            "epoch": 0,
            "data": (0, 0),
            "it": 3,
            "current_batch_index": 0,
            "config": {"accumulate_grad_batches": 1},
            "accumulation_state": {
                "version": 1,
                "accumulate_grad_batches": 1,
                "pending_microbatches": 0,
            },
            "best_val_loss": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(implementation.to_safe(checkpoint), path)
            with mock.patch.object(implementation.wandb, "log"), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                trainer.load_model(path)
        self.assertIsNone(trainer.model.weight.grad)
        self.assertEqual(trainer.pending_microbatches, 0)
        self.assertEqual(trainer.it, 3)

    def test_load_rejects_accumulation_mismatch_before_model_mutation(self):
        trainer = self.make_trainer(accumulation=1)
        trainer.fabric = _RecordingFabric()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        original_weight = trainer.model.weight.detach().clone()
        checkpoint = {
            "config": vars(trainer.config),
            "split_manifest": None,
            "current_batch_index": 0,
            "train_cursor_state": {},
            'format_version': 4,
            'rng_state_cuda': None,
            'experiment_identity': trainer.experiment_identity,
            "model_state_dict": {"weight": torch.zeros_like(original_weight)},
            "accumulation_state": {
                "version": 1,
                "accumulate_grad_batches": 2,
                "pending_microbatches": 0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_checkpoint.pt"
            torch.save(implementation.to_safe(checkpoint), path)
            with self.assertRaisesRegex(RuntimeError, "does not match"), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                trainer.load_model(path)
        self.assertTrue(torch.equal(trainer.model.weight.detach(), original_weight))

    def test_load_rejects_cursor_signature_before_any_live_state_mutation(self):
        participant_ids = list(range(100, 108))
        subset = make_indexed_subset(participant_ids, [0, 1, 2, 3])
        trainer, cursor = CursorIdentityTests.make_trainer(subset)
        trainer.config = implementation.OmegaConf.create({

            "accumulate_grad_batches": 1,

        })
        trainer.fabric = _RecordingFabric()
        trainer.model = torch.nn.Linear(1, 1, bias=False)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.pending_microbatches = 0
        trainer.it = 7
        trainer.current_batch_index = 0
        trainer.best_val_loss = 1.0
        trainer.train_sampler = SimpleNamespace(load_state_dict=mock.Mock())
        cursor.set_epoch(3)

        saved_cursor = trainer._active_train_cursor_checkpoint_state()
        saved_cursor["signature"] = dict(saved_cursor["signature"])
        saved_cursor["signature"]["dataset_length"] += 1
        original_weight = trainer.model.weight.detach().clone()
        original_lr = trainer.optimizer.param_groups[0]["lr"]
        donor_model = torch.nn.Linear(1, 1, bias=False)
        donor_model.weight.data.zero_()
        donor_optimizer = torch.optim.SGD(donor_model.parameters(), lr=0.9)
        checkpoint = {
            "config": dict(trainer.config),
            "split_manifest": None,
            'format_version': 4,
            'rng_state_cuda': None,
            'experiment_identity': trainer.experiment_identity,
            "model_state_dict": donor_model.state_dict(),
            "optimizer_state_dict": donor_optimizer.state_dict(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "python_rng_state": __import__("random").getstate(),
            "epoch": 3,
            "data": (0, 3),
            "it": 99,
            "current_batch_index": 0,
            "train_cursor_state": saved_cursor,
            "train_sampler_state": {"would": "mutate"},
            "accumulation_state": {
                "version": 1,
                "accumulate_grad_batches": 1,
                "pending_microbatches": 0,
            },
            "best_val_loss": 9.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_cursor.pt"
            torch.save(implementation.to_safe(checkpoint), path)
            with self.assertRaisesRegex(RuntimeError, "structurally incompatible"), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                trainer.load_model(path)

        self.assertTrue(torch.equal(trainer.model.weight.detach(), original_weight))
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], original_lr)
        self.assertEqual(trainer.it, 7)
        self.assertEqual(trainer.epoch, 3)
        self.assertEqual(cursor.next_batch_index, 0)
        trainer.train_sampler.load_state_dict.assert_not_called()

    def test_distributed_elis_requires_unused_parameter_discovery(self):
        from healthformer.config import validate_distributed_config
        active = dict(Elis_augmentation_prob=1.0, elis_loss_weight=0.01,
                      find_unused_parameters=False)
        with self.assertRaisesRegex(ValueError, "find_unused_parameters=True"):
            validate_distributed_config(active, 2)

        validate_distributed_config(active, 1)
        active['Elis_augmentation_prob'] = 0.0
        with self.assertRaisesRegex(ValueError, "find_unused_parameters=True"):
            validate_distributed_config(active, 2)


class PaddingAlignmentTests(unittest.TestCase):
    def test_filtering_preserves_alignment_and_pads_all_channels(self):
        dataset = SimpleNamespace(mask_modality_val=9)
        tokens = torch.tensor([11, 22, 33, 44])
        continuous = torch.tensor([1.1, 2.2, 3.3, 4.4])
        modalities = torch.tensor([0, 1, 2, 3])
        times = torch.arange(28).reshape(4, 7)
        channels = (tokens, continuous, modalities, times)
        for mask_values in ([True, False, True, False], [True] * 4, [False] * 4):
            for skip_unchanged in (False, True):
                with self.subTest(mask=mask_values, skip=skip_unchanged):
                    mask = torch.tensor(mask_values)
                    outputs = implementation.CustomDataset._filter_and_pad(
                        dataset, *channels, mask, 4,
                        skip_padding_when_unchanged=skip_unchanged)
                    kept = int(mask.sum())
                    for source, output, padding in zip(channels, outputs, (0, 0, 9, 0)):
                        self.assertEqual(output.shape, source.shape)
                        self.assertEqual(output.dtype, source.dtype)
                        self.assertTrue(torch.equal(output[:kept], source[mask]))
                        self.assertTrue(bool((output[kept:] == padding).all()))

if __name__ == "__main__":
    unittest.main()
