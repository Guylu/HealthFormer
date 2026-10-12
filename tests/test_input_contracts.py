"""Regressions for observed values, context limits and causal metadata."""
import contextlib
import copy
import io
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from lightning.fabric import Fabric
from omegaconf import OmegaConf
from examples.make_training_fixture import create_fixture
from healthformer.config import config
from healthformer.inputs import InputMixin
from healthformer.longitudinal import LongitudinalMixin
from healthformer.model import MultiModalTransformer
from healthformer.trainer import Trainer
from healthformer.visits import elis_modalities
from tests import test_regressions as fixtures


class InputContractTests(unittest.TestCase):
    def test_artifact_rejects_mismatched_pairs_padding_holes_and_fractional_offsets(self):
        original = fixtures.ArtifactAndDatasetTests.valid_artifact()
        InputMixin()._validate_training_artifact(original)
        bad = copy.deepcopy(original); bad['tokens'][0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'Token/modality mismatch'):
            InputMixin()._validate_training_artifact(bad)
        bad = copy.deepcopy(original); bad['modalities'][0] = torch.tensor([2, 0])
        with self.assertRaisesRegex(ValueError, 'Padding must be trailing'):
            InputMixin()._validate_training_artifact(bad)
        bad = copy.deepcopy(original); bad['cum_base_value_per_modality'][0] = 0.5
        with self.assertRaisesRegex(ValueError, 'Invalid token offset'):
            InputMixin()._validate_training_artifact(bad)

    def test_augmentation_changes_only_selected_observed_values_using_declared_bins(self):
        ds = fixtures.make_dataset(modality_name_to_number={'m': 0},
            mod2bininfo={'m': ([10, 12, 17], [.5, 1.5, 2.5])}, bins={'m': [0., 1., 2., 3.]})
        tokens = torch.tensor([10, 12, 17, 12])
        values = torch.tensor([.1, float('nan'), 2.9, 1.1])
        mods = torch.zeros(4, dtype=torch.long)
        for rate in (0., .5, 1.):
            ds.augmentation_rate = rate
            with patch('torch.randperm', side_effect=lambda n, **kw: torch.arange(n)), \
                    patch('torch.randint', side_effect=lambda low, high, size, **kw: torch.ones(size, dtype=torch.long)):
                actual_tokens, actual_values = ds.augment_noise(mods, tokens.clone(), values.clone())
            changed = [0, 2, 3][:int(3 * rate)]
            for i in range(4):
                if i not in changed:
                    torch.testing.assert_close(actual_values[i], values[i], equal_nan=True)
                    self.assertEqual(actual_tokens[i], tokens[i])
                else:
                    ordinal = [10, 12, 17].index(actual_tokens[i].item())
                    self.assertGreaterEqual(actual_values[i].item(), ordinal)
                    self.assertLessEqual(actual_values[i].item(), ordinal + 1)
            self.assertTrue(torch.isin(actual_tokens, torch.tensor([10, 12, 17])).all())

    def test_demographics_cannot_expose_later_or_masked_events(self):
        model = fixtures.GenderTests().make_model({'birth_event': 0, 'gender': 1, 'gender_at_birth': 2}, {0: 0, 1: 10, 2: 20})
        model.age_positional_encoding = torch.arange(1000).float()[:, None].repeat(1, 2)
        tokens = torch.tensor([[5, 21, 1, 11, 5]])
        mods = torch.tensor([[3, 2, 0, 1, 3]])
        times = torch.ones(1, 5, 7, dtype=torch.long); times[:, :, 4] = 140
        times[:, 2, 4] = 139
        template = torch.zeros(1, 5, 2)
        original = model._build_demographic_embedding(tokens, mods, times, 5, template)
        changed_tokens = tokens.clone(); changed_tokens[:, 3] = 12
        changed_times = times.clone(); changed_times[:, 2, 4] = 138
        changed = model._build_demographic_embedding(changed_tokens, mods, changed_times, 5, template)
        torch.testing.assert_close(original[:, :2], changed[:, :2])
        gender_only = model._build_demographic_embedding(changed_tokens, mods, times, 5, template)
        torch.testing.assert_close(original[:, :3], gender_only[:, :3])
        self.assertFalse(torch.equal(original[:, 3:], gender_only[:, 3:]))
        self.assertGreater(original[:, 1].abs().sum().item(), 0)  # earlier fallback is visible
        mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril(); mask[:, :, 4, 2:4] = False
        a = model._build_demographic_embedding(tokens, mods, times, 5, template, mask)
        b = model._build_demographic_embedding(changed_tokens, mods, changed_times, 5, template, mask)
        torch.testing.assert_close(a[:, 4], b[:, 4])

    def test_context_guard_fails_before_model_allocations(self):
        owner = SimpleNamespace(config=SimpleNamespace(max_seq_length=8))
        with self.assertRaisesRegex(ValueError, 'max_seq_length'):
            MultiModalTransformer.forward(owner, torch.ones(1, 9), None, None, None)

    def test_elis_prompt_is_bounded_and_reports_eligible_targets_before_truncation(self):
        owner = SimpleNamespace(device='cpu', mask_modality_val=2, visit_marking=0,
            year_idx=4, modality_name_to_number={'marker_DEXALoader': 0, 'x_BloodTestsLoader': 1},
            config=SimpleNamespace(max_seq_length=8))
        tokens = torch.arange(8); cont = tokens.float()
        mods = torch.tensor([0, 1, 1, 1, 0, 1, 1, 1])
        times = torch.ones(8, 7, dtype=torch.long); times[:, 4] = torch.tensor([140]*4+[141]*4)
        for limit, expected in ((4, 0), (8, 2), (20, 4)):
            owner.config.max_seq_length = limit
            coverage = dict(participants=0, eligible_targets=0, truncated_participants=0)
            sample = LongitudinalMixin._prepare_longitudinal_prompt_sample(owner, tokens, cont, mods, times, coverage)
            self.assertEqual(coverage, dict(participants=1, eligible_targets=4, truncated_participants=int(expected<4)))
            if expected:
                self.assertEqual(len(sample['target_tokens']), expected)
                self.assertLessEqual(len(sample['tokens']), limit)
            else:
                self.assertIsNone(sample)
        mapping = {'x_BloodTestsLoader': 0, 'x_NightingaleLoader': 1, 'x_locations': 2}
        self.assertEqual(elis_modalities(mapping), {0})
        self.assertEqual(elis_modalities(mapping, evaluation=True), {0, 1, 2})

    def test_training_rejects_oversize_records_and_zero_usable_batches(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            overrides = json.loads(create_fixture(directory).read_text())
            fabric = Fabric(accelerator='cpu', devices=1, precision='32-true'); fabric.launch()
            for changes, message in (({'max_seq_length': 32}, 'Training record exceeds'),
                                     ({'batch_per_gpu': 100}, 'No usable training batches')):
                cfg = OmegaConf.create({**config, **overrides, **changes})
                with self.assertRaisesRegex(ValueError, message): Trainer(cfg, fabric)

    def test_resume_after_final_batch_finishes_epoch_without_an_extra_update(self):
        trainer, cursor = fixtures._make_training_harness(2, 1, run_elis=False)
        trainer.it = 2
        trainer.current_batch_index = 2
        cursor.next_batch_index = 2
        with patch.object(fixtures.implementation.wandb, 'log'), contextlib.redirect_stdout(io.StringIO()):
            trainer.train_epoch()
        self.assertEqual(trainer.it, 2)
        self.assertEqual(trainer.epoch, 1)
        self.assertEqual(cursor.next_batch_index, 0)
