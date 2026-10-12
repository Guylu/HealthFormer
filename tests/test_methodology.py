"""Regression checks for objective definitions, event matching and experiment identity."""
import contextlib
import copy
import io
import json
from pathlib import Path
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
from healthformer.identity import experiment_identity, validate_identity
from healthformer.losses import vocabulary_cross_entropy
from healthformer.model import MultiModalTransformer
from healthformer.trainer import Trainer
from healthformer.validation_windows import validation_windows
from healthformer.visits import identify_visits


class MethodologyTests(unittest.TestCase):
    def test_statistics_use_modalities_finite_training_values_and_existing_thresholds(self):
        owner = InputMixin()
        owner.config = SimpleNamespace(continuous_value_embeddings=True)
        owner.bins = {'a': [], 'b': [], 'constant': []}
        owner.modality_name_to_number = {'a': 0, 'b': 1, 'category': 2, 'constant': 3}
        owner.cat_cols = ['category']
        a = torch.arange(50, dtype=torch.float64)
        b = torch.arange(49, dtype=torch.float64) + 100
        values = torch.cat((a, b, torch.tensor([900., 901.]), torch.full((50,), 7.),
                            torch.tensor([float('nan'), float('inf'), 99999.])))
        modalities = torch.tensor([0] * 50 + [1] * 49 + [2] * 2 + [3] * 50 + [0, 1, 4])
        owner.dataset = SimpleNamespace(
            tokens=torch.full((2, len(values)), 42),
            continuous_tokens=torch.stack((values, torch.full_like(values, -9999))),
            modalities=modalities.repeat(2, 1))
        owner.train_dataset = SimpleNamespace(indices=[0])
        stats, ranges = owner.compute_training_statistics()
        self.assertEqual(set(stats), {'a', 'constant'})
        self.assertEqual(stats['a']['mean'], 24.5)
        self.assertAlmostEqual(stats['a']['std'], float(a.std(correction=1)), places=12)
        self.assertEqual(stats['constant'], {'mean': 7., 'std': 1e-8})
        self.assertEqual(ranges, {'a': {'min': 0., 'max': 49.},
                                  'b': {'min': 100., 'max': 148.},
                                  'constant': {'min': 7., 'max': 7.}})
        owner.dataset.tokens.zero_()
        owner.dataset.continuous_tokens[1].fill_(1e9)
        self.assertEqual(owner.compute_training_statistics(), (stats, ranges))
        owner.config.continuous_value_embeddings = False
        self.assertEqual(owner.compute_training_statistics(), (stats, None))

    def test_pre_correction_checkpoint_identity_is_rejected(self):
        current = {'schema_version': 9, 'settings': {}, 'artifacts': {}, 'split': {}}
        old = {**current, 'schema_version': 8}
        with self.assertRaisesRegex(ValueError, 'initialize_from'):
            validate_identity(old, current)
        validate_identity(current, current)

    def test_missing_continuous_targets_preserve_loss_and_gradients(self):
        owner = SimpleNamespace(modality_name_to_number={'a': 0},
            mod2bininfo={'a': ([0, 1], [0., 1.])},
            mod2stats={'a': {'mean': 0., 'std': 1.}},
            config=SimpleNamespace(mae_loss_scale=1.))
        logits = torch.zeros(3, 2, requires_grad=True)
        targets = torch.tensor([0., float('nan'), float('inf')])
        loss = MultiModalTransformer.handle_continuous_loss(owner, logits, torch.tensor(1.),
            {}, targets, torch.zeros(3, dtype=torch.long), 'Train')
        self.assertEqual(loss.item(), 1.5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[1:] == 0).all())
        self.assertGreater(logits.grad[0].abs().sum().item(), 0)
        targets.fill_(float('nan'))
        result = MultiModalTransformer.handle_continuous_loss(owner, logits, torch.tensor(1.),
            {}, targets, torch.zeros(3, dtype=torch.long), 'Train')
        self.assertEqual(result.item(), 1.)

    def test_invalid_bins_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = json.loads(create_fixture(directory).read_text())
            data = torch.load(cfg['data_path'], weights_only=True)
        validator = InputMixin()
        validator._validate_training_artifact(data)
        for edges in ([0, 1], [0] * 16, list(range(15)) + [float('inf')],
                      list(range(15)) + [float('nan')]):
            bad = copy.deepcopy(data)
            bad['bins']['synthetic_BloodTestsLoader'] = edges
            with self.assertRaises(ValueError): validator._validate_training_artifact(bad)
        bad = copy.deepcopy(data)
        bad['bins']['unknown'] = [0, 1]
        with self.assertRaisesRegex(ValueError, 'Unknown bin modality'):
            validator._validate_training_artifact(bad)

    def test_year_boundary_and_missing_values_train_and_validate(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            overrides = json.loads(create_fixture(directory).read_text())
            data = torch.load(overrides['data_path'], weights_only=True)
            data['time_expanded'][:, 32:, 4] = 150
            data['continuous_tokens'][:, 3::16] = float('nan')
            data['continuous_tokens'][:, 7::16] = float('inf')
            torch.save(data, overrides['data_path'])
            cfg = OmegaConf.create({**config, **overrides, 'evaluate': False,
                                    'Elis_augmentation_prob': 1., 'elis_loss_weight': .01})
            fabric = Fabric(accelerator='cpu', devices=1, precision='32-true'); fabric.launch()
            fabric.seed_everything(42)
            trainer = Trainer(cfg, fabric)
            embeddings = trainer.model._build_continuous_value_embedding(
                data['continuous_tokens'], data['modalities'], 64)
            self.assertTrue(torch.isfinite(embeddings).all())
            self.assertTrue((embeddings[:, 3::16] == 0).all())
            trainer.train_epoch()
            self.assertGreater(trainer.it, 0)
            self.assertTrue(all(torch.isfinite(v).all() for v in trainer.model.state_dict().values()))
            with patch('healthformer.evaluation.wandb.log') as log:
                summary = trainer.evaluate()
            self.assertTrue(summary['valid'])
            log.assert_called_once()
            logged = log.call_args.args[0]
            self.assertEqual(logged['Validation/CE Overall Loss'], summary['cross_entropy'])
            self.assertEqual(logged['Validation/coverage/scored_targets'], summary['scored_targets'])
            self.assertTrue(torch.isfinite(torch.tensor(logged['Validation CE/ synthetic_BloodTestsLoader'])))
            self.assertFalse(any('MAE/' in key or 'Pearson' in key or 'Top1/' in key for key in logged))

    def test_longitudinal_correlations_are_descriptive(self):
        from healthformer.longitudinal import LongitudinalMixin
        owner = SimpleNamespace(number_to_modality_name={0: 'a'})
        info = dict(count=3, ce_count=3, ce_sum=1., correct_top1=0,
                    correct_top5=1, correct_top10=2, expected_vals=[1., 2., 3.],
                    real_vals=[1., 3., 2.])
        for values in ([1., 2., 3.], [1., 1., 1.]):
            info['expected_vals'] = values
            with patch('healthformer.longitudinal.wandb.log') as log:
                LongitudinalMixin._log_longitudinal_metrics(owner, mode='Validation',
                    metrics_by_mod={0: info}, total_ce_sum=1., total_ce_count=3,
                    total_count=3,
                    correct_top1=0, correct_top5=1, correct_top10=2)
            log.assert_called_once()
            logged = log.call_args.args[0]
            self.assertAlmostEqual(logged['Longitudinal_Validation_Loss/'], 1 / 3)
            self.assertEqual(logged['Longitudinal_Validation_ContinuousCount/a'], 3)
            keys = set(logged)
            self.assertFalse(any('PearsonP/' in k or 'SpearmanP/' in k for k in keys))
            self.assertEqual('Longitudinal_Validation_PearsonCorr/a' in keys, len(set(values)) > 1)

    def test_final_elis_validation_precedes_checkpoint(self):
        from healthformer.training import OptimizationMixin
        events = []
        owner = SimpleNamespace(epoch=0, current_batch_index=0,
            config=SimpleNamespace(epochs=1, stop_early_now=None, evaluate=True,
                                   longitudinal_evaluation=True, save=True),
            validation_dataloader_with_morethan1_visit=object())
        def train_epoch():
            events.append('train'); owner.epoch += 1
        owner.train_epoch = train_epoch
        owner.evaluate = lambda: events.append('standard')
        owner.evaluate_longitudinal = lambda *args: events.append('elis')
        owner.save_model = lambda **kwargs: events.append('save')
        OptimizationMixin.train(owner)
        self.assertEqual(events, ['train', 'standard', 'elis', 'save'])

    def test_configured_seed_reaches_samplers_and_loaders(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            overrides = json.loads(create_fixture(directory).read_text())
            cfg = OmegaConf.create({**config, **overrides, 'seed': 73})
            fabric = Fabric(accelerator='cpu', devices=1, precision='32-true'); fabric.launch()
            trainer = Trainer(cfg, fabric)
            self.assertEqual(trainer.train_sampler.seed, 73)
            self.assertEqual(trainer.train_dataloader.generator.initial_seed(), 73)
            self.assertEqual(trainer.validation_dataloader.generator.initial_seed(), 73)

    def make_owner(self, vocabulary):
        return SimpleNamespace(config=SimpleNamespace(loss_vocabulary=vocabulary),
            vals_per_modality={'a': [0, 2], 'b': [0, 1]},
            modality_name_to_number={'a': 0, 'b': 1},
            cum_base_value_per_modality={0: 0, 1: 3}, mask_modality_val=2)

    def test_vocabulary_normalization_uses_exact_ids(self):
        logits = torch.tensor([[1., -2., 2., 3., 4.]], requires_grad=True)
        target, mod = torch.tensor([2]), torch.tensor([0])
        for soft in (None, torch.tensor([[0.2, 0., 0.8, 0., 0.]])):
            owner = self.make_owner('modality')
            original = vocabulary_cross_entropy(owner, logits, target, mod, soft)
            perturbed = logits.clone(); perturbed[:, [1, 3, 4]] += 100
            torch.testing.assert_close(original, vocabulary_cross_entropy(owner, perturbed, target, mod, soft))
            grad, = torch.autograd.grad(original.sum(), logits, retain_graph=True)
            self.assertTrue(torch.equal(grad[:, [1, 3, 4]], torch.zeros(1, 3)))
            owner.config.loss_vocabulary = 'global'
            self.assertFalse(torch.equal(vocabulary_cross_entropy(owner, logits, target, mod, soft),
                                         vocabulary_cross_entropy(owner, perturbed, target, mod, soft)))

    def test_soft_label_settings_and_gaussian_behavior_are_preserved(self):
        self.assertTrue(config['soft_labels'])
        self.assertEqual(config['SL_sigma'], 0.01)
        self.assertEqual(config['soft_labels_scale'], 1)
        model = MultiModalTransformer.__new__(MultiModalTransformer)
        torch.nn.Module.__init__(model)
        owner = self.make_owner('global')
        for key, value in vars(owner).items(): setattr(model, key, value)
        model.config.SL_sigma = 0.8  # Check genuine soft targets as well as the default.
        model.config.soft_labels_scale = 1
        model.cat_cols = []
        logits = torch.tensor([[1., -2., 2., 3., 4.]], requires_grad=True)
        targets, mods = torch.tensor([2]), torch.tensor([0])
        ranges = torch.arange(5).float()
        weights = torch.exp(-0.5 * ((ranges - 2) / 0.8)**2)
        weights[ranges > 3] = 0  # Preserve the historical Gaussian support, per user request.
        weights /= weights.sum() + 1e-12
        expected = -(weights * logits.log_softmax(-1)).sum()
        actual = model.apply_soft_cross_entropy(logits, None, {}, mods, targets)
        torch.testing.assert_close(actual, expected)

    def test_repeated_markers_select_second_distinct_year(self):
        mods = torch.tensor([2, 2, 0, 0, 2, 2, 0])
        times = torch.tensor([[140], [140], [140], [140], [141], [141], [141]])
        self.assertEqual(identify_visits(mods, times, 2, 0), ((4, 4, 7, 141), None))
        self.assertEqual(identify_visits(mods, torch.ones_like(times), 2, 0)[1], 'fewer_than_two_marker_years')
        times[-1] = 139
        self.assertEqual(identify_visits(mods, times, 2, 0)[1], 'unordered_years')


    def test_validation_windows_score_every_target_once(self):
        tokens = torch.arange(17).view(1, -1)
        batch = (tokens, tokens.float(), torch.zeros_like(tokens), torch.zeros(1, 17, 7))
        cfg = SimpleNamespace(validation_sequence_policy='sliding', validation_max_tokens=6,
                              max_seq_length=8)
        coverage = dict(participants=0, eligible_targets=0, truncated_participants=0, oversized_participants=0)
        scored = []
        for window, skip in validation_windows([batch], cfg, 2, coverage):
            values = window[0][0, 1:]
            scored.extend(values[skip:].tolist())
            self.assertLessEqual(window[0].shape[1], 6)
        self.assertEqual(scored, list(range(1, 17)))
        self.assertEqual(coverage['eligible_targets'], len(scored))

    def test_identity_detects_content_and_objective_changes_not_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            overrides = json.loads(create_fixture(directory).read_text())
            cfg = OmegaConf.create({**config, **overrides})
            original = experiment_identity(cfg, {'train_ids': [121]})
            cfg.checkpoint_dir = 'elsewhere'
            validate_identity(original, experiment_identity(cfg, {'train_ids': [121]}))
            cfg.mae_loss_scale = 9
            with self.assertRaisesRegex(ValueError, 'mae_loss_scale'):
                validate_identity(original, experiment_identity(cfg, {'train_ids': [121]}))
            cfg.mae_loss_scale = config['mae_loss_scale']
            with Path(cfg.data_path).open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifacts.training'):
                validate_identity(original, experiment_identity(cfg, {'train_ids': [121]}))
            with self.assertRaisesRegex(ValueError, 'initialize_from'):
                validate_identity(None, original)


    def test_visit_marker_ignores_validation_changes_and_weights_only_reset(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            overrides = json.loads(create_fixture(directory).read_text())
            cfg = OmegaConf.create({**config, **overrides, 'evaluate': False})
            fabric = Fabric(accelerator='cpu', devices=1, precision='32-true'); fabric.launch()
            fabric.seed_everything(42)
            trainer = Trainer(cfg, fabric)
            trainer.train_epoch()
            saved = trainer.save_model('source')
            expected = copy.deepcopy(trainer.model.state_dict())
            visit_marker = trainer.visit_marking
            artifact = torch.load(cfg.data_path, weights_only=True)
            # Change validation to another valid modality/token pair; keep training untouched.
            artifact['modalities'][:3, :2] = 3; artifact['tokens'][:3, :2] = 49
            torch.save(artifact, cfg.data_path)
            cfg.initialize_from = saved
            fresh = Trainer(cfg, fabric)
            self.assertEqual(visit_marker, fresh.visit_marking)
            self.assertEqual(fresh.it, 0)
            self.assertEqual(fresh.epoch, 0)
            self.assertEqual(len(fresh.optimizer.state), 0)
            for k, value in expected.items(): torch.testing.assert_close(fresh.model.state_dict()[k], value)


    def test_real_training_with_elis_and_restricted_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            overrides = json.loads(create_fixture(directory).read_text())
            cfg = OmegaConf.create({**config, **overrides, 'evaluate': False,
                                    'loss_vocabulary': 'modality', 'accumulate_grad_batches': 2,
                                    'Elis_augmentation_prob': 1., 'elis_loss_weight': .01})
            fabric = Fabric(accelerator='cpu', devices=1, precision='32-true'); fabric.launch()
            fabric.seed_everything(42)
            trainer = Trainer(cfg, fabric)
            initial = copy.deepcopy(trainer.model.state_dict())
            trainer.train_epoch()
            self.assertGreater(trainer.it, 0)
            self.assertTrue(any(not torch.equal(value, initial[key]) for key, value in trainer.model.state_dict().items()))
            self.assertTrue(all(torch.isfinite(value).all() for value in trainer.model.state_dict().values()))
            self.assertTrue(trainer.evaluate()['valid'])
            trainer.save_model('pooled')
