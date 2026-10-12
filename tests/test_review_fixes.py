"""Failure cases and interactions from the release review."""
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
from torch.utils.data import Subset
from examples.make_training_fixture import create_fixture
from healthformer.config import config, validate_methodology_config
from healthformer.longitudinal import LongitudinalMixin
from healthformer.samplers import EliFriendlyDistributedSampler
from healthformer.visits import identify_visits
from healthformer.prompts import pad_prompts
from healthformer.serialization import load_artifact
from healthformer.trainer import Trainer
from tests import test_regressions as fixtures


class ReviewFixTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.overrides = json.loads(create_fixture(self.directory.name).read_text())
        self.fabric = Fabric(accelerator='cpu', devices=1, precision='32-true')
        self.fabric.launch()
        self.capture_context = contextlib.ExitStack()  # Python 3.9-compatible output capture.
        self.addCleanup(self.capture_context.close)
        self.capture_context.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.capture_context.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def trainer(self, **overrides):
        return Trainer(OmegaConf.create({**config, **self.overrides, 'evaluate': False,
                                        'save': False, **overrides}), self.fabric)

    def test_legacy_checkpoint_setting_names_resume_without_weakening_identity(self):
        from healthformer.config import CONFIG_ALIASES
        from healthformer.serialization import to_safe
        trainer = self.trainer()
        trainer.train_epoch()
        saved = load_artifact(trainer.save_model())
        for settings in (saved['config'], saved['experiment_identity']['settings']):
            for old, new in CONFIG_ALIASES.items():
                settings[old] = settings.pop(new)
        path = Path(self.directory.name) / 'legacy-names.pt'
        torch.save(to_safe(saved), path)
        resumed = self.trainer(resume_from=str(path))
        self.assertEqual(resumed.it, trainer.it)
        self.assertEqual(resumed.epoch, trainer.epoch)
        self.assertEqual(resumed.model.state_dict().keys(), trainer.model.state_dict().keys())
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
        saved['experiment_identity']['settings']['elis_min_len'] += 1
        torch.save(to_safe(saved), path)
        with self.assertRaisesRegex(RuntimeError, 'elis_max_input_length'):
            self.trainer(resume_from=str(path))

    def test_failed_optimizer_update_preserves_completed_counter(self):
        for failure in ('clipping', 'optimizer'):
            trainer = self.trainer()
            trainer.it = 3
            parameter = next(trainer.model.parameters())
            parameter.grad = torch.full_like(parameter, float('inf') if failure == 'clipping' else 1.)
            before = copy.deepcopy(trainer.model.state_dict())
            with self.subTest(failure=failure), \
                    patch.object(trainer.optimizer, 'step', side_effect=RuntimeError('optimizer failure')) as step:
                with self.assertRaises(RuntimeError):
                    trainer.step_optimizer()
                self.assertEqual(trainer.it, 3)
                self.assertEqual(step.call_count, 0 if failure == 'clipping' else 1)
            for name, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_successful_update_uses_next_learning_rate_then_commits_counter(self):
        trainer = self.trainer()
        trainer.it = 3
        trainer.warmup_steps = 6
        trainer.max_steps = 12
        parameter = next(trainer.model.parameters())
        before = parameter.detach().clone()
        parameter.grad = torch.ones_like(parameter)
        expected_lr = trainer.get_lr(4)
        original_step = trainer.optimizer.step

        def step():
            self.assertEqual(trainer.it, 3)
            self.assertTrue(all(group['lr'] == expected_lr for group in trainer.optimizer.param_groups))
            return original_step()

        with patch.object(trainer.optimizer, 'step', side_effect=step):
            metrics = trainer.step_optimizer()
        self.assertEqual(trainer.it, 4)
        self.assertEqual(metrics['misc/Iteration'], 4)
        self.assertEqual(metrics['misc/Learning rate'], expected_lr)
        self.assertFalse(torch.equal(parameter, before))
        self.assertIsNone(parameter.grad)

    def test_rejected_augmentation_restores_all_channels_and_training_updates(self):
        artifact = torch.load(self.overrides['data_path'], weights_only=True)
        artifact['tokens'].zero_(); artifact['modalities'].fill_(4)
        artifact['continuous_tokens'].zero_(); artifact['time_expanded'].zero_()
        artifact['tokens'][:, :2] = torch.tensor([33, 1])
        artifact['modalities'][:, :2] = torch.tensor([2, 0])
        artifact['continuous_tokens'][:, :2] = torch.tensor([1.5, float('nan')])
        torch.save(artifact, self.overrides['data_path'])
        trainer = self.trainer(random_modality_subset_chance=1., augmentation_chance=1.,
                               augmentation_rate=1., reduce_context_in_train=True, add_age_gender_as_pe=True)
        before = copy.deepcopy(trainer.model.state_dict())
        # Select an absent modality, after noise has already changed the marker value.
        with patch('random.sample', return_value=[1]):
            actual = trainer.dataset[3]
            for i, name in enumerate(('tokens', 'continuous_tokens', 'modalities', 'time_expanded')):
                torch.testing.assert_close(actual[i], artifact[name][3].to(actual[i].dtype), equal_nan=True)
                self.assertNotEqual(actual[i].data_ptr(), getattr(trainer.dataset, name)[3].data_ptr())
            trainer.train_epoch()
        self.assertGreater(trainer.it, 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in trainer.model.state_dict().items()))
        for name in ('tokens', 'continuous_tokens', 'modalities', 'time_expanded'):
            torch.testing.assert_close(getattr(trainer.dataset, name), artifact[name], equal_nan=True)

    def test_targetless_batch_fails_before_forward_or_optimizer(self):
        trainer, cursor = fixtures._make_training_harness(1, 1)
        batch = list(fixtures._training_batches(1)[0]); batch[2].fill_(trainer.mask_modality_val)
        trainer._active_train_dataloader = lambda: [tuple(batch)]
        with self.assertRaisesRegex(RuntimeError, 'no prediction targets'):
            trainer.train_epoch()
        self.assertEqual(trainer.it, 0)
        self.assertEqual(cursor.next_batch_index, 0)
        self.assertEqual(trainer.optimizer.gradient_windows, [])

    def test_removal_rate_zero_and_padding_do_not_change_removal_semantics(self):
        ds = fixtures.make_dataset(random_removal_rate=0.)
        tok = torch.tensor([1, 2, 3, 4, 0, 0]); mods = torch.tensor([0, 1, 0, 0, 2, 2])
        time = torch.arange(42).reshape(6, 7)
        original = (tok, tok.float(), mods, time)
        actual = ds.remove_random_tokens_and_pad(*original)
        for a, b in zip(actual, original): torch.testing.assert_close(a, b)
        ds.random_removal_rate = 1.
        removed = ds.remove_random_tokens_and_pad(*original)
        self.assertEqual(removed[2].tolist(), [1, 2, 2, 2, 2, 2])
        self.assertEqual(removed[0][0], tok[1])
        torch.testing.assert_close(removed[3][0], time[1])

    def test_fp16_is_rejected_by_shared_config_validation(self):
        for accelerator in ('cpu', 'cuda'):
            with self.assertRaisesRegex(ValueError, 'FP16 is unsupported'):
                validate_methodology_config({**config, 'accelerator': accelerator, 'precision': '16-mixed'})
        for accelerator, precision in (('cpu', '32-true'), ('cuda', '32-true'), ('cuda', 'bf16-mixed')):
            validate_methodology_config({**config, 'accelerator': accelerator, 'precision': precision})
        with self.assertRaisesRegex(ValueError, 'FP16 is unsupported'):
            self.trainer(precision='16-mixed')

    def test_vocabulary_boundary_allocation_keeps_existing_shapes_when_possible(self):
        artifact = torch.load(self.overrides['data_path'], weights_only=True)
        for size, expected in ((64, 128), (126, 128), (127, 256), (128, 256), (255, 384)):
            artifact['vocab_size'] = size
            torch.save(artifact, self.overrides['data_path'])
            trainer = self.trainer()
            self.assertEqual(trainer.model.embeddings.num_embeddings, expected)
            self.assertEqual(trainer.model.embeddings.padding_idx, size + 1)
            batch = next(iter(trainer.train_dataloader))
            tok, cont, mods, time = batch
            logits = trainer.model(tok[:, :-1], cont, mods, time)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(logits.shape[-1], expected)

    def test_requested_elis_validation_requires_scored_targets(self):
        trainer = self.trainer(longitudinal_evaluation=True)
        for loader in ([], trainer.validation_dataloader_with_morethan1_visit):
            trainer.config.max_seq_length = 2
            with self.assertRaisesRegex(RuntimeError, 'no eligible targets'):
                trainer.evaluate_longitudinal(loader)
            self.assertFalse(trainer.last_longitudinal_summary['valid'])
            self.assertEqual(trainer.last_longitudinal_summary['scored_targets'], 0)
            self.assertTrue(trainer.model.training)

    def test_unscored_future_events_do_not_spend_elis_context(self):
        owner = SimpleNamespace(device='cpu', mask_modality_val=3, visit_marking=0, year_idx=4,
            modality_name_to_number={'m_DEXALoader': 0, 'm_BloodTestsLoader': 1, 'unselected': 2},
            config=SimpleNamespace(max_seq_length=8))
        mods = torch.tensor([0, 1, 1, 1, 0, 2, 2, 1]); tok = torch.arange(8)
        time = torch.zeros(8, 7, dtype=torch.long); time[:, 4] = torch.tensor([140]*4+[141]*4)
        coverage = dict(participants=0, eligible_targets=0, truncated_participants=0)
        sample = LongitudinalMixin._prepare_longitudinal_prompt_sample(owner, tok, tok.float(), mods, time, coverage)
        self.assertEqual(sample['target_tokens'].tolist(), [4, 7])
        self.assertEqual(len(sample['tokens']), 7)
        self.assertEqual(coverage['eligible_targets'], 2)
        self.assertEqual(coverage['truncated_participants'], 0)

    def test_elis_training_and_validation_queries_match_and_hide_future_values(self):
        trainer = self.trainer(n_layers=2, zero_init_mlp=False, Elis_augmentation_prob=1., elis_loss_weight=.01)
        trainer.model.eval()
        tok, cont, mods, time = next(iter(trainer.validation_dataloader))
        tok, cont, mods, time = (x[:1] for x in (tok, cont, mods, time))
        real_model = trainer.model
        calls = []
        class Capture:
            def __call__(self, *args, **kwargs):
                logits = real_model(*args, **kwargs)
                calls.append((args, kwargs, logits))
                return logits
            calculate_loss = real_model.calculate_loss
        trainer.model = Capture()
        with torch.no_grad():
            _, ran = trainer.compute_elis_loss(tok[:, :-1], cont, tok[:, 1:], mods, time)
        self.assertTrue(ran)
        trainer.model = real_model
        sample = trainer._prepare_longitudinal_prompt_sample(tok[0], cont[0], mods[0], time[0])
        args, training_kwargs, training_logits = calls[0]
        for got, expected in zip(args, (sample['tokens'][:-1], sample['continuous_tokens'][:-1],
                                       sample['modalities'].long(), sample['time_expanded'])):
            torch.testing.assert_close(got[0], expected)
        inputs = [sample['tokens'][:-1][None], sample['continuous_tokens'][:-1][None],
                  sample['modalities'][None], sample['time_expanded'][None]]
        mask = sample['attention_mask'][None, None]
        queries = sample['target_logit_indices']
        self.assertTrue(torch.equal(training_kwargs['mask'], mask))
        with torch.no_grad():
            reference = real_model(*inputs, mask=mask)
            torch.testing.assert_close(training_logits[:, queries], reference[:, queries])
            for pos in queries[:-1]:
                inputs[0][0, pos+1] = (inputs[0][0, pos+1] + 1) % 64
                inputs[1][0, pos+1] += 100
            perturbed = real_model(*inputs, mask=mask)
        torch.testing.assert_close(perturbed[:, queries], reference[:, queries])

    def test_shared_elis_mask_preserves_query_gradients_with_dropout(self):
        trainer = self.trainer(n_layers=2, zero_init_mlp=False, dropout=.1,
                               Elis_augmentation_prob=1., elis_loss_weight=.01)
        trainer.model.train()
        tok, cont, mods, times = next(iter(trainer.validation_dataloader))
        sample = trainer._prepare_longitudinal_prompt_sample(tok[0], cont[0], mods[0], times[0])
        queries = sample['target_logit_indices']
        first_end = queries[0] + 1
        inputs = [sample['tokens'][:-1][None], sample['continuous_tokens'][:-1][None],
                  sample['modalities'][None], sample['time_expanded'][None]]
        shared = sample['attention_mask'][None, None]
        previous = shared.clone()
        # Reconstruct the former training-only causal rows for future values.
        for row in range(first_end, shared.shape[-1], 2):
            previous[0, 0, row, :row + 1] = True
        results = []
        for mask in (previous, shared):
            torch.manual_seed(917)
            logits = trainer.model(*inputs, mask=mask)[:, queries]
            gradients = torch.autograd.grad(logits.square().mean(), tuple(trainer.model.parameters()),
                                            allow_unused=True)
            results.append((logits.detach(), gradients, torch.get_rng_state()))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        self.assertTrue(torch.equal(results[0][2], results[1][2]))
        for old, new in zip(results[0][1], results[1][1]):
            if old is None:
                self.assertIsNone(new)
            else:
                torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_sampler_uses_distinct_years_and_subset_indices(self):
        trainer = self.trainer()
        ds = trainer.dataset
        ds.time_expanded[0, :, 4] = 140  # Repeated markers, one year.
        ds.time_expanded[1, -1, 4] = 139  # Unordered.
        subset = Subset(ds, [0, 2, 1, 3])
        # The artifact was deliberately changed above; recompute its view metadata.
        lengths, multi_visit_rows = [], set()
        for local, row in enumerate(subset.indices):
            valid = ds.modalities[row] != ds.mask_modality_val
            lengths.append(int(valid.sum()))
            visits, _ = identify_visits(ds.modalities[row][valid], ds.time_expanded[row][valid],
                                       trainer.visit_marking, trainer.year_idx)
            if visits is not None:
                multi_visit_rows.add(local)
        all_indices = []
        for rank in (0, 1):
            sampler = EliFriendlyDistributedSampler(subset, 1, 2, rank,
                                                    lengths=lengths, multi_visit_rows=multi_visit_rows)
            self.assertEqual(sampler.multi_visit_indices, [1, 3])
            indices = list(sampler)
            self.assertIn(indices[0], [1, 3])
            all_indices.extend(indices)
        self.assertEqual(sorted(all_indices), [0, 1, 2, 3])

    def test_unified_padding_preserves_mixed_length_query_logits_and_gradients(self):
        trainer = self.trainer(n_layers=2, zero_init_mlp=False, dropout=.1)
        tok, cont, mods, times = next(iter(trainer.validation_dataloader))
        samples = [trainer._prepare_longitudinal_prompt_sample(tok[0], cont[0], mods[0], times[0]),
                   trainer._prepare_longitudinal_prompt_sample(tok[1, :43], cont[1, :43],
                                                              mods[1, :43], times[1, :43])]
        tokens, continuous, modalities, times, mask = pad_prompts(samples, trainer.mask_modality_val)
        legacy_tokens, legacy_continuous = tokens.clone(), continuous.clone()
        end = len(samples[1]['tokens']) - 1
        self.assertLess(end, tokens.shape[1])
        # Previously validation retained a short sample's final target in a padded row.
        legacy_tokens[1, end] = samples[1]['tokens'][-1]
        legacy_continuous[1, end] = samples[1]['continuous_tokens'][-1]
        results = []
        trainer.model.train()
        for tok_input, cont_input in ((legacy_tokens, legacy_continuous), (tokens, continuous)):
            torch.manual_seed(621)
            logits = trainer.model(tok_input, cont_input, modalities, times, mask=mask)
            queries = torch.cat([logits[i, sample['target_logit_indices']] for i, sample in enumerate(samples)])
            gradients = torch.autograd.grad(queries.square().mean(), tuple(trainer.model.parameters()),
                                            allow_unused=True)
            results.append((queries.detach(), gradients))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        for old, new in zip(results[0][1], results[1][1]):
            if old is None:
                self.assertIsNone(new)
            else:
                torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_visit_metadata_is_scanned_once_and_reused_by_dataset_views(self):
        with patch('healthformer.trainer.identify_visits', wraps=identify_visits) as scan:
            trainer = self.trainer()
        self.assertEqual(scan.call_count, len(trainer.dataset))
        expected = {i for i in range(len(trainer.dataset))
                    if identify_visits(trainer.dataset.modalities[i], trainer.dataset.time_expanded[i],
                                       trainer.visit_marking, trainer.year_idx)[0] is not None}
        self.assertEqual(trainer.multi_visit_rows, expected)
        self.assertEqual(trainer.validation_dataset_with_morethan1_visit.indices,
                         [i for i in trainer.validation_dataset_index if i in expected])
        self.assertEqual(trainer.train_sampler.multi_visit_indices,
                         [local for local, i in enumerate(trainer.train_dataset_index) if i in expected])

    def test_rolling_checkpoint_survives_interruption_and_resumes_exactly(self):
        settings = dict(epochs=2, batch_per_gpu=1, save=True, dropout=.1,
                        augmentation_chance=1., augmentation_rate=.5,
                        random_modality_subset_chance=.5, add_age_gender_as_pe=True,
                        Elis_augmentation_prob=1., elis_loss_weight=.01)
        self.fabric.seed_everything(42)
        baseline = self.trainer(**settings); baseline.train()
        expected = copy.deepcopy(baseline.model.state_dict())
        self.fabric.seed_everything(42)
        interrupted = self.trainer(**settings)
        save = interrupted.save_model
        class Interrupted(Exception): pass
        def save_then_interrupt(*args, **kwargs):
            path = save(*args, **kwargs)
            if interrupted.epoch == 1: raise Interrupted()
            return path
        interrupted.save_model = save_then_interrupt
        with self.assertRaises(Interrupted): interrupted.train()
        path = Path(self.overrides['checkpoint_dir'])/'model_last.pt'
        checkpoint = load_artifact(path)
        self.assertEqual(checkpoint['epoch'], 1)
        self.assertEqual(checkpoint['current_batch_index'], 0)
        resumed = self.trainer(**settings, resume_from=str(path)); resumed.train()
        for key, value in resumed.model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0, msg=key)
        self.assertEqual(load_artifact(path)['epoch'], 2)
        self.assertEqual(list(path.parent.glob('*.pt')), [path])

    def test_invalid_run_settings_fail_before_launch(self):
        from healthformer.launch import launch
        from train import preflight
        cases = (('epochs', 0), ('devices', None), ('devices', True),
                 ('log_wandb_metric_every_k', 0), ('accumulate_grad_batches', 1.5),
                 ('Elis_augmentation_prob', 2), ('random_removal_rate', -0.1),
                 ('not_elis_loss_weight', -1), ('not_elis_loss_weight', 0), ('lr', float('nan')),
                 ('elis_loss_weight', float('inf')), ('warmup_steps', -1), ('b1', 1.0))
        for key, value in cases:
            settings = {**config, **self.overrides, key: value}
            with self.subTest(key=key), patch('healthformer.launch.Fabric') as fabric:
                self.assertTrue(any(key in error for error in preflight(settings)))
                with self.assertRaisesRegex(ValueError, key):
                    launch(settings)
                fabric.assert_not_called()
        validate_methodology_config({**config, 'warmup_steps': 0, 'max_validation_batches': None,
                                     'stop_early_now': 0, 'elis_loss_weight': 0,
                                     'Elis_augmentation_prob': 0, 'augmentation_rate': 1})

    def test_prefix_validation_skips_targetless_batches_and_keeps_coverage(self):
        trainer = self.trainer(reduce_context_in_train=True, add_age_gender_as_pe=True)
        source = next(iter(trainer.validation_dataloader))
        batches = []
        for length in (0, 1, 2):
            batch = [x[:1].clone() for x in source]
            for x in batch:
                x[:, length:] = 0
            batch[2][:, length:] = trainer.mask_modality_val
            batches.append(batch)
        trainer.validation_dataloader = batches
        calls = []
        hook = trainer.model.register_forward_hook(lambda *args: calls.append(1))
        self.addCleanup(hook.remove)
        result = trainer.evaluate()
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['participants'], 3)
        self.assertEqual(result['eligible_targets'], 1)
        self.assertEqual(result['scored_targets'], 1)
        trainer.validation_dataloader = batches[:2]
        calls.clear()
        with self.assertRaisesRegex(RuntimeError, 'no valid target tokens'):
            trainer.evaluate()
        self.assertEqual(calls, [])

    def test_incompatible_model_dimensions_fail_before_launch(self):
        from healthformer.launch import launch
        from train import preflight
        cases = (
            ({'n_embd': 17}, 'token-embedding width'),
            ({'n_embd': 14}, 'token-embedding width'),
            ({'n_embd': 12}, 'token-embedding width'),
            ({'n_embd': 2, 'temporal_positional_encodings': False}, 'token-embedding width'),
            ({'n_heads': 3}, 'divisible by n_heads'),
            ({'n_embd': 18}, 'divisible by 16'),
            ({'n_embd': 8, 'temporal_positional_encodings': False}, 'divisible by 16'),
        )
        for overrides, message in cases:
            settings = {**config, **self.overrides, **overrides}
            with self.subTest(overrides=overrides), patch('healthformer.launch.Fabric') as fabric:
                self.assertTrue(any(message in error for error in preflight(settings)))
                with self.assertRaisesRegex(ValueError, message):
                    launch(settings)
                fabric.assert_not_called()

    def test_supported_dimension_boundaries_run_forward_and_backward(self):
        for width, temporal, sinusoidal, heads in (
                (16, True, True, 4), (18, True, False, 3),
                (4, False, False, 2), (32, False, True, 4)):
            with self.subTest(width=width, temporal=temporal, sinusoidal=sinusoidal, heads=heads):
                trainer = self.trainer(n_embd=width, n_heads=heads,
                                       temporal_positional_encodings=temporal,
                                       sinusoidal_time_embeddings=sinusoidal)
                tokens, continuous, modalities, times = next(iter(trainer.train_dataloader))
                loss, _ = trainer.compute_standard_loss(tokens[:, :-1], continuous,
                                                        tokens[:, 1:], modalities, times)
                self.assertTrue(torch.isfinite(loss))
                trainer.fabric.backward(loss)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters()
                                    if p.grad is not None))
