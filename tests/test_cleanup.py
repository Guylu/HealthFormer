"""Contracts protected while sharing prompt preparation and update reporting."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from healthformer.prompts import prepare_prompt, pad_prompts
from healthformer.samplers import EliFriendlyDistributedSampler
from healthformer.tracking import tracker
from tests.test_regressions import _make_training_harness


class CleanupTests(unittest.TestCase):
    def test_prompt_selection_masks_budget_and_padding(self):
        owner = SimpleNamespace(device=torch.device('cpu'), mask_modality_val=3,
            visit_marking=2, year_idx=4, config=SimpleNamespace(max_seq_length=8),
            modality_name_to_number={'x_BloodTestsLoader': 0, 'x_NightingaleLoader': 1, 'marker': 2})
        tokens = torch.arange(10, 17, dtype=torch.int16)
        continuous = tokens.double() + 1/3
        modalities = torch.tensor([2, 0, 0, 2, 0, 1, 0], dtype=torch.int32)
        times = torch.zeros(7, 7, dtype=torch.int32)
        times[:, 4] = torch.tensor([140, 140, 140, 141, 141, 141, 141])
        for evaluation, expected in ((False, [14, 16]), (True, [14, 15, 16])):
            with self.subTest(evaluation=evaluation):
                coverage = dict(participants=0, eligible_targets=0, truncated_participants=0)
                sample = prepare_prompt(owner, tokens, continuous, modalities, times,
                                        evaluation=evaluation, coverage=coverage)
                self.assertEqual(sample['target_tokens'].tolist(), expected)
                self.assertEqual(sample['target_logit_indices'], list(range(2, len(expected)*2 + 1, 2)))
                self.assertEqual(coverage, dict(participants=1, eligible_targets=len(expected), truncated_participants=0))
                short = prepare_prompt(owner, tokens[:5], continuous[:5], modalities[:5], times[:5], evaluation=evaluation)
                inputs, conts, mods, _, mask = pad_prompts([sample, short], 3)
                self.assertEqual(inputs.dtype, torch.int64)
                self.assertEqual(mods.dtype, torch.int64)
                self.assertEqual(conts.dtype, torch.float64)
                torch.testing.assert_close(conts[0, :3], continuous[:3], rtol=0, atol=0)
                self.assertEqual(inputs[1, 3].item(), 0)
                for query in sample['target_logit_indices']:
                    allowed = mask[0, 0, query].nonzero().flatten().tolist()
                    self.assertEqual(allowed, list(range(3)) + ([query] if query >= 3 else []))
                self.assertEqual(mask[0, 0, 3].nonzero().flatten().tolist(), [3])
                self.assertTrue(mask.any(dim=-1).all())
        for limit, count in ((3, 0), (4, 1), (5, 1), (6, 2)):
            owner.config.max_seq_length = limit
            sample = prepare_prompt(owner, tokens, continuous, modalities, times)
            self.assertEqual(0 if sample is None else len(sample['target_tokens']), count)
            if sample is not None:
                self.assertLessEqual(len(sample['tokens']), limit)

    def test_sampler_partitions_small_populations_without_repair_or_rng_changes(self):
        for size in range(19):
            for world in (1, 2, 3):
                for batch_size in (1, 2, 4):
                    for multi in (set(), set(range(size)), set(range(0, size, 2))):
                        ranks = []
                        for rank in range(world):
                            sampler = EliFriendlyDistributedSampler(list(range(size)), batch_size, world, rank,
                                seed=17, lengths=[2 + i % 5 for i in range(size)], multi_visit_rows=multi)
                            sampler.set_epoch(2)
                            rng = torch.get_rng_state()
                            indices = list(sampler)
                            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                            self.assertEqual(indices, list(sampler))
                            self.assertEqual(len(indices), len(sampler))
                            ranks.append(indices)
                        flattened = [i for indices in ranks for i in indices]
                        self.assertEqual(len(flattened), size // world * world)
                        self.assertEqual(len(flattened), len(set(flattened)))
                        self.assertTrue(all(0 <= i < size for i in flattened))
                        self.assertEqual(len({len(indices) for indices in ranks}), 1)

    def test_sampler_partition_error_fails_on_every_rank_before_yield(self):
        for rank in range(3):
            sampler = EliFriendlyDistributedSampler(list(range(8)), 2, 3, rank,
                                                    lengths=[2]*8, multi_visit_rows={1, 3})
            # Simulate an internal partitioning error, not a small valid dataset.
            sampler.single_visit_indices = []
            with self.assertRaisesRegex(RuntimeError, 'unexpected rank lengths'):
                iter(sampler)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            EliFriendlyDistributedSampler([0, 1], 1, 1, 0, lengths=[2], multi_visit_rows=set())

    def test_update_report_averages_only_accepted_window(self):
        for rejected, run_elis in ((None, True), (2, True), (None, False)):
            with self.subTest(rejected=rejected, run_elis=run_elis):
                trainer, _ = _make_training_harness(5, 4, standard_nonfinite_step=rejected, run_elis=run_elis)
                values = {}
                standard, elis = trainer.compute_standard_loss, trainer.compute_elis_loss
                def capture_standard(*args, **kwargs):
                    result = standard(*args, **kwargs)
                    values.setdefault(trainer.training_step, {})['standard'] = float(result[0].detach())
                    return result
                def capture_elis(*args, **kwargs):
                    result = elis(*args, **kwargs)
                    values.setdefault(trainer.training_step, {})['elis'] = float(result[0].detach())
                    return result
                trainer.compute_standard_loss = capture_standard
                trainer.compute_elis_loss = capture_elis
                logs = []
                with patch.object(tracker, 'log', side_effect=logs.append), contextlib.redirect_stdout(io.StringIO()):
                    trainer.train_epoch()
                reports = [row for row in logs if 'Training Loss/' in row]
                windows = ([3, 4],) if rejected is not None else ([0, 1, 2, 3], [4])
                self.assertEqual(len(reports), len(windows))
                for report, window in zip(reports, windows):
                    standard_mean = sum(values[i]['standard'] for i in window) / len(window)
                    elis_mean = sum(values[i].get('elis', 0) for i in window) * .01 / len(window)
                    self.assertAlmostEqual(report['Training Losses/Standard weighted contribution'], standard_mean)
                    self.assertAlmostEqual(report['Training Losses/ELIS weighted contribution'], elis_mean)
                    self.assertAlmostEqual(report['Training Loss/'], standard_mean + elis_mean)
                    self.assertIn('misc/Gradient Norm', report)
                    self.assertIn('misc/Learning rate', report)
                self.assertEqual([r['misc/Iteration'] for r in reports], list(range(1, trainer.it + 1)))
                self.assertFalse(any(key.startswith(('My_losses/', 'embed_skip_scale/')) for row in logs for key in row))

    def test_console_progress_follows_completed_updates(self):
        for accumulation, batches in ((1, 20), (2, 40), (3, 59)):
            with self.subTest(accumulation=accumulation):
                trainer, _ = _make_training_harness(batches, accumulation)
                trainer.config.log_wandb_metric_every_k = 10
                with patch.object(tracker, 'log'), contextlib.redirect_stdout(io.StringIO()) as output:
                    trainer.train_epoch()
                lines = [line for line in output.getvalue().splitlines() if 'training loss' in line]
                self.assertEqual(trainer.it, 20)
                self.assertEqual(len(lines), 2)
                self.assertIn('update 10,', lines[0])
                self.assertIn('update 20,', lines[1])
