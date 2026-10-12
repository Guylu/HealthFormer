"""Numerical and alignment regressions from the independent release review."""
import contextlib
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
from healthformer.launch import ContinuousInputPrecision, launch
from healthformer.model import ContinuousValueEncoder, MultiModalTransformer
from healthformer.prompts import prepare_prompt
from healthformer.trainer import Trainer
from tests.test_regressions import make_dataset


class EncodingProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.continuous_encoder = ContinuousValueEncoder(4)
        self.continuous_min_max_stats = {'value': {'min': 100., 'max': 101.}}
        self.modality_name_to_number = {'value': 0}
        self.projection = torch.nn.Linear(4, 2)

    def forward(self, values, modalities):
        encoded = MultiModalTransformer._build_continuous_value_embedding(
            self, values, modalities, values.shape[1])
        return encoded, self.projection(encoded)


class PrecisionAndAugmentationTests(unittest.TestCase):
    def test_launch_selects_input_preserving_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            for precision in ('32-true', 'bf16-mixed'):
                with self.subTest(precision=precision), \
                        patch('healthformer.launch.Fabric') as fabric, \
                        patch('healthformer.launch.Trainer') as trainer:
                    trainer.return_value.train.return_value = 0
                    launch({'accelerator': 'cuda', 'precision': precision,
                            'checkpoint_dir': directory, 'allow_wandb': False})
                    kwargs = fabric.call_args.kwargs
                    if precision == 'bf16-mixed':
                        self.assertIsInstance(kwargs['plugins'], ContinuousInputPrecision)
                        self.assertEqual(kwargs['plugins'].device, 'cuda')
                        self.assertIsNone(kwargs['precision'])
                    else:
                        self.assertIsNone(kwargs['plugins'])
                        self.assertEqual(kwargs['precision'], '32-true')

    def test_fabric_preserves_measurements_but_autocasts_projection(self):
        # CPU exercises the BF16 plugin, not the CLI's supported CPU precision.
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else [])
        for device in devices:
            with self.subTest(device=device):
                fabric = Fabric(accelerator=device, devices=1,
                                plugins=ContinuousInputPrecision('bf16-mixed', device))
                fabric.launch()
                raw = EncodingProbe().to(device)
                values = torch.tensor([[100.1, 100.2]], device=device)
                modalities = torch.zeros_like(values, dtype=torch.long)
                reference = raw(values, modalities)[0]
                projection_dtypes = []
                handle = raw.projection.register_forward_hook(
                    lambda module, args, result: projection_dtypes.append(result.dtype))
                try:
                    wrapped = fabric.setup_module(raw)
                    for encoded, output in (wrapped(values, modalities),
                                            wrapped(values=values, modalities=modalities)):
                        torch.testing.assert_close(encoded, reference, rtol=0, atol=0)
                        self.assertFalse(torch.equal(encoded[0, 0], encoded[0, 1]))
                        self.assertEqual(output.dtype, torch.float32)
                    self.assertEqual(projection_dtypes, [torch.bfloat16, torch.bfloat16])
                finally:
                    handle.remove()

    def test_mae_loss_and_gradient_match_fp32_under_autocast(self):
        cases = [([300., 302.], 301., 301., 1., torch.float32),
                 ([100.1, 100.2], 100.2, 100.15, .05, torch.float32),
                 ([300., 302.], 301.000001, 301., 1., torch.float64)]
        for midpoints, target, mean, std, dtype in cases:
            owner = SimpleNamespace(modality_name_to_number={'value': 0},
                mod2bininfo={'value': ([0, 1], midpoints)},
                mod2stats={'value': {'mean': mean, 'std': std}},
                config=SimpleNamespace(mae_loss_scale=1.))
            results = []
            for enabled in (False, True):
                logits = torch.zeros(1, 2, requires_grad=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=enabled):
                    loss = MultiModalTransformer.handle_continuous_loss(
                        owner, logits, torch.tensor(0.), {}, torch.tensor([target], dtype=dtype),
                        torch.zeros(1, dtype=torch.long), 'Train')
                loss.backward()
                results.append((loss.detach(), logits.grad))
            for actual, expected in zip(results[1], results[0]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if target == mean:
                self.assertEqual(results[0][0].item(), 0.)
                self.assertEqual(results[0][1].abs().sum().item(), 0.)
            else:
                self.assertGreater(results[0][1].abs().sum().item(), 0.)

    def test_block_removal_is_padding_invariant_and_preserves_alignment(self):
        ds = make_dataset(random_block_removal_number=2, random_block_removal_rate=.2,
                          random_removal_rate=.2)
        for preceding_removal in (False, True):
            for seed in range(10):
                results = []
                for width in (10, 100):
                    tokens = torch.zeros(width, dtype=torch.int16)
                    tokens[:10] = torch.arange(1, 11)
                    continuous = tokens.double() + .25
                    modalities = torch.full((width,), 2, dtype=torch.int16)
                    modalities[:10] = 0
                    modalities[[0, 5]] = 1
                    times = torch.arange(width * 7, dtype=torch.int32).reshape(width, 7)
                    channels = (tokens, continuous, modalities, times)
                    torch.manual_seed(seed)
                    if preceding_removal:
                        channels = ds.remove_random_tokens_and_pad(*channels)
                    actual = ds.remove_random_token_block_and_pad(*channels)
                    real = actual[2] != ds.mask_modality_val
                    kept = actual[0][real].long() - 1
                    for original, value in zip((tokens, continuous, modalities, times), actual):
                        self.assertEqual(len(value), width)
                        self.assertEqual(value.dtype, original.dtype)
                        torch.testing.assert_close(value[real], original[kept])
                    self.assertEqual(actual[0][actual[2] == 1].tolist(), [1, 6])
                    results.append(tuple(value[real] for value in actual))
                for a, b in zip(*results):
                    torch.testing.assert_close(a, b)

    def test_zero_block_count_is_noop_without_rng_draws(self):
        ds = make_dataset(random_block_removal_number=0, random_block_removal_rate=1.)
        channels = (ds.tokens[0], ds.continuous_tokens[0], ds.modalities[0], ds.time_expanded[0])
        before = torch.get_rng_state()
        for actual, original in zip(ds.remove_random_token_block_and_pad(*channels), channels):
            self.assertIs(actual, original)
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_prompt_selection_preserves_repeated_targets_and_channel_alignment(self):
        owner = SimpleNamespace(mask_modality_val=5, visit_marking=2, year_idx=4,
            device=torch.device('cpu'), config=SimpleNamespace(max_seq_length=64),
            modality_name_to_number={'x_BloodTestsLoader': 3, 'x_NightingaleLoader': 4})
        for dtype in (torch.int16, torch.int32, torch.int64):
            mods = torch.tensor([2, 3, 0, 3, 0, 2, 3, 3, 4, 0, 3, 5], dtype=dtype)
            tokens = torch.arange(12, dtype=dtype)
            continuous = tokens.float() + .25
            times = torch.zeros(12, 7, dtype=torch.int32)
            times[5:, 4] = 1
            for evaluation, selected in ((False, [6, 7, 10]), (True, [6, 7, 8, 10])):
                sample = prepare_prompt(owner, tokens, continuous, mods, times, evaluation=evaluation)
                self.assertIsNotNone(sample)
                for key, values in (('target_tokens', tokens), ('target_continuous', continuous),
                                    ('target_modalities', mods)):
                    torch.testing.assert_close(sample[key], values[selected])
            owner.modality_name_to_number = {}
            self.assertIsNone(prepare_prompt(owner, tokens, continuous, mods, times))
            owner.modality_name_to_number = {'x_BloodTestsLoader': 3, 'x_NightingaleLoader': 4}

    def test_actual_standard_and_elis_backwards_under_bf16_plugin(self):
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            settings = json.loads(create_fixture(directory).read_text())
            settings.update(Elis_augmentation_prob=1., elis_loss_weight=1., evaluate=False, save=False)
            # Keep the CPU launch contract unchanged while directly testing the plugin.
            fabric = Fabric(accelerator='cpu', devices=1,
                            plugins=ContinuousInputPrecision('bf16-mixed', 'cpu'))
            fabric.launch()
            trainer = Trainer(OmegaConf.create({**config, **settings}), fabric)
            tokens, continuous, modalities, times = next(iter(trainer.train_dataloader))
            args = (tokens[:, :-1], continuous, tokens[:, 1:], modalities, times)
            standard, _ = trainer.compute_standard_loss(*args)
            elis, accepted = trainer.compute_elis_loss(*args)
            self.assertTrue(accepted)
            for loss in (standard, elis):
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
            fabric.backward(standard + elis)
            gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(g.abs().sum() > 0 for g in gradients))


if __name__ == '__main__':
    unittest.main()
