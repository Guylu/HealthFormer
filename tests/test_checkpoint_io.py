"""Checkpoint replacement, failure, and active-device RNG regressions."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from lightning.fabric import Fabric
from omegaconf import OmegaConf

from examples.make_training_fixture import create_fixture
from healthformer.config import config, validate_methodology_config
from healthformer.launch import launch
from healthformer.serialization import load_artifact
from healthformer.trainer import Trainer


class CheckpointIOTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        settings = json.loads(create_fixture(directory.name).read_text())
        self.capture = contextlib.ExitStack()
        self.addCleanup(self.capture.close)
        self.capture.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.capture.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.fabric = Fabric(accelerator='cpu', devices=1, precision='32-true')
        self.fabric.launch()
        self.settings = OmegaConf.create({**config, **settings, 'evaluate': False, 'save': False,
                                         'checkpoint_dir': str(self.root / 'checkpoints')})
        self.trainer = Trainer(self.settings, self.fabric)

    def test_successful_replacement_and_cpu_resume(self):
        trainer = self.trainer
        path = Path(trainer.save_model())
        old = path.read_bytes()
        trainer.train_epoch()
        trainer.save_model()
        self.assertNotEqual(path.read_bytes(), old)
        saved = load_artifact(path)
        self.assertEqual(saved['format_version'], 4)
        self.assertIsNone(saved['rng_state_cuda'])
        self.assertEqual(saved['epoch'], trainer.epoch)
        self.assertEqual(list(path.parent.iterdir()), [path])
        expected_rng = torch.get_rng_state().clone()
        resumed = Trainer(self.settings, self.fabric)
        resumed.load_model(path)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(resumed.it, trainer.it)
        self.assertEqual(resumed.epoch, trainer.epoch)
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)

    def test_write_replace_and_interruption_failures_preserve_previous_checkpoint(self):
        trainer = self.trainer
        path = Path(trainer.save_model())
        previous = path.read_bytes()
        trainer.train_epoch()
        def broken_write(state, stream):
            stream.write(b'incomplete checkpoint')
            raise OSError('injected write failure')
        for target, failure, error in (
            ('torch.save', broken_write, OSError),
            ('os.replace', OSError('injected replacement failure'), OSError),
            ('torch.save', KeyboardInterrupt(), KeyboardInterrupt),
        ):
            output = io.StringIO()
            with self.subTest(target=target, error=error), \
                    patch('healthformer.checkpointing.' + target, side_effect=failure), \
                    contextlib.redirect_stdout(output), self.assertRaises(error):
                trainer.save_model()
            self.assertNotIn('Saved ', output.getvalue())
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(load_artifact(path)['epoch'], 0)
            self.assertEqual(list(path.parent.iterdir()), [path])

    @unittest.skipIf(os.name != 'posix' or os.geteuid() == 0, 'Requires ordinary POSIX permissions')
    def test_real_directory_permission_failure_does_not_report_success(self):
        trainer = self.trainer
        path = Path(trainer.save_model())
        previous = path.read_bytes()
        trainer.train_epoch()
        output = io.StringIO()
        try:
            path.chmod(0o444)
            path.parent.chmod(0o555)
            with contextlib.redirect_stdout(output), self.assertRaises(PermissionError):
                trainer.save_model()
        finally:
            path.parent.chmod(0o755)
            path.chmod(0o644)
        self.assertNotIn('Saved ', output.getvalue())
        self.assertEqual(previous, path.read_bytes())
        self.assertEqual(load_artifact(path)['epoch'], 0)

    def test_old_format_and_bad_rng_rejected_before_mutation(self):
        saved = load_artifact(self.trainer.save_model())
        cases = []
        old = copy.deepcopy(saved); old['format_version'] = 3; old['rng_state_cuda'] = []
        cases.append(old)
        missing = copy.deepcopy(saved); del missing['rng_state_cuda']; cases.append(missing)
        wrong = copy.deepcopy(saved); wrong['rng_state_cuda'] = []; cases.append(wrong)
        distributed = copy.deepcopy(saved); distributed['rank_rng_states'] = [dict(saved), dict(saved)]
        cases.append(distributed)
        for checkpoint in cases:
            with self.subTest(keys=checkpoint.keys()), \
                    patch.object(self.trainer.model, 'load_state_dict') as model_load, \
                    patch.object(self.trainer.optimizer, 'load_state_dict') as optimizer_load, \
                    patch('torch.random.set_rng_state') as rng_restore, \
                    self.assertRaises((ValueError, RuntimeError)):
                self.trainer.load_model('unused', checkpoint)
            model_load.assert_not_called()
            optimizer_load.assert_not_called()
            rng_restore.assert_not_called()

    def test_cuda_rng_control_flow_uses_only_active_device(self):
        # CPU-host control-flow test; real CUDA generator behavior is tested below when available.
        trainer = self.trainer
        trainer.device = torch.device('cuda', 1)
        state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        tensor = torch.tensor
        def cpu_status(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return tensor(*args, **kwargs)
        with patch('torch.cuda.get_rng_state', return_value=state) as get, \
                patch('torch.cuda.get_rng_state_all', side_effect=AssertionError('all-device capture')), \
                patch('healthformer.checkpointing.torch.tensor', side_effect=cpu_status):
            path = trainer.save_model()
        get.assert_called_once_with(torch.device('cuda', 1))
        saved = load_artifact(path)
        self.assertTrue(torch.equal(saved['rng_state_cuda'], state))
        with patch('healthformer.checkpointing.torch.Generator') as generator, \
                patch('torch.cuda.set_rng_state') as restore, \
                patch('torch.cuda.set_rng_state_all', side_effect=AssertionError('all-device restore')):
            trainer.load_model(path, saved)
        restore.assert_called_once_with(saved['rng_state_cuda'], device=torch.device('cuda', 1))
        self.assertIn(unittest.mock.call(device=torch.device('cuda', 1)), generator.call_args_list)
        bad = copy.deepcopy(saved); bad['rng_state_cuda'] = [state]
        with patch.object(trainer.model, 'load_state_dict') as mutate, self.assertRaises(ValueError):
            trainer.load_model(path, bad)
        mutate.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_real_cuda_rng_roundtrip(self):
        fabric = Fabric(accelerator='cuda', devices=1, precision='32-true')
        fabric.launch()
        trainer = Trainer(OmegaConf.merge(self.settings, {'accelerator': 'cuda'}), fabric)
        path = trainer.save_model()
        expected = torch.rand(8, device=fabric.device)
        trainer.load_model(path)
        self.assertTrue(torch.equal(expected, torch.rand(8, device=fabric.device)))

    def test_unsafe_ddp_setting_rejected_before_launch_with_or_without_elis(self):
        for probability in (0., 1.):
            settings = dict(devices=2, find_unused_parameters=False, Elis_augmentation_prob=probability)
            with self.subTest(probability=probability), patch('healthformer.launch.Fabric') as fabric:
                with self.assertRaisesRegex(ValueError, 'find_unused_parameters=True'):
                    launch(settings)
                fabric.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'find_unused_parameters=True'):
                validate_methodology_config(settings)

    def test_save_disabled_launch_does_not_create_checkpoint_directory(self):
        with patch('healthformer.launch.Fabric'), patch('healthformer.launch.Trainer') as trainer:
            trainer.return_value.train.return_value = 0
            launch(dict(self.settings))
        self.assertFalse(Path(self.settings.checkpoint_dir).exists())
