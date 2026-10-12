"""Public training path, split migration, serialization, and logging checks."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from lightning.fabric import Fabric
from omegaconf import OmegaConf
from examples.make_training_fixture import create_fixture
from healthformer.config import config, is_validation_id
from healthformer.serialization import load_artifact, to_safe
from healthformer.tracking import Tracker
from healthformer.trainer import Trainer


class ReleaseTests(unittest.TestCase):
    def test_corrected_split_and_historical_boundary_are_explicit(self):
        for policy, expected in [('modulo', 20), ('historical_modulo_inclusive', 21)]:
            cfg = SimpleNamespace(percent_validation=20, split_policy=policy)
            self.assertEqual(sum(is_validation_id(i, cfg) for i in range(100)), expected)
        for value in (-1, 101, True, 2.5):
            with self.assertRaises(ValueError):
                is_validation_id(20, SimpleNamespace(percent_validation=value, split_policy='modulo'))

    def test_safe_numpy_rng_round_trip_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.pt'
            rng = np.random.RandomState(42).get_state()
            torch.save(to_safe({'rng': rng}), path)
            restored = load_artifact(path)['rng']
            np.testing.assert_array_equal(rng[1], restored[1])
            torch.save(SimpleNamespace(value=1), path)
            with self.assertRaises(pickle.UnpicklingError):
                load_artifact(path)

    def test_application_tracking_filters_non_scalar_metrics(self):
        tracker = Tracker()
        tracker.run = Mock()
        tracker.backend = Mock()
        tracker.log({'loss': 2., 'trajectory': [1, 2], 'table': object()})
        tracker.run.log.assert_called_once_with({'loss': 2.})

    @unittest.skipUnless(importlib.util.find_spec('wandb'), 'Optional W&B SDK is not installed.')
    def test_offline_tracking_records_metrics_without_automatic_capture(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'WANDB_ERROR_REPORTING': 'false', 'WANDB_CACHE_DIR': directory}):
            import wandb
            from wandb.proto.wandb_internal_pb2 import Record
            from wandb.sdk.internal.datastore import DataStore

            init = wandb.init

            def offline_init(**kwargs):
                # Exercise production capture settings without uploading a run.
                kwargs.update(mode='offline', dir=directory)
                return init(**kwargs)

            tracker = Tracker()
            try:
                with patch.object(wandb, 'init', side_effect=offline_init):
                    tracker.start(SimpleNamespace(allow_wandb=True,
                        wandb_entity='offline-check', wandb_project='healthformer-check'))
                print('HEALTHFORMER_CONSOLE_CAPTURE_PROBE', flush=True)
                tracker.log({'loss': 1.25, 'participant_rows': [100, 101]})
            finally:
                tracker.finish()
                wandb.teardown()

            files = [path for path in Path(directory).rglob('*') if path.is_file()]
            records = [path for path in files if path.suffix == '.wandb']
            self.assertEqual(len(records), 1)
            store = DataStore()
            store.open_for_scan(str(records[0]))
            history = {}
            try:
                while (raw := store.scan_data()) is not None:
                    record = Record()
                    record.ParseFromString(raw)
                    self.assertNotIn(record.WhichOneof('record_type'),
                                     ('output', 'output_raw', 'stats', 'environment'))
                    for item in record.history.item:
                        history[item.key or '.'.join(item.nested_key)] = json.loads(item.value_json)
            finally:
                store.close()
            self.assertEqual(history['loss'], 1.25)
            self.assertNotIn('participant_rows', history)
            for path in files:
                self.assertNotIn(path.name, ('output.log', 'wandb-metadata.json',
                                            'requirements.txt', 'diff.patch'))
                self.assertNotIn(b'HEALTHFORMER_CONSOLE_CAPTURE_PROBE', path.read_bytes())

    def test_import_does_not_patch_external_libraries_or_environment(self):
        source = '''import os, torch
from importlib.util import find_spec
wandb = __import__('wandb') if find_spec('wandb') is not None else None
before = dict(os.environ)
numpy = torch.Tensor.numpy
if wandb is not None:
    log, scatter, histogram = wandb.log, wandb.plot.scatter, wandb.Histogram
import healthformer.trainer
assert torch.Tensor.numpy is numpy
if wandb is not None:
    assert wandb.log is log and wandb.plot.scatter is scatter and wandb.Histogram is histogram
for key in ('WANDB_IGNORE_GLOBS', 'PYTORCH_ALLOC_CONF'):
    assert os.environ.get(key) == before.get(key)
'''
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_elis_and_longitudinal_paths_use_distinct_targets(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            overrides = json.loads(create_fixture(directory).read_text())
            settings = OmegaConf.create({**config, **overrides, 'evaluate': False,
                                        'Elis_augmentation_prob': 1, 'elis_loss_weight': 0.01})
            fabric = Fabric(accelerator='cpu', devices=1, precision='32-true')
            fabric.launch()
            fabric.seed_everything(settings.seed)
            trainer = Trainer(settings, fabric)
            batch = next(iter(trainer.validation_dataloader))
            tokens, continuous, modalities, times = batch
            trainer.model.eval()
            with torch.no_grad():
                expected_logits = trainer.model(tokens[:, :-1], continuous, modalities, times)
            self.assertTrue(torch.isfinite(expected_logits).all())
            trainer.model.train()
            original = trainer.compute_elis_loss
            losses = []
            def record(*args, **kwargs):
                result = original(*args, **kwargs)
                if result[1]:
                    losses.append(float(result[0].detach()))
                return result
            trainer.compute_elis_loss = record
            trainer.train_epoch()
            self.assertTrue(losses, 'The real ELIS branch must execute on this fixture.')
            self.assertTrue(all(np.isfinite(x) for x in losses))
            metrics = trainer.evaluate_longitudinal(trainer.validation_dataloader_with_morethan1_visit)
            pairs = [(np.asarray(info['expected_vals']), np.asarray(info['real_vals']))
                     for info in metrics.values() if info.get('expected_vals')]
            self.assertTrue(pairs, 'Longitudinal evaluation must produce continuous targets.')
            self.assertTrue(any(np.any(a != b) for a, b in pairs))

    def test_real_training_checkpoint_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            overrides = json.loads(create_fixture(directory).read_text())
            settings = OmegaConf.create({**config, **overrides, 'epochs': 2, 'evaluate': False, 'batch_per_gpu': 1})
            fabric = Fabric(accelerator='cpu', devices=1, precision='32-true')
            fabric.launch()
            fabric.seed_everything(settings.seed)
            baseline = Trainer(settings, fabric)
            initial = {k: v.clone() for k, v in baseline.model.state_dict().items()}
            baseline.train()
            expected = {k: v.clone() for k, v in baseline.model.state_dict().items()}
            self.assertTrue(any(not torch.equal(initial[k], expected[k]) for k in expected))
            self.assertTrue(all(torch.isfinite(v).all() for v in expected.values()))
            fabric.seed_everything(settings.seed)
            interrupted = Trainer(settings, fabric)
            interrupted.train_epoch()
            saved = interrupted.save_model(id='interrupted')
            self.assertTrue(Path(saved).exists())
            settings.resume_from = saved
            resumed = Trainer(settings, fabric)
            self.assertEqual(resumed.it, interrupted.it)
            resumed.train()
            for key, value in resumed.model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0, msg=key)
            metrics = resumed.evaluate()
            self.assertTrue(np.isfinite(metrics['cross_entropy']))
            self.assertEqual(metrics['target_count'], 189)
            self.assertEqual(resumed.best_val_loss, metrics['cross_entropy'])
            resumed.best_val_loss = 0.0
            resumed.evaluate()
            self.assertEqual(resumed.best_val_loss, 0.0)
            resumed.config.max_validation_batches = 1
            self.assertEqual(resumed.evaluate()['target_count'], 63)
            self.assertTrue(Path(saved).exists())
            settings.split_policy = 'historical_modulo_inclusive'
            with self.assertRaisesRegex(RuntimeError, 'split_policy'):
                Trainer(settings, fabric)
            settings.split_policy = 'modulo'
            settings.resume_from = str(Path(directory)/'missing.pt')
            with self.assertRaisesRegex(RuntimeError, 'Requested resume failed'):
                Trainer(settings, fabric)
