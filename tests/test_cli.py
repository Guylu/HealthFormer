"""Configuration, command-line, and launch integration tests."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_file('healthformer_launcher_under_test', ROOT/'train.py')


class CommandLineTests(unittest.TestCase):
    def test_training_error_survives_tracking_cleanup_failure(self):
        from healthformer.launch import launch
        settings = {'accelerator': 'cpu', 'precision': '32-true'}
        for error in (ValueError('training failed'), KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(error=type(error).__name__), \
                    patch('healthformer.launch.Fabric'), \
                    patch('healthformer.launch.tracker.start'), \
                    patch('healthformer.launch.Trainer') as trainer, \
                    patch('healthformer.launch.tracker.finish',
                          side_effect=OSError('tracking shutdown failed')) as finish:
                trainer.return_value.train.side_effect = error
                with self.assertLogs('healthformer.launch', level='ERROR') as logs:
                    with self.assertRaises(type(error)) as caught:
                        launch(settings)
                self.assertIs(caught.exception, error)
                finish.assert_called_once_with(exit_code=1)
                self.assertIn('tracking shutdown failed', logs.output[0])

    def test_tracking_cleanup_failure_after_success_is_reported(self):
        from healthformer.launch import launch
        with patch('healthformer.launch.Fabric'), \
                patch('healthformer.launch.tracker.start'), \
                patch('healthformer.launch.Trainer') as trainer, \
                patch('healthformer.launch.tracker.finish',
                      side_effect=OSError('tracking shutdown failed')) as finish:
            trainer.return_value.train.return_value = 0
            with self.assertRaisesRegex(OSError, 'tracking shutdown failed'):
                launch({'accelerator': 'cpu', 'precision': '32-true'})
            finish.assert_called_once_with(exit_code=0)

    def test_launch_passes_explicit_settings_and_preserves_checkpoints(self):
        result = subprocess.run([sys.executable, '-m', 'tests._helpers'], cwd=ROOT,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])['status'], 'passed')

    def test_help_and_config_work_without_site_packages(self):
        for flags in (['--help'], ['--show-config']):
            result = subprocess.run([sys.executable, '-S', str(ROOT/'train.py')] + flags,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            if flags == ['--show-config']:
                self.assertEqual(json.loads(result.stdout)['accumulate_grad_batches'], 1)

    def test_override_merges_without_mutating_defaults(self):
        before = launcher.read_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text('{"epochs": 3, "allow_wandb": false}')
            merged = launcher.read_config(path)
        self.assertEqual(merged['epochs'],3)
        self.assertFalse(merged['allow_wandb'])
        self.assertEqual(merged['accumulate_grad_batches'], before['accumulate_grad_batches'])
        self.assertEqual(launcher.read_config(), before)

    def test_legacy_spellings_resolve_before_defaults_and_detect_conflicts(self):
        from healthformer.config import CONFIG_ALIASES
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for old, new in CONFIG_ALIASES.items():
                value = 17 if new == 'elis_max_input_length' else -1.5 if new.endswith('init_bias') else False
                with self.subTest(setting=old):
                    path.write_text(json.dumps({old: value}))
                    resolved = launcher.read_config(path)
                    self.assertNotIn(old, resolved)
                    self.assertEqual(resolved[new], value)
                    path.write_text(json.dumps({old: value, new: value}))
                    self.assertEqual(launcher.read_config(path), resolved)
                    path.write_text(json.dumps({old: value, new: launcher.read_config()[new]}))
                    with self.assertRaisesRegex(ValueError, 'Conflicting configuration'):
                        launcher.read_config(path)


    def test_removed_settings_are_rejected(self):
        for key in ('percent_test', 'floating_point_type', 'take_loss_only_on_mod_vocab',
                    'eval_len_cut_off', 'cluster_job', 'oci_run', 'preempt',
                    'upload_checkpoints', 'data_version', 'compile', 'train_data_percentage',
                    'train_multi_visit_only', 'trust_legacy_artifacts', 'use_dedicated_sink_token',
                    'use_modality_specific_mlp_for_cont_pe', 'continuous_pe_base_dim',
                    'do_liors_matrix', 'replace_gelu_with_squared_relu', 'clip_to',
                    'clip_to_context_len_15k', 'optimizer', 'visit_definition'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps({key: 1}))
                with self.assertRaisesRegex(ValueError, 'Unknown training settings'):
                    launcher.read_config(path)

    def test_data_path_is_explicit(self):
        config = launcher.read_config()
        with self.assertRaisesRegex(ValueError, 'Provide data_path'):
            launcher.artifact_path(config)
        config['data_path'] = 'custom/training.pt'
        self.assertEqual(launcher.artifact_path(config), Path('custom/training.pt'))

    def test_unknown_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text('{"epohcs": 3}')
            with self.assertRaisesRegex(ValueError,'Unknown training settings'):
                launcher.read_config(path)


    def test_tracking_dependency_is_required_only_when_enabled(self):
        config = {**launcher.read_config(), 'accelerator': 'cpu', 'precision': '32-true',
                  'data_path': str(ROOT/'train.py')}
        original = importlib.util.find_spec
        with patch.object(importlib.util, 'find_spec',
                          side_effect=lambda name: None if name == 'wandb' else original(name)):
            self.assertEqual(launcher.preflight(config), [])
            config.update(allow_wandb=True, wandb_entity='example', wandb_project='example')
            self.assertIn('Missing imports: wandb', launcher.preflight(config))


if __name__ == '__main__':
    unittest.main()
