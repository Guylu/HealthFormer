#!/usr/bin/env python3
"""Inspect settings, check prerequisites, or launch HealthFormer training."""

import argparse
import importlib.util
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
REQUIRED_IMPORTS = ('torch', 'numpy', 'scipy', 'omegaconf', 'lightning', 'tqdm')


def read_config(path=None):
    # Load settings without importing the training stack.
    settings = runpy.run_path(str(ROOT/'healthformer/config.py'))
    config = settings['config']
    if path is not None:
        overrides = json.loads(Path(path).read_text())
        if not isinstance(overrides, dict):
            raise ValueError('The configuration file must contain a JSON object.')
        overrides = settings['normalize_config'](overrides)
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError('Unknown training settings: ' + ', '.join(sorted(unknown)))
        config.update(overrides)
    return config


def artifact_path(config):
    if config.get('data_path'):
        return Path(config['data_path'])
    raise ValueError('Provide data_path explicitly.')


def preflight(config):
    problems = []
    try:
        runpy.run_path(str(ROOT/'healthformer/config.py'))['validate_methodology_config'](config)
    except ValueError as exc:
        problems.append(str(exc))
    if config['allow_wandb'] and not (config['wandb_entity'] and config['wandb_project']):
        problems.append('Online tracking requires wandb_entity and wandb_project.')
    for field in ('resume_from', 'initialize_from'):
        if config[field] and not Path(config[field]).is_file():
            problems.append('Requested checkpoint does not exist: ' + config[field])
    required = REQUIRED_IMPORTS + (('wandb',) if config['allow_wandb'] else ())
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        problems.append('Missing imports: ' + ', '.join(missing))
    if 'torch' not in missing:
        try:
            import torch
            if config['accelerator'] == 'cuda' and not torch.cuda.is_available():
                problems.append('The trainer launches Fabric with CUDA; this environment has no available CUDA device.')
        except Exception as exc:
            problems.append('PyTorch import failed: ' + str(exc))
    try:
        data = artifact_path(config)
        if not data.is_file():
            problems.append('Training artifact not found: ' + str(data))
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='HealthFormer: inspect settings or run training.',
        epilog='Read TRAINING.md for data requirements, checkpoint continuation, and completion behavior.',
    )
    parser.add_argument('--config', type=Path, help='JSON overrides merged into the training defaults')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--show-config', action='store_true', help='print resolved settings without importing the training stack')
    actions.add_argument('--check', action='store_true', help='check dependencies, CUDA availability, and required file paths')
    actions.add_argument('--run', action='store_true', help='launch training after checking local prerequisites')
    args = parser.parse_args(argv)
    if not (args.show_config or args.check or args.run):
        parser.print_help()
        return 0
    try:
        config = read_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.show_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    problems = preflight(config)
    if problems:
        print('Prerequisites are incomplete:')
        for problem in problems:
            print('  - ' + problem)
        return 2
    if args.check:
        print('Dependency discovery, selected accelerator, and file paths passed.')
        print('Artifact contents, checkpoints and training behavior have not been validated by this check.')
        return 0
    from healthformer.launch import launch
    return launch(config)


if __name__ == '__main__':
    raise SystemExit(main())
