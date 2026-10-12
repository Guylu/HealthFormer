"""Shared test namespace for patching globals across the implementation modules."""

import contextlib
import io
import os
import json
from pathlib import Path
import tempfile

class ImplementationNamespace:
    def __init__(self, modules):
        object.__setattr__(self, '_modules', modules)

    def __getattr__(self, name):
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        for module in self._modules:
            setattr(module, name, value)

    def __delattr__(self, name):
        for module in self._modules:
            if hasattr(module, name):
                delattr(module, name)


def load_implementation():
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from importlib import import_module
        load_modules = lambda: tuple(import_module("healthformer." + name) for name in ("config", "data", "model", "samplers", "inputs", "training", "evaluation", "longitudinal", "checkpointing", "trainer"))
        return ImplementationNamespace(load_modules())


def check_launch_wiring():
    from unittest.mock import patch
    from healthformer.launch import launch
    from healthformer.config import config
    events = []
    class FakeFabric:
        global_rank = 0
        world_size = 1
        def __init__(self, **kwargs): pass
        def launch(self): events.append('launch')
        def seed_everything(self, seed): pass
    class FakeTrainer:
        def __init__(self, settings, fabric):
            assert settings.allow_wandb is False
        def train(self): return 0
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory)/'keep.m'
        checkpoint.write_text('keep')
        with patch('healthformer.launch.Fabric', FakeFabric), patch('healthformer.launch.Trainer', FakeTrainer):
            assert launch(dict(config)) == 0
        assert checkpoint.read_text() == 'keep'
    print(json.dumps({'status': 'passed', 'real_training': False}))


if __name__ == '__main__':
    check_launch_wiring()
