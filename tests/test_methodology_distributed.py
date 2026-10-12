"""Two-worker check for synchronized validation failures."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from healthformer.evaluation import EvaluationMixin
from tests.test_distributed import run_spawned_processes_bounded


class Adapter:
    def autocast(self):
        return contextlib.nullcontext()

    device = torch.device('cpu')
    world_size = 2

    def __init__(self, rank): self.global_rank = rank
    def backward(self, loss, **kwargs): loss.backward(**kwargs)
    def barrier(self): dist.barrier()
    def all_reduce(self, tensor, reduce_op='sum'):
        dist.all_reduce(tensor, op={'sum': dist.ReduceOp.SUM, 'max': dist.ReduceOp.MAX}[reduce_op])
        return tensor


class InvalidLogits(torch.nn.Module):
    def __init__(self, rank):
        super().__init__(); self.rank = rank
    def forward(self, tokens, *args, **kwargs):
        logits = torch.zeros(*tokens.shape, 2)
        if self.rank == 1: logits[0, 0, 0] = float('nan')
        return logits


def worker(rank, init_file):
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=2)
    try:
        fabric = Adapter(rank)
        owner = EvaluationMixin()
        owner.fabric, owner.device = fabric, fabric.device
        owner.config = SimpleNamespace(validation_sequence_policy='prefix', validation_max_tokens=8,
                                       max_seq_length=8,
                                       reduce_context_in_train=False, loss_vocabulary='global')
        owner.model = InvalidLogits(rank)
        owner.mask_modality_val = 1
        owner.dataset = SimpleNamespace(modalities=torch.zeros(2, 4, dtype=torch.long))
        owner.bins = {}
        owner.modality_name_to_number = {'a': 0}
        owner.number_to_modality_name = {0: 'a'}
        batch = (torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4),
                 torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, 7))
        owner.validation_dataloader = [batch]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                owner.evaluate()
            except RuntimeError as exc:
                assert 'rejected 1 non-finite' in str(exc), str(exc)
            else:
                raise AssertionError('Partial non-finite validation must fail on both workers.')
        assert owner.last_validation_summary['valid'] is False
        assert owner.last_validation_summary['scored_targets'] == 6
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DistributedMethodologyTests(unittest.TestCase):
    def test_one_worker_validation_failure_is_synchronized(self):
        with tempfile.TemporaryDirectory() as directory:
            run_spawned_processes_bounded(worker, (str(Path(directory)/'init'),), timeout=60)


class LongitudinalLogits(torch.nn.Module):
    def forward(self, tokens, *args, **kwargs):
        return torch.zeros(*tokens.shape, 16)


def longitudinal_worker(rank, init_file):
    from healthformer.longitudinal import LongitudinalMixin
    from tests import test_regressions as fixtures
    from tests.test_distributed import DDPFabricAdapter
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=2)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            owner = LongitudinalMixin()
            owner.fabric = Adapter(rank); owner.device = torch.device('cpu')
            owner.config = SimpleNamespace(max_seq_length=8, loss_vocabulary='global')
            owner.model = LongitudinalLogits()
            owner.mask_modality_val = 2; owner.visit_marking = 0; owner.year_idx = 4
            owner.modality_name_to_number = {'m_DEXALoader': 0, 'm_BloodTestsLoader': 1}
            owner.number_to_modality_name = {v: k for k, v in owner.modality_name_to_number.items()}
            owner.mod2bininfo = {}; owner.cat_cols = list(owner.modality_name_to_number)
            tok = torch.ones(1, 4, dtype=torch.long); mods = torch.tensor([[0, 1, 0, 1]])
            time = torch.zeros(1, 4, 7, dtype=torch.long); time[:, :, 4] = torch.tensor([140, 140, 141, 141])
            batch = (tok, tok.float(), mods, time)
            # An empty rank must still participate; only the global absence of targets is an error.
            owner.evaluate_longitudinal([] if rank == 0 else [batch])
            assert owner.last_longitudinal_summary['valid']
            assert owner.last_longitudinal_summary['scored_targets'] == 2
            try: owner.evaluate_longitudinal([])
            except RuntimeError as exc: assert 'no eligible targets' in str(exc)
            else: raise AssertionError('Globally empty ELIS validation must fail on both ranks.')
            trainer, cursor = fixtures._make_training_harness(1, 1)
            trainer.fabric = DDPFabricAdapter(rank, 2)
            batch = list(fixtures._training_batches(1)[0])
            if rank == 1: batch[2].fill_(trainer.mask_modality_val)
            trainer._active_train_dataloader = lambda: [tuple(batch)]
            try: trainer.train_epoch()
            except RuntimeError as exc: assert 'no prediction targets' in str(exc)
            else: raise AssertionError('A targetless rank must stop both workers before forward.')
            assert trainer.it == 0 and cursor.next_batch_index == 0
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DistributedReviewTests(unittest.TestCase):
    def test_empty_longitudinal_and_targetless_training_ranks_synchronize(self):
        with tempfile.TemporaryDirectory() as directory:
            run_spawned_processes_bounded(longitudinal_worker, (str(Path(directory)/'init'),), timeout=60)

    def test_training_reports_worker_mean_with_missing_components_and_short_window(self):
        with tempfile.TemporaryDirectory() as directory:
            run_spawned_processes_bounded(training_metrics_worker, (str(Path(directory)/'init'),), timeout=60)


def training_metrics_worker(rank, init_file):
    from healthformer.training import OptimizationMixin
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=2)
    try:
        owner = OptimizationMixin()
        owner.fabric = Adapter(rank)
        owner.config = SimpleNamespace(log_wandb_metric_every_k=10, allow_wandb=False)
        owner.it, owner.epoch, owner.training_step = 1, 0, 1
        for tracking, update, window in ((False, 1, 2), (False, 10, 1),
                                         (True, 1, 2), (True, 2, 1)):
            owner.config.allow_wandb = tracking
            owner.it = update
            report_progress = update % 10 == 0
            components = {'Standard weighted contribution': (2. + 4 * rank) * window,
                          'ELIS weighted contribution': float(rank) * window}
            # Continuous loss components can be absent on different workers.
            components['ZscoreMAE_Tr_ElisAug' if rank else 'ZscoreMAE_Tr_NTP'] = (10. if rank else 4.) * window
            if rank:
                components = dict(reversed(list(components.items())))
            output = io.StringIO()
            with patch('healthformer.training.wandb.log') as log, contextlib.redirect_stdout(output), \
                    patch.object(dist, 'all_gather_object', wraps=dist.all_gather_object) as gather:
                owner._log_completed_training_window(components, 3, window, 2. + 5 * rank,
                                                      {'misc/Iteration': update, 'misc/Learning rate': .01})
            assert components == {}
            if not (tracking or report_progress):
                gather.assert_not_called()
                log.assert_not_called()
                assert output.getvalue() == ''
                continue
            gather.assert_called_once()
            if rank == 0:
                log.assert_called_once()
                logged = log.call_args.args[0]
                assert logged['Training Loss/'] == 4.5
                assert logged['Training Losses/Standard weighted contribution'] == 4.
                assert logged['Training Losses/ELIS weighted contribution'] == .5
                assert logged['Training Losses/ZscoreMAE_Tr_NTP'] == 2.
                assert logged['Training Losses/ZscoreMAE_Tr_ElisAug'] == 5.
                assert logged['Training Accumulation/Window Size'] == window
                assert logged['misc/Learning rate'] == .01
                assert ('training loss 4.500000' in output.getvalue()) == report_progress
            else:
                log.assert_not_called()
                assert output.getvalue() == ''
    finally:
        dist.destroy_process_group()
