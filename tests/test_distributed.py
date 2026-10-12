"""Two-rank CPU/Gloo regressions for the HealthFormer training implementation."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]


from tests._helpers import load_implementation


implementation = load_implementation()


def run_spawned_processes_bounded(target, args, *, timeout=45.0):
    """Run a two-rank spawn fixture without allowing a collective hang forever."""
    process_context = mp.start_processes(
        target,
        args=args,
        nprocs=2,
        join=False,
        start_method="spawn",
    )
    completed = False
    deadline = time.monotonic() + float(timeout)
    try:
        while time.monotonic() < deadline:
            if process_context.join(
                    timeout=min(0.25, max(0.0, deadline - time.monotonic())),
                    grace_period=1.0,
            ):
                completed = True
                return
        raise TimeoutError(
            f"Two-rank fixture {target.__name__} did not finish within {timeout:.1f} seconds."
        )
    finally:
        if not completed:
            for process in process_context.processes:
                if process.is_alive():
                    process.terminate()
            for process in process_context.processes:
                process.join(timeout=2.0)
            for process in process_context.processes:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2.0)


class IntegerDataset(Dataset):
    def __init__(self, size):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return int(index)


class BufferedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 1)
        self.register_buffer("constant", torch.ones(1))

    def forward(self, values):
        return self.projection(values) + self.constant


def run_uneven_eval(rank, world_size, init_file, dataset_size, result_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        dataset = IntegerDataset(dataset_size)
        sampler = implementation.ExactDistributedEvalSampler(dataset, world_size, rank)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler, drop_last=False)
        model = DistributedDataParallel(BufferedModel(), broadcast_buffers=False)

        local_indices = []
        with torch.no_grad():
            for batch in loader:
                local_indices.extend(batch.tolist())
                model(batch.to(torch.float32).view(-1, 1))

        gathered_indices = [None] * world_size
        dist.all_gather_object(gathered_indices, local_indices)
        local_totals = torch.tensor(
            [len(local_indices), sum(local_indices)], dtype=torch.long
        )
        dist.all_reduce(local_totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            Path(result_file).write_text(
                json.dumps(
                    {
                        "rank_indices": gathered_indices,
                        "count": int(local_totals[0]),
                        "sum": int(local_totals[1]),
                    }
                )
            )
    finally:
        dist.destroy_process_group()


class TinyBranchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(1.0))
        self.elis_only = torch.nn.Parameter(torch.tensor(1.0))
        self.standard_only = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, scale, stage):
        if stage == "ELIS":
            return scale * (self.shared + self.elis_only)
        if stage == "STANDARD":
            return scale * (2.0 * self.shared + self.standard_only)
        raise ValueError(stage)


class StrictCursor:
    def __init__(self, total_batches):
        self.total_batches = total_batches
        self.next_batch_index = 0
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def mark_consumed(self, expected_absolute_index):
        if int(expected_absolute_index) != self.next_batch_index + 1:
            raise RuntimeError("cursor did not advance by one")
        self.next_batch_index = int(expected_absolute_index)

    def finish_epoch(self):
        if self.next_batch_index != self.total_batches:
            raise RuntimeError("incomplete epoch")
        self.next_batch_index = 0


class RecordingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.gradient_windows = []

    def step(self, closure=None):
        self.gradient_windows.append([
            None if parameter.grad is None else float(parameter.grad.detach().item())
            for group in self.param_groups for parameter in group["params"]
        ])
        return super().step(closure)


class DDPCommunicationState:
    def __init__(self, world_size):
        self.world_size = int(world_size)
        self.count = 0


def counting_allreduce_hook(state, bucket):
    """Perform DDP's mean all-reduce while exposing actual bucket communication."""
    state.count += 1
    work = dist.all_reduce(bucket.buffer(), async_op=True)

    def divide_by_world_size(future):
        reduced = future.value()[0]
        reduced.div_(state.world_size)
        return reduced

    return work.get_future().then(divide_by_world_size)


class DDPFabricAdapter:
    def autocast(self):
        return contextlib.nullcontext()

    def __init__(self, rank, world_size):
        self.global_rank = rank
        self.world_size = world_size
        self.events = []
        self.pending_stage = None
        self.pending_step = None
        self.communication_state = None
        self.backward_records = []

    def all_reduce(self, tensor, reduce_op):
        operations = {
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
            "sum": dist.ReduceOp.SUM,
        }
        dist.all_reduce(tensor, op=operations[reduce_op])
        return tensor

    def broadcast(self, value, src):
        dist.broadcast(value, src=src)
        return value

    @contextlib.contextmanager
    def no_backward_sync(self, model, enabled):
        self.events.append(("enter_no_sync", bool(enabled)))
        context = model.no_sync() if enabled else contextlib.nullcontext()
        with context:
            yield
        self.events.append(("exit_no_sync", bool(enabled)))

    def backward(self, loss):
        self.events.append((f"{self.pending_stage}_backward", None))
        loss.backward()
        self.backward_records.append({
            "step": self.pending_step,
            "stage": self.pending_stage,
            "communication_count": (
                None if self.communication_state is None
                else self.communication_state.count
            ),
        })

    def clip_gradients(self, model, optimizer, max_norm, norm_type):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=norm_type)

    def barrier(self):
        dist.barrier()


def training_batches(count):
    return [
        (
            torch.tensor([[step + 1, step + 2]], dtype=torch.int64),
            torch.zeros(1, 2),
            torch.zeros(1, 2, dtype=torch.int64),
            torch.zeros(1, 2, 7, dtype=torch.int64),
        )
        for step in range(count)
    ]


def run_distributed_training(
        rank,
        world_size,
        init_file,
        result_file,
        batch_count,
        accumulation,
        ineligible_boundary,
        nonfinite_rank_one_step,
        elis_enabled,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        fabric = DDPFabricAdapter(rank, world_size)
        model = DistributedDataParallel(
            TinyBranchModel(),
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        communication_state = DDPCommunicationState(world_size)
        model.register_comm_hook(communication_state, counting_allreduce_hook)
        fabric.communication_state = communication_state
        trainer = implementation.Trainer.__new__(implementation.Trainer)
        trainer.fabric = fabric
        trainer.device = torch.device("cpu")
        trainer.model = model
        trainer.optimizer = RecordingSGD(model.parameters(), lr=0.1)
        trainer.config = SimpleNamespace(
            evaluate=False,
            precision='32-true',
            max_seq_length=64,
            reduce_context_in_train=False,

            accumulate_grad_batches=accumulation,
            Elis_augmentation_prob=1.0 if elis_enabled else 0.0,
            elis_loss_weight=0.01,
            not_elis_loss_weight=1.0,

            max_grad_norm=1e9,
            log_wandb_metric_every_k=1000,
            allow_wandb=True,

        )
        trainer.pending_microbatches = 0
        trainer.pending_window_loss = 0.0
        trainer.epoch = 0
        trainer.it = 0
        trainer.current_batch_index = 0
        trainer.mask_modality_val = 99
        trainer.vals_per_modality = {}
        trainer.max_lr = 0.1
        trainer.min_lr = 0.1
        trainer.warmup_steps = 0
        trainer.max_steps = max(batch_count, 1)

        cursor = StrictCursor(batch_count)
        batches = training_batches(batch_count)
        trainer._active_train_cursor = lambda: cursor
        trainer._active_train_dataloader = lambda: batches[cursor.next_batch_index:]

        def compute_elis_loss(self, *args, **kwargs):
            scale = torch.tensor(float((self.training_step + 1) * (rank + 1)))
            self.fabric.events.append(("ELIS_forward", None))
            self.fabric.pending_stage = "ELIS"
            self.fabric.pending_step = int(self.training_step)
            loss = self.model(scale, stage="ELIS")
            if ineligible_boundary and self.training_step == accumulation - 1:
                if rank == 1:
                    return None, False
            return loss, True

        def compute_standard_loss(self, *args, **kwargs):
            scale = torch.tensor(float((self.training_step + 1) * (rank + 1)))
            self.fabric.events.append(("STANDARD_forward", None))
            self.fabric.pending_stage = "STANDARD"
            self.fabric.pending_step = int(self.training_step)
            loss = self.model(scale, stage="STANDARD")
            if rank == 1 and self.training_step == nonfinite_rank_one_step:
                loss = loss * torch.tensor(float("nan"))
            return loss, {}

        trainer.compute_elis_loss = MethodType(compute_elis_loss, trainer)
        trainer.compute_standard_loss = MethodType(compute_standard_loss, trainer)
        implementation.wandb.log = lambda *args, **kwargs: None
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.train_epoch()

        local_result = {
            "parameters": [
                float(parameter.detach().item()) for parameter in model.parameters()
            ],
            "gradient_windows": trainer.optimizer.gradient_windows,
            "events": fabric.events,
            "backward_records": fabric.backward_records,
            "communication_count": communication_state.count,
            "boundary_elis_local_can_run": (
                None if not ineligible_boundary else rank == 0
            ),
            "steps": trainer.it,
            "pending": trainer.pending_microbatches,
        }
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_result)
        if rank == 0:
            Path(result_file).write_text(json.dumps(gathered))
    finally:
        dist.destroy_process_group()


class DistributedEvaluationTests(unittest.TestCase):
    def test_uneven_and_empty_rank_evaluation_complete_exactly_once(self):
        for dataset_size in (5, 1):
            with self.subTest(dataset_size=dataset_size), tempfile.TemporaryDirectory() as directory:
                init_file = str(Path(directory) / "process_group")
                result_file = str(Path(directory) / "result.json")
                run_spawned_processes_bounded(
                    run_uneven_eval,
                    args=(2, init_file, dataset_size, result_file),
                )
                result = json.loads(Path(result_file).read_text())
                flattened = [
                    index
                    for rank_indices in result["rank_indices"]
                    for index in rank_indices
                ]
                self.assertEqual(sorted(flattened), list(range(dataset_size)))
                self.assertEqual(len(flattened), len(set(flattened)))
                self.assertEqual(result["count"], dataset_size)
                self.assertEqual(result["sum"], sum(range(dataset_size)))


class DistributedTrainingTests(unittest.TestCase):
    def run_case(
            self,
            batch_count,
            accumulation,
            ineligible_boundary=False,
            nonfinite_rank_one_step=None,
            elis_enabled=True,
    ):
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "process_group")
            result_file = str(Path(directory) / "result.json")
            run_spawned_processes_bounded(
                run_distributed_training,
                args=(
                    2,
                    init_file,
                    result_file,
                    batch_count,
                    accumulation,
                    ineligible_boundary,
                    nonfinite_rank_one_step,
                    elis_enabled,
                ),
            )
            return json.loads(Path(result_file).read_text())

    def assert_backward_records(self, result, expected):
        records = result["backward_records"]
        self.assertEqual(len(records), len(expected))
        for record, (step, stage, communication_count) in zip(records, expected):
            self.assertEqual(record["step"], step)
            self.assertEqual(record["stage"], stage)
            self.assertEqual(record["communication_count"], communication_count)

    def assert_replicas_equal(self, results):
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["steps"], results[1]["steps"])
        self.assertEqual(results[0]["pending"], 0)
        self.assertEqual(results[1]["pending"], 0)
        self.assertEqual(len(results[0]["parameters"]), len(results[1]["parameters"]))
        for left, right in zip(results[0]["parameters"], results[1]["parameters"]):
            self.assertAlmostEqual(left, right, places=6)
        self.assertEqual(
            len(results[0]["gradient_windows"]),
            len(results[1]["gradient_windows"]),
        )
        for left_window, right_window in zip(
                results[0]["gradient_windows"], results[1]["gradient_windows"]):
            self.assertEqual(len(left_window), len(right_window))
            for left, right in zip(left_window, right_window):
                if left is None or right is None:
                    self.assertIsNone(left)
                    self.assertIsNone(right)
                else:
                    self.assertAlmostEqual(left, right, places=6)

    def test_a1_pairs_forwards_and_backwards_and_synchronizes_all_parameters(self):
        results = self.run_case(batch_count=1, accumulation=1)
        self.assert_replicas_equal(results)
        self.assertEqual(results[0]["steps"], 1)
        self.assertEqual(results[0]["events"][:8], [
            ["enter_no_sync", True],
            ["ELIS_forward", None],
            ["ELIS_backward", None],
            ["exit_no_sync", True],
            ["enter_no_sync", False],
            ["STANDARD_forward", None],
            ["STANDARD_backward", None],
            ["exit_no_sync", False],
        ])
        gradients = results[0]["gradient_windows"][0]
        self.assertAlmostEqual(gradients[0], 2.01 * 1.5, places=6)
        self.assertAlmostEqual(gradients[1], 0.01 * 1.5, places=6)
        self.assertAlmostEqual(gradients[2], 1.5, places=6)
        for result in results:
            self.assert_backward_records(result, [
                (0, "ELIS", 0),
                (0, "STANDARD", 1),
            ])
            self.assertEqual(result["communication_count"], 1)

    def test_accumulation_syncs_full_window_and_short_tail(self):
        results = self.run_case(batch_count=3, accumulation=2)
        self.assert_replicas_equal(results)
        self.assertEqual(results[0]["steps"], 2)
        first, tail = results[0]["gradient_windows"]
        self.assertAlmostEqual(first[0], 2.01 * 2.25, places=6)
        self.assertAlmostEqual(first[1], 0.01 * 2.25, places=6)
        self.assertAlmostEqual(first[2], 2.25, places=6)
        self.assertAlmostEqual(tail[0], 2.01 * 4.5, places=6)
        self.assertAlmostEqual(tail[1], 0.01 * 4.5, places=6)
        self.assertAlmostEqual(tail[2], 4.5, places=6)
        expected_records = [
            (0, "ELIS", 0),
            (0, "STANDARD", 0),
            (1, "ELIS", 0),
            (1, "STANDARD", 1),
            (2, "ELIS", 1),
            (2, "STANDARD", 2),
        ]
        for result in results:
            self.assert_backward_records(result, expected_records)
            self.assertEqual(result["communication_count"], 2)

    def test_earlier_elis_gradient_survives_post_forward_boundary_ineligibility(self):
        results = self.run_case(
            batch_count=2,
            accumulation=2,
            ineligible_boundary=True,
        )
        self.assert_replicas_equal(results)
        gradients = results[0]["gradient_windows"][0]
        self.assertAlmostEqual(gradients[0], 2.0 * 2.25 + 0.01 * 1.5 / 2.0, places=6)
        self.assertAlmostEqual(gradients[1], 0.01 * 1.5 / 2.0, places=6)
        self.assertAlmostEqual(gradients[2], 2.25, places=6)
        self.assertEqual(
            [result["boundary_elis_local_can_run"] for result in results],
            [True, False],
        )
        expected_records = [
            (0, "ELIS", 0),
            (0, "STANDARD", 0),
            (1, "STANDARD", 1),
        ]
        for result in results:
            self.assert_backward_records(result, expected_records)
            event_names = [event[0] for event in result["events"]]
            self.assertEqual(event_names.count("ELIS_forward"), 2)
            self.assertEqual(event_names.count("ELIS_backward"), 1)

    def test_elis_disabled_accumulation_still_syncs_only_at_boundaries(self):
        results = self.run_case(
            batch_count=3,
            accumulation=2,
            elis_enabled=False,
        )
        self.assert_replicas_equal(results)
        self.assertEqual(results[0]["steps"], 2)
        first, tail = results[0]["gradient_windows"]
        self.assertAlmostEqual(first[0], 2.0 * 2.25, places=6)
        self.assertIsNone(first[1])
        self.assertAlmostEqual(first[2], 2.25, places=6)
        self.assertAlmostEqual(tail[0], 2.0 * 4.5, places=6)
        self.assertIsNone(tail[1])
        self.assertAlmostEqual(tail[2], 4.5, places=6)
        expected_records = [
            (0, "STANDARD", 0),
            (1, "STANDARD", 1),
            (2, "STANDARD", 2),
        ]
        for result in results:
            self.assert_backward_records(result, expected_records)
            event_names = [event[0] for event in result["events"]]
            self.assertNotIn("ELIS_forward", event_names)
            self.assertNotIn("ELIS_backward", event_names)

    def test_one_rank_nonfinite_discards_the_partial_window_on_both_ranks(self):
        results = self.run_case(
            batch_count=4,
            accumulation=3,
            nonfinite_rank_one_step=1,
        )
        self.assert_replicas_equal(results)
        self.assertEqual(results[0]["steps"], 1)
        gradients = results[0]["gradient_windows"][0]
        self.assertAlmostEqual(gradients[0], 2.01 * 5.25, places=6)
        self.assertAlmostEqual(gradients[1], 0.01 * 5.25, places=6)
        self.assertAlmostEqual(gradients[2], 5.25, places=6)


if __name__ == "__main__":
    unittest.main()
