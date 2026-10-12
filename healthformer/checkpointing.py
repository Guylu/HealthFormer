"""Checkpoint state, compatibility checks, and resume."""
from pathlib import Path
import os
import tempfile
from .serialization import load_artifact, to_safe

from typing import Any
from typing import Dict
from typing import Optional
import numpy as np
import random
import torch
from .samplers import (
    ResumableBatchSampler,
)

class CheckpointMixin:
    def _active_train_dataloader(self):
        return self.train_dataloader

    def _active_train_cursor(self) -> ResumableBatchSampler:
        return self.train_batch_sampler

    def _active_train_cursor_signature(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        cursor = self._active_train_cursor()
        signature = {
            "version": self.TRAIN_CURSOR_CHECKPOINT_VERSION,
            "world_size": int(self.fabric.world_size),
            "active_loader_key": "oci",  # Preserve the historical checkpoint signature.
            "epoch": cursor.epoch if epoch is None else int(epoch),
        }
        # Version 2 drops the distributed remainder instead of duplicating rows.
        if getattr(self.config, 'train_sampler', None) == 'distributed' and self.fabric.world_size > 1:
            signature['distributed_partitioning_version'] = 2
        signature.update(cursor.structural_metadata())
        return signature

    def _active_train_cursor_checkpoint_state(self) -> Dict[str, Any]:
        cursor = self._active_train_cursor()
        if cursor.epoch != int(self.epoch):
            cursor.set_epoch(self.epoch)
        if cursor.next_batch_index != int(self.current_batch_index):
            raise RuntimeError(
                "Active batch cursor disagrees with current_batch_index: "
                f"{cursor.next_batch_index} != {self.current_batch_index}."
            )
        return {
            "version": self.TRAIN_CURSOR_CHECKPOINT_VERSION,
            "cursor": cursor.state_dict(),
            "signature": self._active_train_cursor_signature(),
        }

    def _validate_active_train_cursor_checkpoint_state(
            self,
            checkpoint: Dict[str, Any],
            expected_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate a saved training cursor without mutating the live cursor."""
        cursor = self._active_train_cursor()
        if expected_epoch is None:
            expected_epoch = int(self.epoch)
        else:
            expected_epoch = int(expected_epoch)
        if int(getattr(cursor, "next_batch_index", 0)) != 0:
            raise RuntimeError("Cannot restore a batch cursor over nonzero in-memory progress.")
        saved_batch_index = int(checkpoint.get("current_batch_index", 0))
        saved = checkpoint.get("train_cursor_state")

        if not isinstance(saved, dict):
            raise TypeError("Checkpoint train_cursor_state must be a dictionary.")
        saved_version = int(saved.get("version", -1))
        if saved_version != self.TRAIN_CURSOR_CHECKPOINT_VERSION:
            raise RuntimeError(
                f"Unsupported train cursor checkpoint version {saved_version!r}; "
                f"expected {self.TRAIN_CURSOR_CHECKPOINT_VERSION}."
            )

        saved_cursor = saved.get("cursor")
        saved_signature = saved.get("signature")
        if not isinstance(saved_cursor, dict) or not isinstance(saved_signature, dict):
            raise TypeError("Checkpoint train_cursor_state must contain cursor and signature dictionaries.")

        if int(saved_cursor.get("version", -1)) != cursor.STATE_VERSION:
            raise RuntimeError(
                f"Unsupported batch cursor version {saved_cursor.get('version')!r}; "
                f"expected {cursor.STATE_VERSION}."
            )
        restored_next_batch = int(saved_cursor.get("next_batch_index", -1))
        restored_epoch = int(saved_cursor.get("epoch", -1))
        if restored_next_batch != saved_batch_index:
            raise RuntimeError(
                "Checkpoint batch cursor disagrees with current_batch_index: "
                f"{restored_next_batch} != {saved_batch_index}."
            )
        if restored_epoch != expected_epoch:
            raise RuntimeError(
                f"Checkpoint batch cursor epoch {restored_epoch} does not match trainer epoch {expected_epoch}."
            )
        if not 0 <= restored_next_batch <= int(cursor.total_batches):
            raise RuntimeError(
                f"Restored batch cursor {restored_next_batch} is outside "
                f"0..{cursor.total_batches}."
            )
        expected_signature = self._active_train_cursor_signature(epoch=expected_epoch)
        differing_fields = [
            key for key in sorted(set(saved_signature) | set(expected_signature))
            if saved_signature.get(key) != expected_signature.get(key)
        ]
        if differing_fields:
            details = "; ".join(
                f"{key}: saved={saved_signature.get(key)!r}, current={expected_signature.get(key)!r}"
                for key in differing_fields
            )
            raise RuntimeError(f"Training cursor is structurally incompatible with this run ({details}).")

        return {
            "epoch": restored_epoch,
            "next_batch_index": restored_next_batch,
            "cursor": saved_cursor,
        }

    def _restore_active_train_cursor(
            self,
            checkpoint: Dict[str, Any],
            validated_state: Optional[Dict[str, Any]] = None,
    ):
        cursor = self._active_train_cursor()
        if validated_state is None:
            validated_state = self._validate_active_train_cursor_checkpoint_state(checkpoint)

        cursor.load_state_dict(validated_state["cursor"])
        self.current_batch_index = cursor.next_batch_index

    def _mark_active_train_batch_consumed(self, expected_absolute_index: int):
        cursor = self._active_train_cursor()
        cursor.mark_consumed(expected_absolute_index)
        self.current_batch_index = cursor.next_batch_index

    def _accumulation_boundary_is_empty(self) -> bool:
        """Return whether a checkpoint can be saved without losing gradients."""
        if int(getattr(self, "pending_microbatches", 0)) != 0:
            return False
        model = getattr(self, "model", None)
        return model is None or all(parameter.grad is None for parameter in model.parameters())

    def _require_empty_accumulation_boundary(self):
        """Reject checkpoints that would silently omit accumulated gradients."""
        pending = int(getattr(self, "pending_microbatches", 0))
        model = getattr(self, "model", None)
        gradients_present = 0 if model is None else sum(
            parameter.grad is not None for parameter in model.parameters()
        )
        if pending != 0 or gradients_present != 0:
            raise RuntimeError(
                "Cannot save a checkpoint with a non-empty gradient accumulation window: "
                f"pending_microbatches={pending}, gradients_present={gradients_present}."
            )

    def _accumulation_checkpoint_state(self) -> Dict[str, int]:
        self._require_empty_accumulation_boundary()
        return {
            "version": self.ACCUMULATION_CHECKPOINT_VERSION,
            "accumulate_grad_batches": int(self.config.accumulate_grad_batches),
            "pending_microbatches": 0,
        }

    def _validate_accumulation_resume_state(self, checkpoint):
        """Validate that resume starts at an accumulation boundary."""
        current_accumulation = int(self.config.accumulate_grad_batches)
        accumulation_state = checkpoint.get("accumulation_state")

        if accumulation_state is not None:
            if not isinstance(accumulation_state, dict):
                raise TypeError("Checkpoint accumulation_state must be a dictionary.")
            version = int(accumulation_state.get("version", -1))
            if version != self.ACCUMULATION_CHECKPOINT_VERSION:
                raise RuntimeError(
                    f"Unsupported accumulation checkpoint version {version}; "
                    f"expected {self.ACCUMULATION_CHECKPOINT_VERSION}."
                )
            saved_accumulation = self._validate_accumulate_grad_batches(
                accumulation_state.get("accumulate_grad_batches")
            )
            if saved_accumulation != current_accumulation:
                raise RuntimeError(
                    "Checkpoint accumulate_grad_batches does not match this run: "
                    f"saved={saved_accumulation}, current={current_accumulation}."
                )
            if int(accumulation_state.get("pending_microbatches", -1)) != 0:
                raise RuntimeError("Checkpoint contains a non-empty gradient accumulation window.")
            return

        raise RuntimeError('Exact resume requires accumulation_state; use initialize_from for older weights.')

    def _preflight_resume_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Validate required resume state before changing live training objects."""
        if not isinstance(checkpoint, dict):
            raise TypeError("Training checkpoint must be a dictionary.")

        if checkpoint.get('format_version') != 4:
            raise ValueError('Exact resume requires checkpoint format 4; use initialize_from for older weights.')
        required_metadata = ('config', 'train_cursor_state', 'accumulation_state',
                             'current_batch_index', 'experiment_identity', 'split_manifest')
        missing = [key for key in required_metadata if key not in checkpoint]
        if missing:
            raise ValueError('Exact resume requires complete checkpoint metadata: ' + ', '.join(missing))
        from .identity import validate_identity
        validate_identity(checkpoint.get('experiment_identity'), self.experiment_identity)

        saved_config = checkpoint['config']
        current_policy = getattr(self.config, 'split_policy', 'historical_modulo_inclusive')
        saved_policy = saved_config.get('split_policy', 'historical_modulo_inclusive')
        if saved_policy != current_policy:
            raise RuntimeError('Checkpoint split policy does not match this run.')
        if checkpoint.get('split_manifest') is not None and checkpoint['split_manifest'] != getattr(self, 'split_manifest', None):
            raise RuntimeError('Checkpoint participant split does not match this run.')
        self._validate_accumulation_resume_state(checkpoint)
        for key in ('epochs', 'lr', 'gamma', 'warmup_steps', 'precision', 'seed'):
            if key in saved_config and hasattr(self.config, key) and saved_config[key] != getattr(self.config, key):
                raise RuntimeError(f'Checkpoint training setting differs: {key}.')
        required_fields = (
            "model_state_dict",
            "optimizer_state_dict",
            "numpy_rng_state",
            "torch_rng_state",
            "python_rng_state",
            "epoch",
            "it",
            "best_val_loss",
        )
        missing_fields = [field for field in required_fields if field not in checkpoint]
        if missing_fields:
            raise KeyError(
                "Checkpoint is missing required resume fields: " + ", ".join(missing_fields)
            )

        rank_states = checkpoint.get('rank_rng_states')
        if self.fabric.world_size > 1 and rank_states is None:
            raise RuntimeError('Distributed resume requires per-rank RNG states.')
        if rank_states is not None:
            if len(rank_states) != self.fabric.world_size:
                raise RuntimeError('Checkpoint rank RNG states do not match world size.')
            for state in rank_states:
                np.random.RandomState().set_state(state['numpy_rng_state'])
                random.Random().setstate(state['python_rng_state'])
                torch.Generator().set_state(state['torch_rng_state'])
        # Format 4 stores one CPU byte tensor for each worker's active CUDA device.
        for state in [checkpoint] + (rank_states or []):
            if 'rng_state_cuda' not in state:
                raise ValueError('Checkpoint is missing active-device CUDA RNG metadata.')
            cuda_state = state['rng_state_cuda']
            if self.device.type == 'cuda':
                if not isinstance(cuda_state, torch.Tensor) or cuda_state.device.type != 'cpu' or cuda_state.dtype != torch.uint8 or cuda_state.ndim != 1:
                    raise ValueError('Checkpoint CUDA RNG state must be a CPU byte vector.')
            elif cuda_state is not None:
                raise ValueError('CPU resume requires a null CUDA RNG state.')
        if self.device.type == 'cuda':
            local_state = rank_states[self.fabric.global_rank] if rank_states else checkpoint
            torch.Generator(device=self.device).set_state(local_state['rng_state_cuda'])
        checkpoint_epoch = int(checkpoint["epoch"])
        int(checkpoint["it"])
        int(checkpoint.get("current_batch_index", 0))
        cursor_state = self._validate_active_train_cursor_checkpoint_state(
            checkpoint, expected_epoch=checkpoint_epoch)
        current = self.model.state_dict()
        saved = checkpoint['model_state_dict']
        if current.keys() != saved.keys() or any(current[k].shape != saved[k].shape for k in current):
            raise RuntimeError('Checkpoint model keys or shapes do not match this model.')
        groups = checkpoint['optimizer_state_dict'].get('param_groups', [])
        if len(groups) != len(self.optimizer.param_groups) or any(
                len(a['params']) != len(b['params']) for a, b in zip(groups, self.optimizer.param_groups)):
            raise RuntimeError('Checkpoint optimizer parameter groups do not match.')
        # Validate RNG encodings on disposable generators before touching live state.
        np.random.RandomState().set_state(checkpoint['numpy_rng_state'])
        random.Random().setstate(checkpoint['python_rng_state'])
        torch.Generator().set_state(checkpoint['torch_rng_state'])
        return cursor_state

    def load_model(self, filename, checkpoint=None):
        # Full checkpoints contain optimizer and RNG objects and must come from a trusted source.
        if checkpoint is None:
            checkpoint = load_artifact(filename)
        validated_cursor_state = self._preflight_resume_checkpoint(checkpoint)
        self._restore_checkpoint(filename, checkpoint, validated_cursor_state)

    def _restore_checkpoint(self, filename, checkpoint, validated_cursor_state):
        """Apply a checkpoint whose resume state has already passed preflight."""
        model_params = checkpoint['model_state_dict']
        rank_states = checkpoint.get('rank_rng_states')
        if rank_states is not None:
            checkpoint = {**checkpoint, **rank_states[self.fabric.global_rank]}
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        torch.random.set_rng_state(checkpoint['torch_rng_state'])
        random.setstate(checkpoint["python_rng_state"])
        self.epoch = checkpoint['epoch']

        self.model.load_state_dict(model_params)

        self.it = checkpoint['it']

        self.current_batch_index = checkpoint.get('current_batch_index', 0)
        # Restore the rank-local sampler suffix from the shared logical cursor.
        self._restore_active_train_cursor(checkpoint, validated_cursor_state)
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_pending_window_state()

        if self.device.type == 'cuda':
            torch.cuda.set_rng_state(checkpoint['rng_state_cuda'], device=self.device)

        self.best_val_loss = checkpoint['best_val_loss']


        if self.fabric.global_rank == 0:
            print(f"Resumed {filename}: completed_epochs={self.epoch}, consumed_batches={self.current_batch_index}")

    def initialize_weights(self, filename, checkpoint=None):
        """Load compatible weights only; keep this run's fresh optimizer, RNG and cursor."""
        error = None
        state = None
        model = self.model
        try:
            if checkpoint is None:
                checkpoint = load_artifact(filename)
            state = checkpoint['model_state_dict']
            expected = model.state_dict()
            if set(state) != set(expected):
                raise ValueError('Initialization parameter names differ.')
            if any(state[k].shape != expected[k].shape for k in expected):
                raise ValueError('Initialization parameter shapes differ.')
        except Exception as exc:
            error = str(exc)
        failed = torch.tensor(int(error is not None), device=self.fabric.device)
        self.fabric.all_reduce(failed, reduce_op='max')
        if failed.item():
            raise RuntimeError('Weight initialization failed: ' + (error or 'see failing rank'))
        model.load_state_dict(state, strict=True)

    def maybe_resume(self):
        initial = getattr(self.config, 'initialize_from', None)
        filename = getattr(self.config, 'resume_from', None)
        if initial and filename:
            raise ValueError('Choose resume_from or initialize_from, not both.')
        if initial:
            self.initialize_weights(initial)
            return
        if filename is None:
            return
        error = None
        checkpoint = None
        try:
            checkpoint = load_artifact(filename)
            validated_cursor_state = self._preflight_resume_checkpoint(checkpoint)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        failed = torch.tensor(int(error is not None), device=self.device)
        self.fabric.all_reduce(failed, reduce_op='max')
        if failed.item():
            raise RuntimeError('Requested resume failed on one or more ranks: ' + (error or 'see failing rank'))
        # Every rank has loaded and validated its local state before any mutation.
        self._restore_checkpoint(filename, checkpoint, validated_cursor_state)

    def save_model(self, id='last'):
        accumulation_state = self._accumulation_checkpoint_state()
        base_dir = Path(getattr(self.config, 'checkpoint_dir', 'checkpoints'))
        if Path(str(id)).name != str(id):
            raise ValueError('Checkpoint id must be a filename component.')
        fname = str(base_dir / f'model_{id}.pt')
        state_dict = {
            'format_version': 4,
            'experiment_identity': self.experiment_identity,
            'preprocessing_provenance': getattr(self, 'preprocessing_provenance', {'status': 'unknown'}),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'it': self.it,
            'current_batch_index': self.current_batch_index,
            'train_cursor_state': self._active_train_cursor_checkpoint_state(),
            'accumulation_state': accumulation_state,

            'rng_state_cuda': (torch.cuda.get_rng_state(self.device)
                               if self.device.type == 'cuda' else None),
            "python_rng_state": random.getstate(),

            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.random.get_rng_state(),

            'epoch': self.epoch,

            'config': dict(self.config),
            'split_manifest': getattr(self, 'split_manifest', None),
            'best_val_loss': self.best_val_loss,

        }

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            local_rng = {key: state_dict[key] for key in (
                'numpy_rng_state', 'python_rng_state', 'torch_rng_state', 'rng_state_cuda')}
            rank_states = [None] * self.fabric.world_size
            torch.distributed.all_gather_object(rank_states, local_rng)
            state_dict['rank_rng_states'] = rank_states
        error = None
        temporary = None
        if self.fabric.global_rank == 0:
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
                # Keep the temporary file on the destination filesystem. Close it
                # before replacement so a failed write cannot replace a good save.
                with tempfile.NamedTemporaryFile(mode='wb', dir=base_dir,
                                                 prefix=f'.model_{id}.', suffix='.tmp',
                                                 delete=False) as stream:
                    temporary = Path(stream.name)
                    torch.save(to_safe(state_dict), stream)
                os.replace(temporary, fname)
            except BaseException as exc:
                error = exc
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError as exc:
                        if error is None:
                            error = exc
        failed = torch.tensor(int(error is not None), device=self.device)
        failed = self.fabric.all_reduce(failed, reduce_op='max')
        if failed.item():
            if error is not None:
                raise error
            raise RuntimeError('Checkpoint save failed on rank zero; see its error.')
        if self.fabric.global_rank == 0:
            print(f"Saved {fname}: completed_epochs={self.epoch}")


        return fname
