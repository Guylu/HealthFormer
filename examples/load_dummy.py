#!/usr/bin/env python3
"""
Walk through the synthetic ``HPP_tokenized_dummy.pt`` fixture.

Loads the file, prints the schema, slices a single participant out of the
batch, decodes their first few tokens into ``(token_id, modality_name,
temporal_features)`` triplets, and shows how to mask padding cleanly.

Run from the repo root::

    python3 examples/load_dummy.py
    python3 examples/load_dummy.py --path HPP_tokenized_dummy.pt --participant 0 --first-n 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _print_top_level(blob: dict) -> None:
    print("=== Top-level keys ===")
    for key in sorted(blob.keys()):
        val = blob[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key:30s} tensor  shape={tuple(val.shape)}  dtype={val.dtype}")
        elif isinstance(val, dict):
            print(f"  {key:30s} dict    n_entries={len(val)}")
        elif isinstance(val, list):
            print(f"  {key:30s} list    length={len(val)}")
        else:
            print(f"  {key:30s} {type(val).__name__}: {val}")
    print()


def _print_global_metadata(blob: dict) -> None:
    print("=== Global metadata ===")
    print(f"  vocab_size            : {blob['vocab_size']}")
    print(f"  number_of_modalities  : {blob['number_of_modalities']}")
    print(f"  temporal_vocab_size   : {blob['temporal_vocab_size']}")
    print(f"  time_dim_order        : {blob['time_dim_order']}")
    print(f"  participants in batch : {len(blob['index'])}")
    print()


def _print_participant_slice(
    blob: dict,
    participant: int,
    first_n: int,
) -> None:
    tokens = blob["tokens"][participant]
    modalities = blob["modalities"][participant]
    time_expanded = blob["time_expanded"][participant]

    n_mod = blob["number_of_modalities"]
    real_mask = modalities != n_mod
    real_count = int(real_mask.sum())

    pid = blob["index"][participant]
    print(f"=== Participant {participant} (id={pid}) ===")
    print(f"  total positions : {tokens.shape[0]}")
    print(f"  real positions  : {real_count}")
    print(f"  padding         : {tokens.shape[0] - real_count}")
    print()

    print(f"  First {first_n} real tokens:")
    num_to_name = blob["number_to_modality_name"]
    time_dims = blob["time_dim_order"]

    real_indices = torch.nonzero(real_mask, as_tuple=False).squeeze(-1)
    shown = 0
    for idx in real_indices.tolist():
        if shown >= first_n:
            break
        tok = int(tokens[idx])
        mod = int(modalities[idx])
        time_vec = time_expanded[idx].tolist()
        time_str = ", ".join(f"{name}={v}" for name, v in zip(time_dims, time_vec))
        print(f"    pos={idx:4d}  token={tok:4d}  modality={mod} ({num_to_name[mod]})")
        print(f"             time: {time_str}")
        shown += 1
    print()


def _print_modality_token_distribution(blob: dict) -> None:
    n_mod = blob["number_of_modalities"]
    modalities = blob["modalities"]
    valid = modalities[modalities != n_mod].flatten()
    counts = torch.bincount(valid.to(torch.int64), minlength=n_mod)
    num_to_name = blob["number_to_modality_name"]
    total = int(counts.sum())
    print("=== Modality token counts (across all participants) ===")
    print(f"  Total non-padding tokens: {total:,}")
    print(f"  {'modality':30s}  {'count':>10s}  {'pct':>6s}")
    pairs = sorted(
        ((num_to_name[mod_id], int(c)) for mod_id, c in enumerate(counts.tolist())),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for name, count in pairs:
        pct = 100.0 * count / max(total, 1)
        print(f"  {name:30s}  {count:>10,}  {pct:>5.2f}%")
    print()


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_path = repo_root / "HPP_tokenized_dummy.pt"

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--path", type=Path, default=default_path)
    parser.add_argument("--participant", type=int, default=0)
    parser.add_argument("--first-n", type=int, default=8)
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"ERROR: fixture not found: {args.path}", file=sys.stderr)
        return 2

    print(f"Loading {args.path}\n")
    blob = torch.load(args.path, map_location="cpu", weights_only=False)

    _print_top_level(blob)
    _print_global_metadata(blob)
    _print_participant_slice(blob, args.participant, args.first_n)
    _print_modality_token_distribution(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
