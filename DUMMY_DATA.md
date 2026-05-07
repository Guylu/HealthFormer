# Dummy data: `HPP_tokenized_dummy.pt`

This repository ships with a small, fully synthetic fixture
(`HPP_tokenized_dummy.pt`, about 700 KB) so contributors can run
the modeling code, write loaders, and verify pipeline plumbing
without access to any real cohort data.

## Provenance


It contains:

- **No real participant IDs.** `index` is `[10001, 10002, ..., 10020]`, fabricated by the script.
- **No real token values.** All tokens come from the seeded RNG.
- **No real bin edges.** Per-modality bin counts are random integers in `[5, 100]`.
- **No real cohort distribution statistics.**
- **No real modality names.** The 20 strings (`cgm`, `lab_hba1c`, `body_bmi`, ...) are generic medical-informatics concepts hand-written in the generator, not LabData loader names.
- **No real column names.** `cat_cols` and `nutritional_cols` are short generic placeholders (`cat_col_0..4`, `nutr_col_0..4`).

## Schema

The file is a `dict` with 15 keys, matching the canonical 15-key output
of the HPP tokenization pipeline:

| Key | Type | Shape / Length | Notes |
|---|---|---|---|
| `tokens` | `torch.int16` | `(20, 2000)` | pad value `0`, real values in `[1, vocab_size)` |
| `modalities` | `torch.int16` | `(20, 2000)` | pad value = `number_of_modalities` |
| `time_expanded` | `torch.int16` | `(20, 2000, 7)` | seven temporal dims |
| `vocab_size` | `int` | scalar | 1123 in the default fixture |
| `temporal_vocab_size` | `list[int]` | length 7 | `[8, 25, 5, 13, 6, 32, 4]` |
| `time_dim_order` | `list[str]` | length 7 | `['day_of_week', 'hour', 'minute', 'month', 'year', 'day_of_month', 'sleep']` |
| `number_of_modalities` | `int` | scalar | 20 |
| `modality_name_to_number` | `dict[str, int]` | 20 entries | name to id |
| `number_to_modality_name` | `dict[int, str]` | 20 entries | inverse |
| `vals_per_modality` | `dict[str, set]` | 20 entries | observed token values per modality (keyed by name) |
| `base_value_per_modality` | `dict[int, int]` | 20 entries | per-modality bin count (keyed by id) |
| `cum_base_value_per_modality` | `dict[int, int]` | 20 entries | cumulative offset (keyed by id) |
| `cat_cols` | `list[str]` | length 5 | placeholder categorical column names |
| `nutritional_cols` | `list[str]` | length 5 | placeholder nutritional column names |
| `index` | `list[int]` | length 20 | synthetic participant IDs `[10001..10020]` |

## Loading

```python
import torch

blob = torch.load("HPP_tokenized_dummy.pt", map_location="cpu", weights_only=False)
tokens = blob["tokens"]            # (20, 2000) int16
modalities = blob["modalities"]    # (20, 2000) int16
time_expanded = blob["time_expanded"]  # (20, 2000, 7) int16
```

## Token space

Each modality occupies a contiguous slice of the unified vocabulary:

```
modality 0 (cgm)        -> tokens in [0, base_0]
modality 1 (diet_log)   -> tokens in [cum_base_1, cum_base_1 + base_1]
...
```

Padding is zero-valued in `tokens` and `time_expanded`, and equal to
`number_of_modalities` in `modalities`. Use `modalities != number_of_modalities`
as the "real token" mask.


## Use it for what

This file is intended for:

- Running `train.py` end-to-end on a laptop in seconds
- Unit-testing data loaders against the real schema
- Sketching model architectures that consume the unified token / modality / time tensors

It is not intended to be a meaningful training set. The vocabulary, the
sequence lengths, and the cross-modality structure are all random.
Reported model results from this fixture are not informative.
