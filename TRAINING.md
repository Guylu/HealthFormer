# Training HealthFormer

## Inputs

Set `data_path` in a JSON configuration. Run from the
repository root. The training artifact path must be supplied explicitly.

For `N` participants, sequence length `L`, and `M` modalities, the dataset is a
serialized mapping containing:

| Field | Structure |
| --- | --- |
| `tokens`, `continuous_tokens`, `modalities` | Aligned tensors `[N, L]`; modality `M` denotes trailing padding; no real event may follow it. |
| `time_expanded` | Integer-valued tensor `[N, L, 7]`. |
| `index` | Length-`N` integer participant IDs. |
| `cum_base_value_per_modality` | Vocabulary offsets by modality. |
| `modality_name_to_number`, `number_to_modality_name` | Forward and inverse modality mappings. |
| `vals_per_modality`, `vocab_size` | Per-modality values and shared vocabulary size. |
| `temporal_vocab_size`, `time_dim_order` | Seven temporal vocabulary sizes and their ordered names. |
| `number_of_modalities`, `cat_cols`, `bins` | Modality count, categorical names, and continuous bin edges. |

Time order is `day_of_week, hour, minute, month, year, day_of_month, sleep`.
Temporal encoding follows the artifact's vocabulary. Initialization expects at
least one `_DEXALoader` modality for visit identification. The synthetic training
generator includes a compatible marker.

Artifact contents are validated when training starts. `--check` only checks
configuration, dependencies, accelerator availability, and paths.

Demographic encodings recognize `gender`, `gender_at_birth`, and `birth_event`.
The two gender values use the modality vocabulary offset plus 1 and plus 2.
A visible `gender` event takes precedence over `gender_at_birth`; visibility follows
the causal or ELIS attention mask. Age uses a visible `birth_event`.
Time-series-removal augmentation selects exact names `GlucoseValue` and
`sleep_event`, and suffixes `_apple_google`, `_species`, `_gene`, and `_questionires`.
These names and spellings are part of the current artifact convention.

## Configuration

```bash
python train.py --show-config
python train.py --config examples/training_config.json --show-config
python train.py --config examples/training_config.json --check
python train.py --config examples/training_config.json --run
```

JSON overrides merge with `healthformer/config.py`; unknown keys are rejected.
The example selects `modulo`; the default is `historical_modulo_inclusive`.
Both split policies are defined below.

Earlier configuration spellings remain accepted:

| Earlier name | Current name |
| --- | --- |
| `yoads_temporal_enconding` | `sinusoidal_time_embeddings` |
| `use_yoads_encoding_for_tokens` | `continuous_value_embeddings` |
| `elis_min_len` | `elis_max_input_length` |
| `use_sparse_attn_gate` | `use_attention_value_gate` |
| `sparse_gate_init_bias` | `attention_value_gate_init_bias` |

Conflicting values under both spellings are rejected. Resume compares the resolved
names, so these renames do not invalidate compatible format-4 checkpoints. Stored
parameter names remain unchanged. Progress messages use one-based epochs and batches;
`stop_early_now` remains a zero-based epoch-index cutoff.

`n_embd` must be divisible by `n_heads`. The token-embedding width is `n_embd - 12`
when `temporal_positional_encodings` is enabled, and `n_embd` otherwise;
it must be even and at least 4.
With `sinusoidal_time_embeddings` enabled, `n_embd` must also be divisible by 16.

| Setting | Behavior |
| --- | --- |
| `accelerator`, `devices`, `precision` | CPU with `32-true`, or CUDA with `32-true` or `bf16-mixed`. FP16 is unsupported. Fabric manages precision throughout. |
| `epochs`, `batch_per_gpu`, `accumulate_grad_batches`, `lr` | Training duration, per-device batch size, accumulation, and learning rate. |
| `log_wandb_metric_every_k` | Console progress interval in completed optimizer updates; tracking records every update. |
| `max_validation_batches` | Standard-validation debug limit per rank; `null` means unrestricted. Does not limit longitudinal validation. |
| `evaluate` | Standard validation during training. |
| `train_sampler` | `eli_friendly` (default) or `distributed`. |
| `longitudinal_evaluation` | Opt-in longitudinal validation. |
| `save`, `checkpoint_dir` | Keep one rolling checkpoint after completed epochs in the selected local directory. |
| `resume_from`, `initialize_from` | Exact continuation, or weights-only initialization of a new run; mutually exclusive. |
| `stop_early_now` | Optional epoch-index cutoff; disabled by default. |

BF16 training preserves raw continuous inputs through normalization and encoding,
and computes the continuous MAE expectation in FP32, retaining higher-precision
targets when supplied. The backbone still uses mixed precision.
The trainer sets float32 matmul precision to `high`,
permitting TF32 on supported CUDA GPUs even with `32-true`. TF32 changes internal
matmul arithmetic, not tensor dtypes.

Training statistics are recomputed from unaugmented training participants.
Values are grouped by modality IDs, filtered to finite values, and collected only
from training participants. Mean/std require at least 50 observations; min/max
use all available finite values.
Both training samplers drop the distributed remainder and incomplete batches
without repeating participants within an epoch. Validation uses exact,
non-dropping sampling. Distributed runs require `find_unused_parameters=true`,
including when ELIS is disabled, because optional branches can leave parameters unused.

`max_seq_length` (default 25,000) bounds real training records and constructed
ELIS prompts, including prompter tokens and the final target. That final target is
removed before model input, so the budget can leave one input position unused.
Prepare oversized training records before
training. ELIS target lists are shortened to fit, and validation reports coverage.
The model rejects oversized inputs. The optimizer is AdamW; `seed` controls model
randomness, the training sampler, and loader generators.

## Training and validation split

`percent_validation` is the modulo threshold, with two explicit policies:

- `modulo`: validation IDs satisfy `id % 100 < percent_validation`.
- `historical_modulo_inclusive`: validation IDs satisfy `id % 100 <= percent_validation`.

At 20, these select 20 and 21 residue buckets respectively; neither guarantees
an exact participant percentage. Actual participant counts are reported. The
checkpoint stores policy, threshold, and participant membership. Resume rejects
a changed policy or membership. This trainer does not construct a final test set.

Changing the split changes the experiment and requires recomputing baselines.

## Checkpoints and resume

Checkpoints preserve model, optimizer, random states, configuration, participant
split, and training cursor. Saves require an empty gradient-accumulation boundary.
A completed epoch resumes at the next epoch. With `save=true`, intermediate epochs
overwrite `model_last.pt`; the final save follows the requested final validations.
Restart uses the last successfully saved checkpoint. Rank zero writes and closes
a unique temporary file inside the checkpoint directory, then atomically replaces
the rolling checkpoint. Write or replacement failures stop all workers without
reporting success; an existing checkpoint survives those failures.
Intermediate checkpoints record training progress rather than best-model selection.

To continue, set `resume_from` to the checkpoint and keep the intended total
`epochs` consistent with the original learning-rate schedule. Requested resume
failures stop the run. Checkpoints compare experiment-critical settings,
participant membership, and SHA-256 hashes of the training artifact.
Input contents must match even if paths change. Logging settings and output paths
may change. Evaluation flags, validation limits, and validation sequence settings
must match, keeping the saved metric history comparable.
The check reports differing fields before loading live state.

Use `initialize_from` to load compatible weights into a fresh experiment without
restoring optimizer, schedule counters, RNG, or data cursor. Use the matching
vocabulary and preprocessing artifacts: matching parameter shapes alone do not
establish compatible token meanings.
Normalization and MAE statistics are recomputed from the new run's training
participants. A changed split or cohort can therefore change input encodings and
the auxiliary objective despite identical loaded weights.

Exact continuation requires checkpoint format 4, experiment-identity schema 9,
complete cursor and accumulation metadata, and matching settings and artifacts.
Format 4 stores only each worker's active CUDA device RNG state (null on CPU),
so unused visible GPUs do not determine checkpoint compatibility.
Multi-worker runs with `train_sampler="distributed"` also require partitioning
version 2, which drops the distributed remainder without repeating participants.
Older checkpoints that fail these compatibility checks require weights-only
initialization with `initialize_from` and a compatible model and vocabulary.

Artifacts must use tensors and primitive containers supported by restricted loading.
There is no unrestricted pickle-loading option. Convert incompatible legacy
artifacts separately in an environment where their source is trusted.

## Tracking and data handling

Online tracking is off by default. Install its optional dependency with
`python -m pip install wandb==0.26.1`. Enabling it requires `allow_wandb=true`,
`wandb_entity`, and `wandb_project`. Application logging sends aggregate scalar
metrics. Automatic console capture, code/Git capture, installed-package listings,
system metrics, and machine metadata collection are disabled. W&B still manages
run/account metadata, SDK version/feature metadata, and diagnostic logs.
Participant plots and tables are not exported. Checkpoints are saved locally and
contain participant IDs; treat them as sensitive artifacts when training on real data.

## Objective and validation definitions

`loss_vocabulary="global"` preserves the historical normalization over the shared
vocabulary. `"modality"` normalizes over the exact token IDs declared for the target
modality; this is a different objective. Standard and longitudinal
validation CE use the selected vocabulary. Top-k accuracy still uses global logits.
Validation CE measures hard-target likelihood; training retains the existing
Gaussian soft-label construction, `soft_labels=true`, `SL_sigma=0.01`, and scale.
At this default sigma, neighboring integer-bin weights underflow in float32,
so the default does not provide meaningful smoothing. These settings are preserved.

Non-finite continuous values contribute no continuous input embedding or MAE;
their discrete tokens still contribute to CE, including in ELIS. Longitudinal continuous
validation metrics use finite observed targets and report their own counts.
Bin edges must be finite and increasing, with one interval per declared bin token;
invalid mappings and token/modality pairs fail during initialization. Shared token
IDs are allowed when explicitly declared by each modality. Noise augmentation changes
only selected finite observations, using the sorted declared token IDs as bin order;
unselected and missing values remain unchanged. Random token removal draws only
from real non-marker events; a zero rate removes nothing. Block removal samples
starts and sizes over the real record, protects visit markers, and restores the
original padded width. The configured block count has integer jitter from -3 to
+2, clamped to at least one and at most the real record length. The default of two
therefore proposes one to four blocks, which may overlap or include protected
markers. A zero block count removes nothing. If the combined
augmentations leave fewer than two real tokens, the original aligned sample is
restored. Batches without prediction targets cannot advance the optimizer.

Training averages losses over accepted microbatches in each optimizer update,
including a short final window. Continuous MAE is averaged per eligible modality
within each microbatch. Standard and ELIS objective weights remain explicit.
`not_elis_loss_weight` must be positive; this wrapper supports standard pretraining
with optional ELIS, not ELIS-only training. ELIS may be disabled with a zero weight
or zero attempt probability.
Training logs objective scalars. Standard validation reports overall and per-modality
CE, counts, and coverage. ELIS longitudinal validation reports prediction metrics.
Distributed training losses are averaged equally across workers, matching gradient
averaging, and reported once on rank zero. A loss component absent on a worker
contributes zero to this average.

Validation sequence policies are explicit:

- `prefix`: score the first `validation_max_tokens` positions, limited by model
  capacity. The default remains 10,000 positions.
- `full`: score each complete record; reject records beyond model capacity.
- `sliding`: use windows of at most `validation_max_tokens` positions, with roughly
  half a window of preceding context at subsequent boundaries. Score each target
  once. Context resets between windows, so scores are not equivalent to unlimited
  full-record attention.

Validation reports eligible/scored targets, coverage fraction, and truncated
participant counts for the selected validation batches. A debug batch limit does
not establish coverage of the entire cohort. Any non-finite validation CE invalidates
that validation and stops the run after workers synchronize; it cannot update a
best score.

When enabled, validation runs before each epoch and once after training ends,
before the final checkpoint is saved. Longitudinal validation reports metrics
without returning participant IDs or event identifiers.

## Longitudinal validation

The current artifact schema has no explicit visit IDs. The fixed `distinct_year` protocol
therefore groups events by encoded year. Repeated DEXA markers in one year do not
create extra groups. Unordered years and fewer than two marker years are excluded.
Their artifact-wide counts are reported once during setup; validation coverage is
reported separately. Two clinical visits in one year cannot be distinguished by
this protocol. Marker selection uses selected training participants. Modality-subset augmentation
samples uniformly from eligible modalities. The ELIS-friendly sampler uses the same
distinct-year eligibility rule as the training and evaluation paths. Context includes
all retained events through the end of the first marker year, including earlier
history. Targets come from eligible modalities in the second marker year. Events
strictly between those marker years are excluded from both context and targets.

ELIS training also has a batch-length gate: `elis_max_input_length` is
a maximum input width (default 8,000), separate from `max_seq_length` (25,000).
With context reduction enabled, the longest augmented real record across workers
determines that width, excluding the final target position. One long record can
therefore suppress ELIS for the entire global microbatch. Without context reduction,
storage padding counts toward the gate. Longitudinal validation does not use this gate.

Optional artifact metadata `preprocessing_provenance` should record the participants
or split used to fit bin edges and identify the preprocessing procedure. Without
it, checkpoint metadata records fitting provenance as unknown. Training-only
runtime statistics and artifact hashes do not prove that upstream preprocessing
was fitted without validation data.

Longitudinal predictions are conditioned on supplied target modality and time.
They do not forecast which measurements will be ordered or when a visit will occur.
The retained target selection matches names containing `_BloodTestsLoader`,
`_ItamarSleepLoader`, `_BodyMeasuresLoader`, `_ECGTextLoader`, `_UltrasoundLoader`,
`_DEXALoader`, `_NightingaleLoader`, `_ABILoader`, `_RetinaScanLoader`, `_MicroStats`,
`_questionires`, `_locations`, `medication`, or `medical_condition_code`.
ELIS training uses this same selection except for `_NightingaleLoader` and
`_locations`, preserving the original training/evaluation distinction. The selections
are defined together in `healthformer/visits.py`; neither covers all modalities.
Validation reports eligible targets before context truncation, scored targets, their
fraction, and participants whose eligible targets were truncated. Eligibility here
means selected modalities in the second marker year, not every future event.
Only selected targets spend the future-prompt budget. Explicitly requested ELIS
validation fails if no targets can be scored, including when context truncation
excludes all eligible targets.

Longitudinal correlations are descriptive and pooled across events. Ordinary
correlation p-values are omitted because events within participants are dependent;
constant series have no reported correlation. Inference would require a separate
participant-clustered analysis, such as participant-level bootstrap confidence
intervals. More frequently observed participants contribute more events.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the input representations, decoder,
and shared ELIS attention rules. Configuration defaults are in
[`healthformer/config.py`](healthformer/config.py).

## Research scope

This release contains research code and a synthetic training example. Real HPP
records, preprocessing, and trained weights are not distributed. See the
[paper](https://arxiv.org/abs/2604.27899) for the experimental settings and reported results.
The synthetic CPU checks train the real model, save, resume, and compare final
weights with uninterrupted training. Other checks exercise ELIS, longitudinal
validation, losses, temporal handling, and distributed accumulation. CUDA numerical
parity and multi-node execution remain unverified.

See the [model card](MODEL_CARD.md#intended-use) for intended use and interpretation.
