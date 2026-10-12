# Architecture

This overview describes the research implementation in this repository.
The research code also includes experimental options explored during development;
the [paper](https://arxiv.org/abs/2604.27899) specifies which components and settings
were used for the reported results.

Each position contains a discrete value token, a continuous value, a modality ID,
and seven time features. Padding is identified by modality ID: token ranges can
share endpoints. The decoder combines learned token and modality embeddings,
continuous-value encodings, temporal encodings, and optional age/gender features.
Age/gender features become available only at or after their first recorded event,
and must also be visible through the ELIS attention mask. Earlier queries cannot
use demographic values recorded later. The target modality and time representations are supplied to the backbone, so
predictions are conditional on the measurement being queried and its time.

The decoder uses GELU, causal attention, and tanh logit soft-capping.
Configurable options include embedding skip connections, value-related attention
features, and attention-value gates. Defaults are in `healthformer/config.py`.

Standard training predicts shifted tokens, with the existing soft-label objective
and optional continuous MAE. ELIS adds its longitudinal loss with eligibility
checks, query construction, and attention masks. Both branches accumulate gradients
and use the same optimizer. Each completed optimizer update reports the mean weighted
objective and its standard/ELIS contributions, learning rate, and gradient norm.
Skipped ELIS branches contribute zero; rejected accumulation windows are discarded.
Every selected training participant remains eligible for standard training.

`model.py` defines representations and the decoder; `data.py` and `inputs.py`
handle artifacts, augmentation, and loaders. `training.py` implements both loss
branches; `prompts.py` shares ELIS sequence construction and batching with validation.
`evaluation.py` monitors CE; `longitudinal.py` implements ELIS validation.
`checkpointing.py` and `identity.py` manage continuation and experiment identity.
ELIS training and validation use the same attention rule: first-visit rows are
causal, queries see the first visit and themselves, and other future rows are
self-only. Padding rows also have self-attention to avoid empty attention rows.
Both branches remove the final target before padding inputs, use integer token
and modality IDs, and preserve the artifact's continuous-value dtype. Visit eligibility
and record lengths are computed once and reused by dataset views and the ELIS sampler.

The model returns logits. Training and validation share source tensors; training
clones samples before augmentation. Runs with zero usable training batches or an
epoch with no accepted optimizer updates fail explicitly. OOM and other unexpected
runtime errors abort the run. Use a smaller context or batch size for a fresh run
after an OOM.

See the [training guide](TRAINING.md) for input fields, objective definitions,
validation protocols, and checkpoint behavior.
