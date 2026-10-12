---
license: apache-2.0
language:
  - en
tags:
  - multimodal
  - transformer
  - health
  - time-series
  - generative
  - decoder-only
datasets:
  - human-phenotype-project
library_name: pytorch
---

# Model Card for HealthFormer

HealthFormer is a multimodal generative transformer for longitudinal health
records, developed using the Human Phenotype Project (HPP / 10K cohort).
The research implementation supports pretraining, standard validation, and
ELIS longitudinal training and validation.

The code includes experimental options explored during development. The
[paper](https://arxiv.org/abs/2604.27899) describes the model and settings used
for the reported results.

## Model details

- **Developed by:** Guy Lutsker and collaborators (Weizmann Institute of Science, Eran Segal lab; NVIDIA).
- **Model type:** Decoder-only transformer with multimodal inputs and target modality/time conditioning.
- **Inputs:** Discrete tokens, continuous values, modality IDs, and seven temporal features.
- **Outputs:** Token logits; validation also derives continuous predictions from token probabilities and bin values.
- **License:** [Apache-2.0](LICENSE), with the retained notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Availability:** Research code and synthetic examples. HPP records, preprocessing, and trained weights are not distributed.

See the [architecture overview](ARCHITECTURE.md) and [training guide](TRAINING.md)
for implementation details.

## Intended use

Research on longitudinal multimodal health records, including next-token
prediction and prediction of future measurements conditioned on their modality
and time. ELIS provides the longitudinal training and validation path.

HealthFormer is a research model. Clinical or diagnostic validity and suitability
for individual treatment decisions are not established by this code. Generalization
to other populations and measurement practices requires independent evaluation.
Intervention-conditioned generation alone does not identify causal treatment effects.

## Training and validation

Standard training uses shifted token targets, the retained soft-label objective,
and optional continuous MAE. ELIS adds a weighted longitudinal objective. The
[training guide](TRAINING.md#objective-and-validation-definitions) defines the
losses, averaging, vocabulary normalization, and validation coverage.

Standard validation reports hard-target cross-entropy. Longitudinal validation
reports metrics for selected future measurements using the documented
[distinct-year protocol](TRAINING.md#longitudinal-validation).
The paper provides the experimental protocols and reported research results.

## Interpretation

- Predictions are conditional on supplied measurement types and times.
- Continuous input encodings can retain information within token bins; continuous predictions derived from token probabilities depend on the supplied bins.
- Temporal resolution follows the input artifact's encoding. The bundled schema fixture and generated training fixture use illustrative vocabularies.
- Longitudinal grouping uses encoded years; it cannot distinguish two visits in the same year.
- Validation coverage depends on target selection and context limits. Reported coverage and counts accompany the metrics.

## Getting started

Follow the [README](README.md#installation) to install dependencies, then run:

```bash
python examples/make_training_fixture.py
python train.py --config data/synthetic/config.json --run
```

This runs a small synthetic training and validation example on CPU. The separate
[`HPP_tokenized_dummy.pt`](DUMMY_DATA.md) fixture illustrates the token/modality/time
schema and can be inspected with `python examples/load_dummy.py`.

Regression checks cover synthetic training, checkpoint continuation, ELIS, and
local distributed CPU execution. CUDA numerical parity and multi-node execution
remain unverified. Synthetic checks do not reproduce the paper's experiments.

## Citation

Please cite the [paper](https://arxiv.org/abs/2604.27899). The full BibTeX entry is
in [CITATION.bib](CITATION.bib).

## Model card author

Guy Lutsker.
