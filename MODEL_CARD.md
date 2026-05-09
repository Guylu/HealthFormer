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

HealthFormer is a multimodal generative transformer trained on the
Human Phenotype Project (HPP / 10K cohort), modeling longitudinal
health trajectories across labs, continuous glucose monitoring, body
composition, sleep, diet, wearables, microbiome, questionnaires, and
other modalities as a single sequence of typed tokens.

## Model Details

### Model Description

- **Developed by:** Guy Lutsker and collaborators (Weizmann Institute of Science, Eran Segal lab; NVIDIA).
- **Model type:** Decoder-only transformer over a unified per-modality token vocabulary.
- **Language:** Not applicable (the input is tokenized health data, not natural language).
- **License:** Apache 2.0 (this repository / source code). Model weights are not released; if and when published, license terms will be updated here.
- **Finetuned from:** Trained from scratch.

### Model Sources

- **Repository:** https://github.com/Guylu/HealthFormer
- **Paper:** https://arxiv.org/abs/2604.27899
- **Architecture overview:** [`ARCHITECTURE.md`](ARCHITECTURE.md)

## Uses

### Direct Use

- Modeling longitudinal multimodal health trajectories.
- Generating plausible continuations of a participant's tokenized history.
- Intervention-conditioned simulation: given a prefix of a trajectory, generate continuations conditioned on an injected intervention token (for example, the start of a medication) to study counterfactual outcomes.

### Downstream Use

- Probing the learned representations for clinically meaningful structure.
- Initialization for transfer to smaller, narrower health-trajectory tasks.

### Out-of-Scope Use

HealthFormer is a research model. It is **not** a clinical decision
support tool, **not** a diagnostic device, and **not** a digital twin
of any individual. Generated trajectories are samples from a learned
conditional distribution; they are not patient-specific predictions.
Do not use the model for individual treatment decisions, eligibility
determinations, or any application where errors carry clinical, legal,
or financial consequences.

## Bias, Risks, and Limitations

- **Cohort composition.** The model is trained on a specific cohort (HPP / 10K). Demographic, geographic, and clinical-practice biases of that cohort propagate into the model. Generalization to other populations is an empirical question and is partially addressed in the paper's external-cohort evaluation.
- **Modality coverage.** Some modalities are sparse (annual labs) and others are dense (CGM, wearables). The model is trained with per-modality loss weighting to mitigate dense-modality dominance, but rare modalities are still less well-modeled.
- **Tokenization-induced limits.** Continuous signals are discretized into per-modality bins; resolution within a bin is lost.
- **Temporal granularity.** Time features are bucketed (15-minute minute buckets, day/month/year). Sub-15-minute timing is not represented.
- **Counterfactual claims.** Intervention-conditioned generations describe distributional shifts the model has learned, not causal outcomes for a specific person.

### Recommendations

Users should evaluate the model on their target population before any
downstream use. Any application that surfaces model output to
clinicians or patients requires an independent clinical and ethical
review.

## How to Get Started with the Model

The model weights are not currently distributed with this repository.
To exercise the data path end-to-end against a fully synthetic
fixture (no real cohort data), see [`DUMMY_DATA.md`](DUMMY_DATA.md)
and [`examples/load_dummy.py`](examples/load_dummy.py):

```bash
pip install -r requirements.txt
python3 examples/load_dummy.py
```

## Training Details

### Training Data

- **Source:** Human Phenotype Project (HPP) / 10K cohort.
- **Modalities:** continuous glucose monitoring (CGM), clinical labs (blood, DEXA, ECG, ultrasound, ABI, retina, Nightingale, microbiome), body composition, sleep, diet logs, exercise, medications, gait, wearables (Apple Health / Google Fit), and lifestyle and demographic questionnaires.
- **Preprocessing:** per-modality binning into a unified token vocabulary, with seven temporal channels (`day_of_week`, `hour`, `minute`, `month`, `year`, `day_of_month`, `sleep`). See [`DUMMY_DATA.md`](DUMMY_DATA.md) for the saved schema.
- **Cohort scale:** [TBD: confirm against the paper's reported number of participants and total tokens].

### Training Procedure

- **Objective:** next-token cross-entropy over the unified vocabulary, masked over real (non-padding) positions, with per-modality loss weighting.
- **Architecture:** decoder-only transformer with token, modality, and per-dim temporal embeddings summed and layer-normalized at the input. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full description.

#### Training Hyperparameters

[TBD: depth, hidden dim, head count, context length, optimizer, learning rate schedule, batch size, total training steps, precision are all reported in the paper. Fill in once the camera-ready is available.]

#### Speeds, Sizes, Times

- **Parameters:** [TBD: fill in from the paper].
- **Training compute:** [TBD: fill in from the paper].
- **Training duration:** [TBD: fill in from the paper].

## Evaluation

### Testing Data, Factors and Metrics

#### Testing Data

- **Held-out split:** a portion of the primary cohort.
- **External cohorts:** the paper reports zero-shot transfer to independent external cohorts. [TBD: list the specific cohorts once the paper is public.]

#### Factors

- Per-modality breakdowns (dense vs sparse modalities).
- Demographic subgroups (sex, age band) for fairness analysis.

#### Metrics

- Per-modality next-token negative log-likelihood / accuracy.
- Quantitative comparisons against documented outcomes for the intervention experiments.
- [TBD: list any additional metrics reported in the paper.]

### Results

[TBD: do not transcribe numbers here. Refer readers to the paper:
arXiv [2604.27899](https://arxiv.org/abs/2604.27899).]

## Environmental Impact

[TBD: report training compute, hardware (GPU type and count), runtime,
and estimated CO2 emissions per the
[ML CO2 Impact calculator](https://mlco2.github.io/impact#compute) once
training-run telemetry is available.]

## Technical Specifications

### Model Architecture and Objective

Decoder-only transformer with three input channels per token (token id,
modality id, 7-dim temporal vector). Single next-token cross-entropy
head over the unified vocabulary. Full description in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

### Compute Infrastructure

#### Hardware

[TBD: GPU type and count.]

#### Software

- PyTorch (see [`requirements.txt`](requirements.txt)).

## Citation

**BibTeX:** see [`CITATION.bib`](CITATION.bib).

```
@article{lutsker2026healthformer,
  title   = {HealthFormer (preprint)},
  author  = {Lutsker, Guy and others},
  journal = {arXiv preprint arXiv:2604.27899},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.27899},
}
```

## Glossary

- **HPP / 10K cohort:** Human Phenotype Project, a deeply phenotyped longitudinal cohort.
- **Modality:** a typed signal source (CGM, lab panel, sleep stage, ...). Each occupies a contiguous slice of the unified token vocabulary.
- **Intervention-conditioned simulation:** generating continuations of a tokenized trajectory after injecting a typed intervention token at a chosen position.

## More Information

For data schema, padding semantics, and a working synthetic example
that runs without cohort access, see [`DUMMY_DATA.md`](DUMMY_DATA.md)
and [`examples/load_dummy.py`](examples/load_dummy.py).

## Model Card Authors

Guy Lutsker.

## Model Card Contact

[TBD: maintainer contact, e.g. GitHub Issues link or email.]
