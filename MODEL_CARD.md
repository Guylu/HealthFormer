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

- **Developed by:** Guy Lutsker and collaborators (Weizmann Institute of Science, Eran Segal lab; NVIDIA).
- **Model type:** Decoder-only transformer over a unified per-modality token vocabulary.
- **Language:** Not applicable (the input is tokenized health data, not natural language).
- **License:** Apache 2.0 (this repository / source code). Model weights are not released.

### Sources

- **Repository:** https://github.com/Guylu/HealthFormer
- **Paper:** https://arxiv.org/abs/2604.27899
- **Architecture overview:** [`ARCHITECTURE.md`](ARCHITECTURE.md)

## Uses

### Direct Use

- Modeling longitudinal multimodal health trajectories.
- Generating plausible continuations of a participant's tokenized history.
- Intervention-conditioned simulation: given a prefix of a trajectory, generate continuations conditioned on an injected intervention token (for example, the start of a medication) to study counterfactual outcomes.

### Out-of-Scope Use

HealthFormer is a research model. It is **not** a clinical decision
support tool, **not** a diagnostic device, and **not** a digital twin
of any individual. Generated trajectories are samples from a learned
conditional distribution; they are not patient-specific predictions.
Do not use the model for individual treatment decisions, eligibility
determinations, or any application where errors carry clinical, legal,
or financial consequences.

## Bias, Risks, and Limitations

- **Cohort composition.** The model is trained on a specific cohort (HPP / 10K). Demographic, geographic, and clinical-practice biases of that cohort propagate into the model.
- **Modality coverage.** Some modalities are sparse (annual labs) and others are dense (CGM, wearables). Per-modality loss weighting mitigates dense-modality dominance, but rare modalities are still less well-modeled.
- **Tokenization-induced limits.** Continuous signals are discretized into per-modality bins; resolution within a bin is lost.
- **Temporal granularity.** Time features are bucketed (15-minute minute buckets, day / month / year). Sub-15-minute timing is not represented.
- **Counterfactual claims.** Intervention-conditioned generations describe distributional shifts the model has learned, not causal outcomes for a specific person.

Users should evaluate the model on their target population before any
downstream use.

## How to Get Started

The model weights are not currently distributed with this repository.
To exercise the data path end-to-end against a fully synthetic
fixture (no real cohort data):

```bash
pip install -r requirements.txt
python3 examples/load_dummy.py
```

See [`DUMMY_DATA.md`](DUMMY_DATA.md) and [`examples/load_dummy.py`](examples/load_dummy.py)
for the schema and a walkthrough.

## Training

- **Data:** Human Phenotype Project (HPP) / 10K cohort. Modalities include continuous glucose monitoring (CGM), clinical labs (blood, DEXA, ECG, ultrasound, ABI, retina, Nightingale, microbiome), body composition, sleep, diet logs, exercise, medications, gait, wearables (Apple Health / Google Fit), and lifestyle and demographic questionnaires.
- **Preprocessing:** per-modality binning into a unified token vocabulary, with seven temporal channels (`day_of_week`, `hour`, `minute`, `month`, `year`, `day_of_month`, `sleep`). See [`DUMMY_DATA.md`](DUMMY_DATA.md).
- **Objective:** next-token cross-entropy over the unified vocabulary, masked over real (non-padding) positions, with per-modality loss weighting.
- **Architecture:** decoder-only transformer with token, modality, and per-dim temporal embeddings summed at the input. See [`ARCHITECTURE.md`](ARCHITECTURE.md).
- **Hyperparameters, scale, training compute:** reported in the paper, [arXiv 2604.27899](https://arxiv.org/abs/2604.27899).

## Evaluation

The paper evaluates on a held-out cohort split, on independent
external cohorts (zero-shot transfer), and through intervention
experiments comparing model-generated counterfactual trajectories
against documented outcomes. Numbers and protocol details are in the
paper, [arXiv 2604.27899](https://arxiv.org/abs/2604.27899).

## Citation

```bibtex
@article{lutsker2026healthformer,
  title   = {HealthFormer (preprint)},
  author  = {Lutsker, Guy and others},
  journal = {arXiv preprint arXiv:2604.27899},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.27899},
}
```

See also [`CITATION.bib`](CITATION.bib).

## Model Card Authors

Guy Lutsker.
