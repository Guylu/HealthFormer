# HealthFormer

HealthFormer is a multimodal generative transformer for longitudinal health
records, developed using the Human Phenotype Project (HPP / 10K cohort). This
repository provides its research training implementation, validation and
longitudinal validation routines, and synthetic examples.

[Paper](https://arxiv.org/abs/2604.27899) · [Training guide](TRAINING.md) ·
[Architecture](ARCHITECTURE.md) · [Model card](MODEL_CARD.md)

The HealthFormer research code includes experimental components and alternative
configurations explored during development. Some were not used in the reported
model because they did not improve performance. The paper describes the model
and experimental settings used for the reported results.

## Installation

```bash
git clone https://github.com/Guylu/HealthFormer.git
cd HealthFormer
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional online tracking requires a separate W&B installation; see
[Tracking and data handling](TRAINING.md#tracking-and-data-handling).

The dependency pins match the checked Python 3.9.6 CPU environment. For GPU
training, install the appropriate CUDA build of PyTorch. GPU numerical parity
and multi-node execution have not been verified for this layout.

## Run a synthetic training example

```bash
python examples/make_training_fixture.py
python train.py --config data/synthetic/config.json --run
```

This creates synthetic data and runs a small model on CPU, validates it, and
saves a checkpoint. It exercises the standard training branch; ELIS has separate
regression and distributed checks. It does not reproduce the paper's experiments.

For a walkthrough of the separate token/modality/time schema example, run
`python examples/load_dummy.py`. That bundled fixture illustrates the schema;
use the generated fixture above for training. See [Dummy data](DUMMY_DATA.md).

## Train on your artifacts

Edit `examples/training_config.json` to point to compatible tokenized data, then run:

```bash
python train.py --config examples/training_config.json --check
python train.py --config examples/training_config.json --run
```

The example enables standard validation and checkpoint saving. Longitudinal
validation remains opt-in; see the [training guide](TRAINING.md#longitudinal-validation).

The original HPP data, preprocessing pipeline, and trained
weights are not distributed. The [training guide](TRAINING.md) describes the
input contract, split policies, checkpoint resume, and optional tracking.

## Code and checks

`train.py` is the entry point. `healthformer/` contains the model, data handling,
training, validation, and checkpoint code. The [training guide](TRAINING.md)
describes ELIS objectives and validation scope; the
[architecture overview](ARCHITECTURE.md) describes the model.

```bash
python -m unittest discover -s tests -t . -v
```

The checks include actual synthetic CPU training and checkpoint continuation.
Distributed checks use two CPU processes and local sockets. On macOS, set
`GLOO_SOCKET_IFNAME=lo0` if interface discovery fails.

## Citation and acknowledgment

Please cite the [paper](https://arxiv.org/abs/2604.27899); the BibTeX entry is in
[CITATION.bib](CITATION.bib).

Thanks to Katie Link ([@katielink](https://github.com/katielink)) for helping
organize the code!

## License

[Apache-2.0](LICENSE). Decoder adaptations from nanoGPT retain their MIT notice;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Training retains the original soft-label defaults. Loss vocabulary, accumulation
weighting, context limits, causal demographic features, validation coverage,
distinct-year grouping, and exact-resume requirements
are documented in [TRAINING.md](TRAINING.md#objective-and-validation-definitions).
