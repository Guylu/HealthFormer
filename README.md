# HealthFormer

A multimodal generative transformer trained on the Human Phenotype Project
(HPP / 10K cohort), modeling longitudinal health trajectories across labs,
continuous glucose monitoring, body composition, sleep, diet, wearables,
microbiome, questionnaires, and other modalities as a single sequence of
typed tokens.

The model takes the tokenized output of the HPP tokenization pipeline
(see [HPPTokenization](https://github.com/Guylu/HPPTokenization)) as
input. Each token carries three channels: a token id from a unified
vocabulary, a modality id identifying the source signal, and a vector of
seven temporal features (`day_of_week`, `hour`, `minute`, `month`, `year`,
`day_of_month`, `sleep`). The decoder-only transformer learns
intervention-conditioned simulation of future health states from past
ones, with per-modality masked prediction heads at training time.

Reference: arXiv [2604.27899](https://arxiv.org/abs/2604.27899).

## Repository contents

- `train.py` entry point for training (work in progress)
- `HPP_tokenized_dummy.pt` a fully synthetic 700 KB fixture matching the real pipeline schema, for development without cohort access
- `DUMMY_DATA.md` provenance and schema documentation for the fixture
- `LICENSE` Apache 2.0

## Quick start

```python
import torch

blob = torch.load("HPP_tokenized_dummy.pt", map_location="cpu", weights_only=False)
print(blob["tokens"].shape)            # (20, 2000)
print(blob["vocab_size"])              # 1123
print(blob["number_of_modalities"])    # 20
```

See [`DUMMY_DATA.md`](DUMMY_DATA.md) for the full schema and a description of
what the fixture contains (and what it deliberately does not).

## Status

Code release is in progress. The dummy fixture and schema documentation
land first so that downstream consumers can start building loaders.

## License

Apache 2.0. See [`LICENSE`](LICENSE).
