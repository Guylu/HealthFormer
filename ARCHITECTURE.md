# Architecture

A high-level prose description of HealthFormer. Exact hyperparameters,
ablations, and quantitative results are reported in the paper
(arXiv [2604.27899](https://arxiv.org/abs/2604.27899)); this document is
the conceptual map.

## Inputs

HealthFormer consumes the unified, padded token sequence produced by
the HPP tokenization pipeline. Each position in the sequence carries
three parallel channels:

| Channel | Tensor | Dtype | Semantics |
|---|---|---|---|
| Token id | `tokens` | `int16` | The discretized value within a modality's vocabulary slice |
| Modality id | `modalities` | `int16` | Which signal this token came from (CGM, lab, diet log, ...) |
| Temporal vector | `time_expanded` | `int16`, 7 dims | `day_of_week`, `hour`, `minute` (15-min buckets), `month`, `year`, `day_of_month`, `sleep` |

Padding uses `tokens == 0`, `modalities == number_of_modalities`, and an
all-zero temporal vector. The modality channel is the canonical mask:
`modalities != number_of_modalities` selects real positions.

## Embedding

Each input position is embedded by summing four learned components:

1. **Token id embedding.** A standard `nn.Embedding` over the unified
   vocabulary. The token vocabulary is partitioned into per-modality
   contiguous slices (so token id 273 always belongs to one specific
   modality), letting the model exploit local structure inside a
   slice without needing a separate per-modality vocab.
2. **Modality id embedding.** A learned embedding over the
   `number_of_modalities + 1` modalities (the `+1` is the padding
   sentinel). This carries a strong prior about what the token
   "means" in physical terms.
3. **Temporal embeddings.** Seven independent embeddings, one per
   temporal dimension, summed into a single temporal vector. Splitting
   per-dim (rather than concatenating a learned position over a
   product of dims) keeps the parameter count manageable and lets each
   dim share statistics across all positions sharing that value.
4. **Sequence position.** Either rotary position embeddings on the
   attention or a small learned positional embedding added to the sum.

All four are summed and passed through a final layer norm.

## Backbone

A decoder-only transformer with pre-norm self-attention. The exact
depth, hidden dimension, and head count are tuned in the paper; the
architecture is intentionally vanilla so contributions from the
input representation and training objective remain isolated from
backbone choices.

Standard components:

- Pre-norm residual blocks (RMSNorm or LayerNorm)
- Multi-head self-attention with causal masking
- SwiGLU or GELU feedforward
- Rotary or learned positional encoding
- Tied or untied input and output token embeddings

## Output heads

Two heads share the backbone:

1. **Per-modality next-token prediction.** A linear projection from
   the backbone output to the unified vocabulary, trained with
   cross-entropy. Because each modality occupies a contiguous slice
   of the vocab, the model effectively learns per-modality
   distributions while sharing the trunk.
2. **Time-delta regression.** A small MLP head that predicts the
   gap (in some normalized time unit) to the next event. This
   shapes the model's notion of "when" the next typed event arrives,
   not just "what" it is.

## Training objective

The total loss is a weighted sum:

- Next-token cross-entropy, masked over real (non-padding) positions
- Time-delta regression loss (Huber or MSE) on the gap targets
- Optional: an intervention-prediction auxiliary loss for the
  intervention-conditioned simulation experiments

Per-modality loss weighting balances the contribution of dense
modalities (CGM, wearables) against sparse ones (annual labs).
Without it, the model collapses toward the dense modalities and
under-fits the rare-but-clinically-important sparse signals.

## Generation and intervention-conditioned simulation

At inference, given a prefix of a participant's tokenized trajectory,
the decoder generates plausible continuations autoregressively. The
"intervention" mechanism is implemented by injecting a typed
intervention token (for example, a specific medication start) at a
chosen position and conditioning the continuation on that injected
prefix. Counterfactual trajectories are obtained by running the same
prefix with and without the injected token and comparing the
generated continuations.

This is **not a digital twin**. The model produces samples from a
learned conditional distribution over future trajectories; it does
not claim to simulate a specific individual's biology.

## Evaluation

Three protocols, summarized here, detailed in the paper:

1. **Held-out cohort split.** Standard log-likelihood and
   per-modality next-token accuracy on a held-out portion of the
   primary cohort.
2. **External transfer.** Zero-shot evaluation on independent
   external cohorts to measure out-of-distribution generalization.
3. **Intervention experiments.** Quantitative comparison of model-
   generated counterfactual trajectories against documented
   intervention outcomes, where available.

## What is *not* in this repository

- The trained model weights.
- The real tokenized dataset (only the synthetic dummy ships here).
- The exact hyperparameters and training schedule (in the paper).
- The four external cohort splits used for transfer evaluation.
