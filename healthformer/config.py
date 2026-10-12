"""HealthFormer model, data, and training settings."""
import math


config = dict(

    allow_wandb=False,
    # Whether to enable Weights & Biases logging.

    seed=42,
    # Random seed for reproducibility.

    embed_skip_connections=True,
    embed_skip_init=0.0,

    use_value_extras=True,
    n_value_extras=4,
    v_extra_gate_init_bias=-2.0,

    use_attention_value_gate=True,
    # Mix attention output with the value path; attention is always computed.
    attention_value_gate_init_bias=2.0,

    log_wandb_metric_every_k=10,
    # Console progress interval in optimizer updates; training metrics are logged every update.

    reduce_context_in_train=True,
    # Whether to reduce the context length during training.

    epochs=200,
    # Number of training epochs.

    lr=3e-5,
    # Base learning rate.

    b1=0.88,
    # Beta1 parameter for optimizers (e.g., Adam/AdamW).

    b2=0.99,
    # Beta2 parameter for optimizers.

    weight_decay=0.0,
    # Weight decay coefficient (L2 regularization).

    gamma=0.001,
    # Lower-bound factor for learning rate decay (used in cosine decay or scheduled methods).

    dropout=0.0,
    # Dropout probability within the model.

    batch_per_gpu=2,
    # Batch size per GPU device.

    n_embd=320,
    # Dimensionality of each token / embedding.

    n_heads=1,
    # Number of attention heads in the transformer.

    n_layers=1,
    # Number of transformer blocks/layers.

    dim_feedforward_scaling=1,
    # Multiplier for the hidden dimension in the feed-forward layer (e.g., 4 would mean 4 * n_embd).

    embd_scaling=1,
    # Additional scaling factor for embedding transformations (e.g., in MLP expansions).

    max_seq_length=25_000,
    # Maximum sequence length supported (context window).

    validation_sequence_policy='prefix',
    validation_max_tokens=10_000,
    # full rejects records beyond model capacity; sliding scores each target once.

    Elis_augmentation_prob=1,
    # Probability of attempting ELIS future-visit queries on an eligible batch.

    elis_max_input_length=8000,
    # Maximum input length allowed by the ELIS precheck.

    elis_loss_weight=0.01,
    # Weight for the Elis augmentation loss term.

    not_elis_loss_weight=1,
    # Positive weight for standard pretraining; ELIS is an optional auxiliary loss.

    loss_vocabulary='global',
    # Global preserves the effective historical loss; modality restricts normalization.
    add_mae_loss_on_continuous_metrics=True,
    # If True, then add a mean absolute error loss term on continuous metrics (e.g., glucose values).

    mae_loss_scale=0.1,
    # Coefficient for the MAE loss term.

    accumulate_grad_batches=1,
    # Number of steps to accumulate gradients before performing an optimizer step.

    warmup_steps=500,
    # Number of steps for linear warmup of the learning rate.

    precision='bf16-mixed',
    # Fabric owns the precision context for training and evaluation.

    max_grad_norm=1,
    # Gradient clipping norm.

    tanh_norm=30,
    # Logit soft-capping scale.

    add_rms_norm=True,
    # Apply RMS normalization to query/key tensors and block inputs.

    zero_init_mlp=True,
    # Zero-initialize attention and MLP output projections.

    evaluate=False,
    # Whether to run evaluation steps at certain intervals during training.

    reg_positional_encodings=True,
    # If True, use regular (sine-cosine) positional encodings in addition to other encodings.

    temporal_modeling=True,
    # If True, enable separate embeddings for time features (minute, hour, etc.).

    temporal_positional_encodings=True,
    # If True, add additional sine-cosine embeddings for the time dimension.

    sinusoidal_time_embeddings=True,
    # Add per-feature sinusoidal time embeddings and a learned offset.

    use_learned_token_embds=True,
    # Add learned token embeddings.

    continuous_value_embeddings=True,
    # Add sinusoidal encodings of normalized continuous values.

    modality_token=True,
    # If True, add learned embeddings to indicate each token’s modality.

    complex_mod_time_emb=True,
    # If True, apply an MLP transform to the modality/time embeddings before combining with token embeddings.

    soft_labels=True,
    # Whether to apply soft-label (Gaussian-like) targets instead of hard one-hot.

    soft_labels_scale=1,
    # Scaling factor for the soft-label targets.

    SL_sigma=0.01,
    # Standard deviation used in the Gaussian soft-label approach.

    augmentation_rate=0.2,
    # Proportion of tokens to alter during noise augmentation.

    augmentation_chance=0.2,
    # Probability to apply noise augmentation on a given sequence.

    random_removal_rate=0.2,
    # Fraction of tokens randomly removed when random_removal_chance is triggered.

    random_removal_chance=0.2,
    # Probability to trigger random token removal.

    random_block_removal_rate=0.2,
    # Relative size of each random block removal if triggered.

    random_block_removal_chance=0.2,
    # Probability to trigger random block removal.

    random_block_removal_number=2,
    # Base block count with integer jitter [-3, 2], clamped to [1, real length]; 0 disables.

    random_modality_subset_chance=0.2,
    # Probability to select only a subset of modalities for a given sequence.

    random_modality_subset_fraction=0.2,
    # Fraction of modalities to keep if random_modality_subset_chance is triggered.

    chance_to_remove_time_series_data=0.2,
    # Probability to remove the configured time-series modality groups.

    random_modality_exclusion_chance=0.2,
    # Probability to exclude a random set of modalities from a given sequence.

    add_age_gender_as_pe=True,
    # If True, adds positional encodings for age and gender to each token's embedding.

    save=False,
    # Whether to save the model at certain checkpoints.

    find_unused_parameters=True,
    # Required for distributed runs: optional branches can leave parameters unused.

    percent_validation=20,
    # Historical inclusive ID-modulo threshold; 20 selects residues 0 through 20.

    train_sampler='eli_friendly',
    # ELIS-friendly sampling or ordinary distributed sampling.

    stop_early_now=None,
    data_path=None,
    accelerator='cuda',
    devices=1,
    split_policy='historical_modulo_inclusive',
    max_validation_batches=None,
    checkpoint_dir='checkpoints',
    resume_from=None,
    initialize_from=None,
    wandb_entity=None,
    wandb_project=None,
    longitudinal_evaluation=False,

)


# Accept earlier configuration spellings without changing the trained objective.
CONFIG_ALIASES = {
    'yoads_temporal_enconding': 'sinusoidal_time_embeddings',
    'use_yoads_encoding_for_tokens': 'continuous_value_embeddings',
    'elis_min_len': 'elis_max_input_length',
    'use_sparse_attn_gate': 'use_attention_value_gate',
    'sparse_gate_init_bias': 'attention_value_gate_init_bias',
}


def normalize_config(settings):
    normalized = dict(settings)
    for old, new in CONFIG_ALIASES.items():
        if old in normalized:
            value = normalized.pop(old)
            if new in normalized and normalized[new] != value:
                raise ValueError(f'Conflicting configuration values for {old} and {new}.')
            normalized[new] = value
    return normalized


def is_validation_id(participant_id, settings):
    threshold = settings.percent_validation
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 100:
        raise ValueError('percent_validation must be an integer from 0 to 100.')
    policy = getattr(settings, 'split_policy', 'historical_modulo_inclusive')
    if policy == 'historical_modulo_inclusive':
        return participant_id % 100 <= threshold
    if policy == 'modulo':
        return participant_id % 100 < threshold
    raise ValueError('Unknown split_policy: ' + str(policy))

def validate_distributed_config(settings, world_size):
    if world_size > 1 and not settings.get('find_unused_parameters', True):
        raise ValueError('Distributed training requires find_unused_parameters=True because '
                         'optional branches can leave parameters unused.')


def validate_methodology_config(settings):
    settings = normalize_config(settings)
    precision = settings.get('precision', config['precision'])
    if precision not in ('32-true', 'bf16-mixed'):
        raise ValueError('precision must be 32-true or bf16-mixed; FP16 is unsupported.')
    if settings.get('accelerator') == 'cpu' and precision != '32-true':
        raise ValueError('CPU runs require precision=32-true.')
    unknown = set(settings) - set(config)
    if unknown:
        raise ValueError('Unknown or removed training settings: ' + ', '.join(sorted(unknown)))
    if settings.get('train_sampler', 'eli_friendly') not in ('eli_friendly', 'distributed'):
        raise ValueError('train_sampler must be eli_friendly or distributed.')
    if settings.get('loss_vocabulary', 'global') not in ('global', 'modality'):
        raise ValueError('loss_vocabulary must be global or modality.')
    if settings.get('validation_sequence_policy', 'prefix') not in ('prefix', 'full', 'sliding'):
        raise ValueError('validation_sequence_policy must be prefix, full or sliding.')
    if settings.get('accelerator', config['accelerator']) not in ('cpu', 'cuda'):
        raise ValueError('accelerator must be cpu or cuda.')
    minimums = dict(epochs=1, devices=1, batch_per_gpu=1, accumulate_grad_batches=1,
                    log_wandb_metric_every_k=1, n_embd=1, n_heads=1, n_layers=1,
                    dim_feedforward_scaling=1, embd_scaling=1, n_value_extras=1,
                    max_seq_length=2, validation_max_tokens=2, elis_max_input_length=1,
                    warmup_steps=0, random_block_removal_number=0,
                    max_validation_batches=1, stop_early_now=0)
    for name, minimum in minimums.items():
        value = settings.get(name, config[name])
        if value is None and name in ('max_validation_batches', 'stop_early_now'):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    validate_distributed_config(settings, settings.get('devices', config['devices']))
    width = settings.get('n_embd', config['n_embd'])
    token_width = width - (12 if settings.get('temporal_positional_encodings',
                                            config['temporal_positional_encodings']) else 0)
    if token_width < 4 or token_width % 2:
        raise ValueError('n_embd must give an even token-embedding width >= 4; '
                         'temporal_positional_encodings reserves 12 dimensions.')
    if width % settings.get('n_heads', config['n_heads']):
        raise ValueError('n_embd must be divisible by n_heads.')
    if settings.get('sinusoidal_time_embeddings', config['sinusoidal_time_embeddings']) and width % 16:
        raise ValueError('n_embd must be divisible by 16 when sinusoidal_time_embeddings is enabled.')
    probabilities = ('Elis_augmentation_prob', 'dropout', 'gamma', 'augmentation_rate',
                     'augmentation_chance', 'random_removal_rate', 'random_removal_chance',
                     'random_block_removal_rate', 'random_block_removal_chance',
                     'random_modality_subset_chance', 'random_modality_subset_fraction',
                     'chance_to_remove_time_series_data', 'random_modality_exclusion_chance')
    positive = ('lr', 'max_grad_norm', 'tanh_norm', 'SL_sigma', 'not_elis_loss_weight')
    nonnegative = ('weight_decay', 'elis_loss_weight', 'mae_loss_scale', 'soft_labels_scale')
    for name in (*probabilities, *positive, *nonnegative, 'b1', 'b2'):
        value = settings.get(name, config[name])
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (name in positive and value == 0)
                or (name in probabilities and value > 1)
                or (name in ('b1', 'b2') and value >= 1)):
            raise ValueError(f'{name} is outside its supported numeric range.')
    if settings.get('resume_from') and settings.get('initialize_from'):
        raise ValueError('Choose resume_from or initialize_from, not both.')
