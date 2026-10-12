"""Decoder layers, multimodal representations, and model-level losses."""
from .losses import vocabulary_cross_entropy

import torch.nn.functional as F
from types import SimpleNamespace
import inspect
import math
import torch.nn as nn
import numpy as np
import torch

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.dropout_resid = nn.Dropout(config.dropout)

        self.head_dim = self.n_embd // self.n_head

        # Extra embeddings mixed into V
        self.use_value_extras = getattr(config, "use_value_extras", False)
        if self.use_value_extras:
            self.n_value_extras = int(getattr(config, "n_value_extras", 4))

            # Learned bank: [n_extras, n_head, head_dim]
            self.v_extra_bank = nn.Parameter(
                0.02 * torch.randn(self.n_value_extras, self.n_head, self.head_dim)
            )

            # Router produces mixture weights per token/head: [B,T,n_head,n_extras]
            self.v_extra_router = nn.Linear(self.n_embd, self.n_head * self.n_value_extras, bias=False)

            # Gate controls how much extra gets injected per token/head: [B,T,n_head]
            self.v_extra_gate = nn.Linear(self.n_embd, self.n_head, bias=True)

        # Gate between the attention output and the value path; attention is always computed.
        self.use_attention_value_gate = getattr(config, "use_attention_value_gate", False)
        if self.use_attention_value_gate:
            # Keep this module name for checkpoint compatibility; it does not sparsify attention.
            self.sparse_attn_gate = nn.Linear(self.n_embd, self.n_head, bias=True)

        self.config = config

    def forward(self, x, mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # Mix extra embeddings into V
        if self.use_value_extras:
            # router weights: [B,T,n_head,n_extras]
            router_logits = self.v_extra_router(x).view(B, T, self.n_head, self.n_value_extras)
            mix_w = F.softmax(router_logits, dim=-1)

            # v_extra: [B,T,n_head,head_dim]
            v_extra = torch.einsum("bthm,mhd->bthd", mix_w, self.v_extra_bank)

            # gate: [B,T,n_head,1]
            gate = torch.sigmoid(self.v_extra_gate(x)).unsqueeze(-1)

            v = v + gate * v_extra

        if self.config.add_rms_norm:
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))  # QK normalization.

        # ---- SDPA attention ----
        qh = q.transpose(1, 2)  # [B, nh, T, hd]
        kh = k.transpose(1, 2)  # [B, nh, T, hd]
        vh = v.transpose(1, 2)  # [B, nh, T, hd]

        if mask is None:
            y = F.scaled_dot_product_attention(
                qh, kh, vh,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            y = F.scaled_dot_product_attention(
                qh, kh, vh,
                dropout_p=self.dropout if self.training else 0.0,
                attn_mask=mask
            )

        # Mix attention output with the value path.
        if self.use_attention_value_gate:
            # gate in [0,1], per token/head
            # sparse_attn_gate(x): [B, T, nh] -> transpose to [B, nh, T] -> unsqueeze to [B, nh, T, 1]
            g = torch.sigmoid(self.sparse_attn_gate(x)).transpose(1, 2).unsqueeze(-1)
            y = g * y + (1.0 - g) * vh

        # merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y = self.dropout_resid(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        scaling = config.dim_feedforward_scaling
        self.c_fc = nn.Linear(config.n_embd, scaling * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(scaling * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.config = config

    def forward(self, x, mask=None):
        if self.config.add_rms_norm:
            x = F.rms_norm(x, (x.size(-1),))
        x = x + self.attn(self.ln_1(x), mask)
        if self.config.add_rms_norm:
            x = F.rms_norm(x, (x.size(-1),))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """Autoregressive decoder with modality/time queries and an untied output projection."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tanh_norm = self.config.tanh_norm

        self.transformer = nn.ModuleDict(dict(
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # embedding -> every block skip
        self.embed_skip_connections = getattr(config, "embed_skip_connections", False)
        if self.embed_skip_connections:
            init = float(getattr(config, "embed_skip_init", 0.0))
            # Learnable per-layer residual strength.
            self.embed_skip_scale = nn.Parameter(torch.full((config.n_layer,), init))
        else:
            self.embed_skip_scale = None


        if config.complex_mod_time_emb:
            n_embd_ = config.n_embd

            scaling = config.embd_scaling
            self.mlp_for_mod_emb = nn.Sequential(
                nn.Linear(n_embd_, scaling * n_embd_),
                nn.GELU(approximate='tanh'),
                nn.Linear(scaling * n_embd_, n_embd_),
                nn.Dropout(config.dropout),
            )
            self.mlp_for_time_emb = nn.Sequential(
                nn.Linear(n_embd_, scaling * n_embd_),
                nn.GELU(approximate='tanh'),
                nn.Linear(scaling * n_embd_, n_embd_),
                nn.Dropout(config.dropout),
            )
        else:
            self.mlp_for_mod_emb = nn.Identity()
            self.mlp_for_time_emb = nn.Identity()

        # init params
        self.apply(self._init_weights)
        self._restore_special_initialization()

    def _restore_special_initialization(self):
        """Restore specialized values overwritten by the generic initializer."""
        with torch.no_grad():
            for block in self.transformer.h:
                if self.config.zero_init_mlp:
                    block.attn.c_proj.weight.zero_()
                    block.mlp.c_proj.weight.zero_()

                if hasattr(block.attn, "v_extra_gate"):
                    block.attn.v_extra_gate.bias.fill_(
                        float(getattr(self.config, "v_extra_gate_init_bias", -2.0))
                    )

                if hasattr(block.attn, "sparse_attn_gate"):
                    block.attn.sparse_attn_gate.bias.fill_(
                        float(getattr(self.config, "attention_value_gate_init_bias", 2.0))
                    )

    def _init_weights(self, module):
        """
        Initialize the weights of the model by sampling from a normal distribution.
        What we do here exactly is:
        - For linear layers, we sample from a normal distribution with mean 0 and std 0.02.
        - For embeddings, we sample from a normal distribution with mean 0 and std 0.02.
        - For bias terms, we set them to zero.

        This is a standard initialization scheme for transformer models.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, emb, next_mod_emb, time_emb, mask=None):

        next_mod_emb = self.mlp_for_mod_emb(next_mod_emb)
        time_emb = self.mlp_for_time_emb(time_emb)

        # Keep both query branches before the transformer and average their contributions.
        emb = emb + ((time_emb + next_mod_emb) / 2.0)

        if self.config.add_rms_norm:
            emb = F.rms_norm(emb, (emb.size(-1),))

        # Save the “input embedding to the stack”
        emb0 = emb

        for i, block in enumerate(self.transformer.h):
            emb = block(emb, mask)

            # add embedding skip into every block output
            if self.embed_skip_scale is not None:
                emb = emb + self.embed_skip_scale[i] * emb0


        # forward the final layernorm and the classifier
        emb = self.transformer.ln_f(emb)

        logits = self.lm_head(emb)  # (B, T, vocab_size)
        logits = self.tanh_norm * torch.tanh(logits / self.tanh_norm)  # Gemma 2-style logit soft-capping.

        return logits

def sinusoidal_encoding(length, width):
    """Build the shared float sine/cosine table without changing caller dimensions."""
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(position * frequencies)
    encoding[:, 1::2] = torch.cos(position * frequencies)
    return encoding

class ContinuousValueEncoder(nn.Module):
    """
    Creates sinusoidal embeddings from continuous values min-max normalized,
    clipped to [0, 1], and then scaled to [0, 1000] by the caller.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Pre-compute the frequency term (div_term) for the sinusoidal functions.
        # This is constant and does not need to be recalculated.
        half_dim = embed_dim // 2
        div_term = torch.exp(torch.arange(0, half_dim, 1).float() * (-math.log(10000.0) / (half_dim - 1)))
        self.register_buffer('div_term', div_term)

    def forward(self, normalized_continuous_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            normalized_continuous_values (torch.Tensor):
                Values normalized and clipped to [0, 1], then scaled to [0, 1000].
                Shape: [batch_size, seq_len]

        Returns:
            torch.Tensor: The sinusoidal embeddings for the continuous values.
                          Shape: [batch_size, seq_len, embed_dim]
        """
        # Add a new dimension for the embedding features
        pos = normalized_continuous_values.unsqueeze(-1)  # Shape: [B, T, 1]

        # Create the positional encoding tensor
        pe = torch.zeros(pos.shape[0], pos.shape[1], self.embed_dim, device=pos.device)

        # Apply sin to even indices and cos to odd indices
        pe[:, :, 0::2] = torch.sin(pos * self.div_term)
        pe[:, :, 1::2] = torch.cos(pos * self.div_term)

        return pe

def _stable_softmax_over_token_ids(logits: torch.Tensor, token_ids) -> torch.Tensor:
    """Return a stable softmax restricted to absolute vocabulary token IDs."""
    token_ids_tensor = torch.as_tensor(token_ids, dtype=torch.long, device=logits.device)
    if token_ids_tensor.numel() == 0:
        return logits.new_empty((*logits.shape[:-1], 0))

    selected_logits = logits.index_select(-1, token_ids_tensor)
    selected_max = torch.amax(selected_logits, dim=-1, keepdim=True)
    exp_values = (selected_logits - selected_max).exp()
    denominator = exp_values.sum(dim=-1, keepdim=True).clamp_min(1e-15)
    return exp_values / denominator

class MultiModalTransformer(nn.Module):
    """Multimodal decoder with value, modality, and temporal representations."""

    def __init__(self, num_modalities, vocab_size, mask_modality_val, temporal_vocab_size,
                 config, vals_per_modality, modality_name_to_number, cum_base_value_per_modality,
                 cat_cols, mod2bininfo, mod2stats, continuous_min_max_stats):
        """Construct multimodal representations and the configured decoder."""
        super().__init__()
        self.temporal_modeling = config.temporal_modeling
        self.modality_token = config.modality_token
        self.temporal_positional_encodings = config.temporal_positional_encodings
        self.config = config
        self.num_modalities = num_modalities
        self.mask_modality_val = mask_modality_val

        self.vals_per_modality = vals_per_modality
        self.modality_name_to_number = modality_name_to_number
        self.cum_base_value_per_modality = cum_base_value_per_modality
        self.cat_cols = cat_cols
        self.mod2bininfo = mod2bininfo
        self.mod2stats = mod2stats
        self.continuous_min_max_stats = continuous_min_max_stats

        embed_dim = config.n_embd
        if self.temporal_positional_encodings:
            embed_dim -= 12

        self.continuous_encoder = ContinuousValueEncoder(embed_dim)

        old_vocab_size = vocab_size
        vocab_size = ((vocab_size + 1) // 128 + 1) * 128  # Include the reserved padding index.

        self.vocab_size = vocab_size
        self.old_vocab_size = old_vocab_size

        if self.config.use_learned_token_embds:
            self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=self.old_vocab_size + 1)

        self.embed_dim = embed_dim


        self.positional_encoding = sinusoidal_encoding(max(config.max_seq_length, 30_000), embed_dim)

        if self.temporal_positional_encodings:
            embed_dim += 12

        self.embedding_dim = embed_dim
        if self.config.modality_token:
            self.modality_embedding = nn.Embedding(num_modalities + 2, embed_dim, padding_idx=self.num_modalities)

        if self.config.temporal_modeling:
            self.minute_embedding = nn.Embedding(temporal_vocab_size[2], embed_dim)
            self.hour_embedding = nn.Embedding(temporal_vocab_size[1], embed_dim)
            self.month_embedding = nn.Embedding(temporal_vocab_size[3], embed_dim)
            self.day_of_week_embedding = nn.Embedding(temporal_vocab_size[0], embed_dim)
            self.year_embedding = nn.Embedding(temporal_vocab_size[4], embed_dim)
            self.day_of_month_embedding = nn.Embedding(temporal_vocab_size[5], embed_dim)
            self.sleep_embedding = nn.Embedding(temporal_vocab_size[6], embed_dim)


        if self.config.sinusoidal_time_embeddings:
            self.minute_positional_encoding = sinusoidal_encoding(
                temporal_vocab_size[2], embed_dim // 8,
            )
            self.hour_positional_encoding = sinusoidal_encoding(
                temporal_vocab_size[1], embed_dim // 8,
            )
            self.month_positional_encoding = sinusoidal_encoding(
                temporal_vocab_size[3], embed_dim // 8,
            )
            self.day_of_week_positional_encoding = sinusoidal_encoding(
                temporal_vocab_size[0], embed_dim // 8,
            )
            self.year_positional_encoding = sinusoidal_encoding(
                temporal_vocab_size[4], embed_dim // 8,
            )
            self.day_of_month_positional_encoding = sinusoidal_encoding(
                temporal_vocab_size[5], embed_dim // 8,
            )
            self.sleep_positional_encoding = sinusoidal_encoding(
                temporal_vocab_size[6], embed_dim // 8,
            )

            # Learned time-encoding offset; retain the ambd checkpoint key.
            self.ambd = nn.Parameter(torch.randn(embed_dim))

        # Initialize layers for Age and Gender Positional Encodings
        self.add_age_gender_as_pe = self.config.add_age_gender_as_pe

        if self.add_age_gender_as_pe:
            self.age_positional_encoding = sinusoidal_encoding(365 * 200, embed_dim)
            self.gender_positional_encoding = sinusoidal_encoding(2, embed_dim)

            self.age_learned = nn.Parameter(torch.randn(embed_dim))
            self.gender_learned = nn.Parameter(torch.randn(embed_dim))

        gpt_config = SimpleNamespace(
            vocab_size=vocab_size, n_layer=config.n_layers, n_head=config.n_heads,
            n_embd=embed_dim, dropout=config.dropout,
            tanh_norm=config.tanh_norm,
            complex_mod_time_emb=config.complex_mod_time_emb,
            dim_feedforward_scaling=config.dim_feedforward_scaling,
            embd_scaling=config.embd_scaling,
            add_rms_norm=config.add_rms_norm,
            zero_init_mlp=config.zero_init_mlp,
            n_value_extras=config.n_value_extras,
            v_extra_gate_init_bias=config.v_extra_gate_init_bias,
            attention_value_gate_init_bias=config.attention_value_gate_init_bias,
            embed_skip_init=config.embed_skip_init,
            use_value_extras=config.use_value_extras,
            use_attention_value_gate=config.use_attention_value_gate,
            embed_skip_connections=config.embed_skip_connections,
        )
        self.transformer_encoder = GPT(gpt_config)

    def temporal_positional_encoding(self, time_expanded):
        minute = time_expanded[:, :, 2]
        hour = time_expanded[:, :, 1]
        month = time_expanded[:, :, 3]
        day_of_week = time_expanded[:, :, 0]
        year = time_expanded[:, :, 4]
        day_of_month = time_expanded[:, :, 5]

        minute_enc = torch.stack([torch.sin(minute * (2 * np.pi / 4)),
                                  torch.cos(minute * (2 * np.pi / 4))], dim=-1)
        hour_enc = torch.stack([torch.sin(hour * (2 * np.pi / 24)),
                                torch.cos(hour * (2 * np.pi / 24))], dim=-1)
        month_enc = torch.stack([torch.sin(month * (2 * np.pi / 12)),
                                 torch.cos(month * (2 * np.pi / 12))], dim=-1)
        day_of_week_enc = torch.stack([torch.sin(day_of_week * (2 * np.pi / 7)),
                                       torch.cos(day_of_week * (2 * np.pi / 7))], dim=-1)
        year_enc = torch.stack([torch.sin(year * (2 * np.pi / 365)),
                                torch.cos(year * (2 * np.pi / 365))], dim=-1)
        day_of_month_enc = torch.stack([torch.sin(day_of_month * (2 * np.pi / 31)),
                                        torch.cos(day_of_month * (2 * np.pi / 31))], dim=-1)

        positional_encodings = torch.cat([minute_enc, hour_enc, month_enc, day_of_week_enc, year_enc, day_of_month_enc],
                                         dim=-1)
        return positional_encodings

    def _build_continuous_value_embedding(self, continuous_tokens, modalities, seq_len):
        """Encode continuous values only at recognized continuous positions."""
        scaled_normalized_canvas = torch.zeros(
            continuous_tokens.shape[0],
            seq_len,
            device=continuous_tokens.device,
            dtype=continuous_tokens.dtype,
        )
        continuous_position_mask = torch.zeros_like(scaled_normalized_canvas, dtype=torch.bool)

        for mod_name, stats in self.continuous_min_max_stats.items():
            mod_num = self.modality_name_to_number.get(mod_name)
            if mod_num is None:
                continue

            mod_mask = (modalities[:, :seq_len] == mod_num) & torch.isfinite(continuous_tokens[:, :seq_len])
            continuous_position_mask |= mod_mask
            if not mod_mask.any():
                continue

            min_val, max_val = stats['min'], stats['max']
            range_val = max_val - min_val
            if range_val < 1e-6:
                continue

            normalized = (continuous_tokens[:, :seq_len][mod_mask] - min_val) / range_val
            scaled_normalized_canvas[mod_mask] = torch.clamp(normalized, 0.0, 1.0) * 1000

        continuous_pe = self.continuous_encoder(scaled_normalized_canvas.float())
        position_mask = continuous_position_mask.unsqueeze(-1)
        return continuous_pe * position_mask.to(continuous_pe.dtype)

    def _build_demographic_embedding(self, tokens, modalities, time_expanded, seq_len,
                                     embedding_template, mask=None):
        """Use first demographic events only once visible to the current query."""
        batch = tokens.shape[0]
        device = tokens.device
        mods = modalities[:, :seq_len]
        times = time_expanded[:, :seq_len]
        positions = torch.arange(seq_len, device=device)[None, :]
        rows = torch.arange(batch, device=device)

        def first_visible(name):
            present = mods == self.modality_name_to_number.get(name, -1)
            first = present.long().argmax(dim=1)
            visible = present.any(dim=1)[:, None] & (positions >= first[:, None])
            if mask is not None:
                # ELIS masks can exclude earlier future targets as well as later events.
                allowed = mask[:, 0] if mask.ndim == 4 else mask
                if allowed.ndim == 2:
                    allowed = allowed[None].expand(batch, -1, -1)
                visible &= allowed.gather(2, first[:, None, None].expand(-1, seq_len, 1)).squeeze(-1)
            return first, visible

        gender_pe = torch.zeros_like(embedding_template)
        first_gender, visible_gender = first_visible('gender')
        first_fallback, visible_fallback = first_visible('gender_at_birth')
        for name, first, visible in (
                ('gender', first_gender, visible_gender),
                ('gender_at_birth', first_fallback, visible_fallback & ~visible_gender)):
            mod = self.modality_name_to_number.get(name)
            if mod is None:
                continue
            values = tokens[rows, first] - (self.cum_base_value_per_modality[mod] + 1)
            valid = (values >= 0) & (values <= 1)
            encoded = self.gender_positional_encoding.to(device)[values.clamp(0, 1).long()]
            encoded = (encoded + self.gender_learned) / 2
            gender_pe += encoded[:, None, :] * (visible & valid[:, None])[:, :, None]

        first_birth, visible_birth = first_visible('birth_event')
        birth = times[rows, first_birth]
        age_years = (times[:, :, 4].float() - birth[:, None, 4].float()
                     + (times[:, :, 3].float() - birth[:, None, 3].float()) / 12
                     + (times[:, :, 5].float() - birth[:, None, 5].float()) / 365.25)
        age_days = (age_years * 365.25).long().clamp(0, self.age_positional_encoding.shape[0] - 1)
        age_pe = self.age_positional_encoding.to(device)[age_days]
        age_pe = age_pe * visible_birth[:, :, None]
        age_pe = (age_pe + self.age_learned) / 2
        return (age_pe + gender_pe) / 2

    def _build_modality_embeddings(self, modalities, seq_len, dtype):
        """Return the modality context contribution and full shifted-query tensor."""
        context_contribution = None
        full_modality_embeddings = None

        if self.modality_token:
            learned_modality_embeddings = self.modality_embedding(modalities)
            full_modality_embeddings = learned_modality_embeddings
            context_contribution = learned_modality_embeddings[:, :seq_len]

        if full_modality_embeddings is None:
            full_modality_embeddings = torch.zeros(
                modalities.shape[0], seq_len + 1, self.embedding_dim,
                device=modalities.device, dtype=dtype,
            )
            context_contribution = full_modality_embeddings[:, :seq_len]

        return context_contribution, full_modality_embeddings

    def _build_time_embeddings(self, time_expanded, seq_len, dtype):
        """Return the time context contribution and full shifted-query tensor."""
        context_contribution = None
        full_time_embeddings = None
        year_indices = time_expanded[:, :, 4]

        if self.temporal_modeling:

            learned_time_embeddings = (
                self.minute_embedding(time_expanded[:, :, 2])
                + self.hour_embedding(time_expanded[:, :, 1])
                + self.month_embedding(time_expanded[:, :, 3])
                + self.day_of_week_embedding(time_expanded[:, :, 0])
                + self.year_embedding(year_indices)
                + self.day_of_month_embedding(time_expanded[:, :, 5])
                + self.sleep_embedding(time_expanded[:, :, 6])
            ) / 7
            full_time_embeddings = learned_time_embeddings
            context_contribution = learned_time_embeddings[:, :seq_len]

        if self.config.sinusoidal_time_embeddings:
            minute_encoding = self.minute_positional_encoding[time_expanded[:, :, 2].long()]
            hour_encoding = self.hour_positional_encoding[time_expanded[:, :, 1].long()]
            month_encoding = self.month_positional_encoding[time_expanded[:, :, 3].long()]
            day_of_week_encoding = self.day_of_week_positional_encoding[time_expanded[:, :, 0].long()]
            year_encoding = self.year_positional_encoding[year_indices.long()]
            day_of_month_encoding = self.day_of_month_positional_encoding[time_expanded[:, :, 5].long()]
            sleep_encoding = self.sleep_positional_encoding[time_expanded[:, :, 6].long()]
            empty_encoding = torch.zeros_like(sleep_encoding)
            sinusoidal_time_embeddings = torch.cat(
                [minute_encoding, hour_encoding, month_encoding, day_of_week_encoding,
                 year_encoding, day_of_month_encoding, sleep_encoding, empty_encoding],
                dim=-1,
            )
            sinusoidal_time_embeddings = sinusoidal_time_embeddings + self.ambd

            if full_time_embeddings is None:
                full_time_embeddings = sinusoidal_time_embeddings
                context_contribution = sinusoidal_time_embeddings[:, :seq_len]
            else:
                full_time_embeddings = (full_time_embeddings + sinusoidal_time_embeddings) / 2
                context_contribution = context_contribution + sinusoidal_time_embeddings[:, :seq_len]

        if full_time_embeddings is None:
            full_time_embeddings = torch.zeros(
                time_expanded.shape[0], seq_len + 1, self.embedding_dim,
                device=time_expanded.device, dtype=dtype,
            )
            context_contribution = full_time_embeddings[:, :seq_len]

        return context_contribution, full_time_embeddings

    def forward(self, tokens, continuous_tokens, modalities, time_expanded,
                mask=None):

        batch_size, seq_len = tokens.size()
        if seq_len > self.config.max_seq_length:
            raise ValueError('Model input exceeds max_seq_length.')

        if self.config.use_learned_token_embds:
            embeddings = self.embeddings(tokens)
        else:
            embeddings = torch.zeros((batch_size, seq_len, self.embed_dim), device=tokens.device)

        if self.config.continuous_value_embeddings:
            embeddings = embeddings + self._build_continuous_value_embedding(
                continuous_tokens, modalities, seq_len
            )

        if self.config.reg_positional_encodings:
            self.positional_encoding = self.positional_encoding.to(tokens.device)
            embeddings = embeddings + self.positional_encoding[:seq_len]

        if self.temporal_positional_encodings:
            temporal_positional_encodings = self.temporal_positional_encoding(time_expanded[:, :seq_len])
            temporal_positional_encodings = temporal_positional_encodings.to(tokens.device)
            embeddings = torch.cat([embeddings, temporal_positional_encodings], dim=-1)

        if self.add_age_gender_as_pe:
            embeddings = embeddings + self._build_demographic_embedding(
                tokens, modalities, time_expanded, seq_len, embeddings, mask=mask
            )

        modality_context, modality_embeddings = self._build_modality_embeddings(
            modalities, seq_len, embeddings.dtype
        )
        embeddings = embeddings + modality_context

        time_context, time_embeddings = self._build_time_embeddings(
            time_expanded, seq_len, embeddings.dtype
        )
        embeddings = embeddings + time_context

        next_modality_embeddings = modality_embeddings[:, 1:]
        next_time_embeddings = time_embeddings[:, 1:]

        # Condition each prediction on the target measurement's modality and time.

        logits = self.transformer_encoder(embeddings, next_modality_embeddings, next_time_embeddings,
                                          mask=mask)

        return logits

    def calculate_loss(self, logits, continuous_tokens, targets, modalities, mode='Train'):
        losses = {}
        if len(modalities.shape) == 2 and logits.shape[1] != modalities.shape[1]:
            modalities = modalities[:, 1:].contiguous()
            continuous_tokens = continuous_tokens[:, 1:].contiguous()
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        modalities = modalities.reshape(-1)
        continuous_tokens = continuous_tokens.reshape(-1)
        valid = modalities != self.mask_modality_val
        if not valid.any():
            return logits.float().sum() * 0.0, {}
        if self.config.soft_labels:
            loss = self.apply_soft_cross_entropy(logits, None, losses, modalities, targets)
        else:
            loss = vocabulary_cross_entropy(self, logits[valid], targets[valid], modalities[valid]).mean()
        if self.config.add_mae_loss_on_continuous_metrics:
            loss = self.handle_continuous_loss(logits, loss, losses, continuous_tokens, modalities, mode)
        return loss, losses

    def apply_soft_cross_entropy(self, logits, loss, losses, modalities, targets):
        num_classes = logits.size(-1)  # Total number of classes in logits
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)
        valid_mask = modalities.contiguous().view(-1) != self.mask_modality_val
        valid_targets = targets_flat[valid_mask]
        valid_mods = modalities.contiguous().view(-1)[valid_mask]
        # Prepare modality-specific parameters
        modality_params = {}
        for modality_name, set_of_vals_per_mod in self.vals_per_modality.items():

            sigma = self.config.SL_sigma
            if modality_name in self.cat_cols:
                sigma = 0.001
            modality = self.modality_name_to_number[modality_name]
            set_of_vals_per_mod = [i + self.cum_base_value_per_modality[modality] for i in set_of_vals_per_mod]
            mod_min, mod_max = min(set_of_vals_per_mod), max(set_of_vals_per_mod)
            modality_params[modality] = {
                'min_val': mod_min - 1,
                'max_val': mod_max + 1,
                'sigma': sigma
            }
        # Create tensors for min_val, max_val, and sigma per sample
        modality_min = torch.zeros_like(valid_targets, dtype=torch.float)
        modality_max = torch.full_like(valid_targets, num_classes - 1, dtype=torch.float)
        modality_sigma = torch.full_like(valid_targets, self.config.SL_sigma, dtype=torch.float)
        for modality, params in modality_params.items():
            mod_mask = valid_mods == modality
            modality_min[mod_mask] = params['min_val']
            modality_max[mod_mask] = params['max_val']
            modality_sigma[mod_mask] = params['sigma']
        # Compute Gaussian soft labels in a vectorized manner
        ranges = torch.arange(0, num_classes, device=valid_targets.device).float()  # Shape [num_classes]
        targets_expanded = valid_targets.unsqueeze(1).float()  # Shape [num_valid_samples, 1]
        # Compute Gaussian distributions
        gaussian_vals = torch.exp(-0.5 * ((ranges - targets_expanded) / modality_sigma.unsqueeze(1)) ** 2)
        # Apply modality-specific min and max masks
        min_mask = ranges >= modality_min.unsqueeze(1)
        max_mask = ranges <= modality_max.unsqueeze(1)
        range_mask = min_mask & max_mask
        gaussian_vals = gaussian_vals * range_mask.float()
        # Normalize soft labels
        gaussian_sums = gaussian_vals.sum(dim=1, keepdim=True)
        # Avoid division by zero
        gaussian_sums = gaussian_sums + (gaussian_sums == 0).float()
        gaussian_vals = gaussian_vals / (gaussian_sums + 1e-12)
        soft_cross_entropy = vocabulary_cross_entropy(
            self, logits_flat[valid_mask], valid_targets, valid_mods, soft_targets=gaussian_vals)

        soft_cross_entropy = soft_cross_entropy.mean()

        loss = soft_cross_entropy * self.config.soft_labels_scale
        losses['Soft Cross Entropy'] = soft_cross_entropy
        return loss

    def handle_continuous_loss(self, logits, loss, losses, continuous_tokens, modalities, mode):
        """Add scaled standardized MAE, averaged equally over observed continuous modalities."""
        logits_2d = logits
        cont_2d = continuous_tokens.contiguous()
        mods_1d = modalities.contiguous()

        all_mae_terms = []

        for mod_name, mod_num in self.modality_name_to_number.items():
            # Process only modalities with bin info.
            if mod_name not in self.mod2bininfo:
                continue
            token_indices, bin_midpoints = self.mod2bininfo[mod_name]
            if len(token_indices) == 0:
                continue
            if mod_name not in self.mod2stats:
                continue

            mod_mean = self.mod2stats[mod_name]["mean"]
            mod_std = self.mod2stats[mod_name]["std"]
            eps = 1e-8
            effective_std = max(mod_std, eps)

            # Gather positions for this modality.
            mask_mod = (mods_1d == mod_num) & torch.isfinite(cont_2d)
            if not mask_mod.any():
                continue

            sub_logits = logits_2d[mask_mod].float()  # shape: [K, vocab_size]
            sub_cont_gt = cont_2d[mask_mod]  # shape: [K]; preserve higher-precision targets.

            probs_bins = _stable_softmax_over_token_ids(sub_logits, token_indices)
            if probs_bins.shape[1] != len(bin_midpoints):
                raise ValueError(f"Bin/token count mismatch for {mod_name!r}.")

            bin_midpoints_t = torch.as_tensor(bin_midpoints, device=probs_bins.device, dtype=probs_bins.dtype)
            # FP32 elementwise reduction avoids autocast's BF16 matmul rounding.
            pred_real = (probs_bins * bin_midpoints_t).sum(dim=-1)

            # Z‑score transform.
            zscore_pred = (pred_real - mod_mean) / effective_std
            zscore_true = (sub_cont_gt - mod_mean) / effective_std

            mae_mod = torch.mean(torch.abs(zscore_pred - zscore_true))

            all_mae_terms.append(mae_mod)

        if len(all_mae_terms) > 0:
            mae_all = torch.stack(all_mae_terms).mean()
            scale = self.config.mae_loss_scale
            mae_scaled = mae_all * scale
            loss = loss + mae_scaled
            losses[f"ZscoreMAE_{mode}"] = mae_all
            losses[f"ZscoreMAE_scaled_{mode}"] = mae_scaled

        return loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """
        This function will configure the optimizer for the model. It will create two groups of parameters, one for
        the weight decay, and one for the no weight decay. It will then create an AdamW optimizer with the given
        learning rate and the two groups of parameters.
        """
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Apply weight decay to matrices and higher-dimensional parameters.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.device(device).type == 'cuda'

        return torch.optim.AdamW(params=optim_groups, lr=learning_rate,
                                 betas=(self.config.b1, self.config.b2), eps=1e-8, fused=use_fused)
