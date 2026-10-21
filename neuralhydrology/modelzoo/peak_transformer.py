"""Non-stationary ("PeakFocus") Transformer backbone for NeuralHydrology.

Port of ``models/peak_Transformer.py`` from the PeakFocus repository
(https://github.com/.../PeakFocus), together with the layers it depends on (``layers/Embed.py``,
``layers/SelfAttention_Family.py``, ``layers/Transformer_EncDec.py``).

The reference model is a Non-stationary Transformer: the input window is instance-normalized, two small
projectors learn de-stationary factors ``tau`` (a positive scale) and ``delta`` (an additive shift) from the
*raw* window, the attention logits are rescaled with them, and the decoder output is finally mapped back to the
original scale with the window statistics.

Two adaptations were necessary for the NeuralHydrology setting, where the target is not one of the input
channels:

1. **De-normalization.** The reference multiplies the decoder output by the standard deviation of input channel
   0 (the target's own history). That channel does not exist here, so the reference's ``with_shift='ws'`` path
   is used instead: a :class:`ShiftingModule` learns a scale and a shift *from the window statistics*, which is
   the same information the reference de-normalization uses. Set ``peak_transformer_with_shift: none`` to drop
   the output rescaling altogether.
2. **Decoder input.** The reference builds the decoder input as ``[x_enc[-label_len:], zeros(pred_len)]``. That
   construction is target-free and is kept verbatim; the ``label_len`` part is the tail of the forcing window.

Note also that in the reference, ``hour_day='h'`` computes hour-wise de-stationary factors
(``tau_learner_hour`` / ``delta_learner_hour``) but never passes them to the attention -- only the hour-wise
instance normalization takes effect. This port reproduces the effective behaviour and does not build the unused
projectors.
"""

import logging
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class TokenEmbedding(nn.Module):
    """Circular 1d convolution over the feature axis (port of ``layers/Embed.py``)."""

    def __init__(self, c_in: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=1,
                                   padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding (port of ``layers/Embed.py``)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    """Value + positional embedding.

    The reference also adds a temporal embedding built from calendar features. NeuralHydrology does not pass
    calendar features to the model, so this port always takes the ``x_mark is None`` branch of the reference
    implementation (value + position only).
    """

    def __init__(self, c_in: int, d_model: int, dropout: float):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class DSAttention(nn.Module):
    """De-stationary attention (port of ``layers/SelfAttention_Family.DSAttention``)."""

    def __init__(self, mask_flag: bool = True, attention_dropout: float = 0.1):
        super(DSAttention, self).__init__()
        self.mask_flag = mask_flag
        self.dropout_p = attention_dropout

    def forward(self, queries, keys, values, tau=None, delta=None):
        """De-stationary attention, computed as ``softmax(scale * (tau * q.k + delta))``.

        The reference materializes the full score matrix. The same result is obtained by folding ``tau`` into
        the queries and passing ``scale * delta`` as an additive attention mask, which lets PyTorch use its
        memory-efficient attention kernels -- important here, because the input sequence is 720 steps long.
        """
        head_dim = queries.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)

        # [batch, seq_length, n_heads, head_dim] -> [batch, n_heads, seq_length, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if tau is not None:
            queries = queries * tau.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, 1]

        attn_mask = None
        if delta is not None:
            attn_mask = scale * delta.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_length]
            if self.mask_flag:
                causal = torch.triu(torch.ones(queries.shape[2], keys.shape[2], dtype=torch.bool,
                                               device=queries.device),
                                    diagonal=1)
                attn_mask = attn_mask.masked_fill(causal, -float('inf'))

        out = F.scaled_dot_product_attention(queries,
                                             keys,
                                             values,
                                             attn_mask=attn_mask,
                                             dropout_p=self.dropout_p if self.training else 0.0,
                                             is_causal=self.mask_flag and attn_mask is None,
                                             scale=scale)
        return out.transpose(1, 2).contiguous()


class AttentionLayer(nn.Module):
    """Projection wrapper around an attention module (port of ``layers/SelfAttention_Family.AttentionLayer``)."""

    def __init__(self, attention: nn.Module, d_model: int, n_heads: int):
        super(AttentionLayer, self).__init__()
        d_keys = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, tau=None, delta=None):
        batch_size, len_q, _ = queries.shape
        len_k = keys.shape[1]
        queries = self.query_projection(queries).view(batch_size, len_q, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch_size, len_k, self.n_heads, -1)
        values = self.value_projection(values).view(batch_size, len_k, self.n_heads, -1)
        out = self.inner_attention(queries, keys, values, tau=tau, delta=delta)
        return self.out_projection(out.view(batch_size, len_q, -1))


class EncoderLayer(nn.Module):
    """Transformer encoder layer (port of ``layers/Transformer_EncDec.EncoderLayer``)."""

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tau=None, delta=None):
        x = x + self.dropout(self.attention(x, x, x, tau=tau, delta=delta))
        y = x = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class DecoderLayer(nn.Module):
    """Transformer decoder layer (port of ``layers/Transformer_EncDec.DecoderLayer``)."""

    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, d_model: int, d_ff: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(x, x, x, tau=tau, delta=None))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, tau=tau, delta=delta))
        y = x = self.norm2(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Projector(nn.Module):
    """MLP that learns a de-stationary factor from the raw window and its statistics.

    Port of ``models/peak_Transformer.Projector``.
    """

    def __init__(self, enc_in: int, seq_len: int, hidden_dims: List[int], hidden_layers: int, output_dim: int):
        super(Projector, self).__init__()
        # the convolution runs over the feature axis; reflect padding needs at least two features, so fall back to
        # replicate padding for single-variable inputs (e.g., only basin-average precipitation).
        self.series_conv = nn.Conv1d(in_channels=seq_len,
                                     out_channels=1,
                                     kernel_size=3,
                                     padding=1,
                                     padding_mode='reflect' if enc_in > 1 else 'replicate',
                                     bias=False)
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_length, n_channels], stats: [batch, 1, n_channels]
        batch_size = x.shape[0]
        x = self.series_conv(x)  # [batch, 1, n_channels]
        x = torch.cat([x, stats], dim=1).view(batch_size, -1)  # [batch, 2 * n_channels]
        return self.backbone(x)


class ShiftingModule(nn.Module):
    """Learn an output scale and shift from the statistics of the input window.

    Port of ``models/peak_Transformer.ShiftingModule``, simplified to the ``seq_len >= pred_len`` branch and
    generalized to an arbitrary number of target variables.
    """

    def __init__(self, n_channels: int, n_segments: int, hidden_dim: int, n_targets: int):
        super(ShiftingModule, self).__init__()
        self.n_segments = n_segments
        self.backbone = nn.Sequential(nn.Linear((2 + 2 * n_segments) * n_channels, hidden_dim, bias=False),
                                      nn.Linear(hidden_dim, 2 * n_targets, bias=False))
        self._n_targets = n_targets

    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_length, n_channels] (raw), mean/std: [batch, 1, n_channels]
        batch_size, seq_len, n_channels = x.shape
        segment_len = seq_len // self.n_segments
        # statistics of the most recent segments first (the reference flips the window before chunking)
        segments = torch.flip(x, dims=[1])[:, :self.n_segments * segment_len, :]
        segments = segments.reshape(batch_size, self.n_segments, segment_len, n_channels)
        hist_mean = segments.mean(dim=2).detach()
        hist_std = torch.sqrt(torch.var(segments, dim=2, unbiased=False) + 1e-5).detach()

        features = torch.cat([hist_mean, hist_std, mean, std], dim=1).view(batch_size, -1)
        out = self.backbone(features)  # [batch, 2 * n_targets]
        return out.view(batch_size, 2, self._n_targets)


class PeakTransformer(BaseModel):
    """Non-stationary Transformer with an optional peak (block-maximum) decoder.

    The model configuration is specified in the config file using the following options:

    * ``peak_transformer_d_model``, ``peak_transformer_n_heads``, ``peak_transformer_d_ff``,
      ``peak_transformer_e_layers``, ``peak_transformer_d_layers``, ``peak_transformer_dropout``: the usual
      encoder/decoder dimensions.
    * ``peak_transformer_label_len``: how many steps of the input window are used as decoder "start token".
    * ``peak_transformer_hour_day``: ``n`` for a single instance normalization over the window (the standard
      Non-stationary Transformer, with de-stationary ``tau``/``delta``), or ``h`` for hour-of-day-wise
      normalization (requires ``seq_length`` and ``label_len + predict_last_n`` to be multiples of 24; the
      de-stationary factors are disabled in this mode, as in the reference implementation).
    * ``peak_transformer_with_shift``: ``ws`` to learn the output scale/shift from the window statistics,
      ``none`` to skip the output rescaling. ``peak_transformer_shift_segments`` sets how many chunks of the
      input window the statistics are computed over.
    * ``peak_transformer_p_hidden_dims`` / ``peak_transformer_p_hidden_layers``: size of the de-stationary
      projectors.
    * ``peak_transformer_peak_decoder``: if True, additionally return the 24-step block maximum of the
      prediction as ``y_hat_peak`` (the reference's ``busy_decoder``). Requires ``predict_last_n`` to be a
      multiple of 24.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    module_parts = ['embedding_net', 'enc_embedding', 'encoder', 'dec_embedding', 'decoder', 'head']

    def __init__(self, cfg: Config):
        super(PeakTransformer, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)
        self._n_channels = self.embedding_net.output_size
        self._seq_len = cfg.seq_length
        self._pred_len = cfg.predict_last_n
        if not isinstance(self._pred_len, int):
            raise ValueError("PeakTransformer does not support multiple frequencies.")
        self._label_len = cfg.peak_transformer_label_len
        self._hour_day = cfg.peak_transformer_hour_day.lower()
        self._with_shift = cfg.peak_transformer_with_shift.lower()
        self._peak_decoder = cfg.peak_transformer_peak_decoder

        if self._hour_day == 'h':
            if self._seq_len % 24 != 0:
                raise ValueError("peak_transformer_hour_day: 'h' requires seq_length to be a multiple of 24.")
            if (self._label_len + self._pred_len) % 24 != 0:
                raise ValueError("peak_transformer_hour_day: 'h' requires peak_transformer_label_len + "
                                 "predict_last_n to be a multiple of 24.")
        elif self._hour_day != 'n':
            raise ValueError(f"Unknown peak_transformer_hour_day: {self._hour_day}. Use 'n' or 'h'.")

        if self._peak_decoder and self._pred_len % 24 != 0:
            raise ValueError("peak_transformer_peak_decoder requires predict_last_n to be a multiple of 24.")

        d_model = cfg.peak_transformer_d_model
        n_heads = cfg.peak_transformer_n_heads
        dropout = cfg.peak_transformer_dropout

        self.enc_embedding = DataEmbedding(self._n_channels, d_model, dropout)
        self.dec_embedding = DataEmbedding(self._n_channels, d_model, dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(AttentionLayer(DSAttention(mask_flag=False, attention_dropout=dropout), d_model, n_heads),
                         d_model=d_model,
                         d_ff=cfg.peak_transformer_d_ff,
                         dropout=dropout) for _ in range(cfg.peak_transformer_e_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder = nn.ModuleList([
            DecoderLayer(AttentionLayer(DSAttention(mask_flag=True, attention_dropout=dropout), d_model, n_heads),
                         AttentionLayer(DSAttention(mask_flag=False, attention_dropout=dropout), d_model, n_heads),
                         d_model=d_model,
                         d_ff=cfg.peak_transformer_d_ff,
                         dropout=dropout) for _ in range(cfg.peak_transformer_d_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        if self._hour_day == 'n':
            self.tau_learner = Projector(enc_in=self._n_channels,
                                         seq_len=self._seq_len,
                                         hidden_dims=cfg.peak_transformer_p_hidden_dims,
                                         hidden_layers=cfg.peak_transformer_p_hidden_layers,
                                         output_dim=1)
            self.delta_learner = Projector(enc_in=self._n_channels,
                                           seq_len=self._seq_len,
                                           hidden_dims=cfg.peak_transformer_p_hidden_dims,
                                           hidden_layers=cfg.peak_transformer_p_hidden_layers,
                                           output_dim=self._seq_len)

        if self._with_shift == 'ws':
            if cfg.head.lower() != 'regression':
                raise ValueError("peak_transformer_with_shift: 'ws' rescales `y_hat` and therefore only works "
                                 "with head: regression. Use peak_transformer_with_shift: none instead.")
            self.shift_learner = ShiftingModule(n_channels=self._n_channels,
                                                n_segments=cfg.peak_transformer_shift_segments,
                                                hidden_dim=cfg.peak_transformer_p_hidden_dims[0],
                                                n_targets=self.output_size)
        elif self._with_shift != 'none':
            raise ValueError(f"Unknown peak_transformer_with_shift: {self._with_shift}. Use 'ws' or 'none'.")

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=d_model, n_out=self.output_size)

    def _normalize(self, x: torch.Tensor):
        """Instance-normalize the input window and return the statistics needed for the de-stationary factors."""
        if self._hour_day == 'h':
            batch_size, seq_len, n_channels = x.shape
            blocks = x.reshape(batch_size, seq_len // 24, 24, n_channels)
            # statistics per hour-of-day, across days
            mean = blocks.mean(1, keepdim=True).detach()
            std = torch.sqrt(torch.var(blocks, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            normalized = ((blocks - mean) / std).reshape(batch_size, seq_len, n_channels)
            return normalized, None, None
        mean = x.mean(1, keepdim=True).detach()
        std = torch.sqrt(torch.var(x - mean, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return (x - mean) / std, mean, std

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Non-stationary (PeakFocus) Transformer.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, predict_last_n, number of target variables].
                - `y_hat_peak`: 24-step block maxima of `y_hat`, if ``peak_transformer_peak_decoder`` is True.
        """
        # [seq_length, batch, n_channels] -> [batch, seq_length, n_channels]
        x_enc = self.embedding_net(data).transpose(0, 1)
        x_raw = x_enc.clone().detach()

        x_enc, mean, std = self._normalize(x_enc)

        tau, delta = None, None
        if self._hour_day == 'n':
            tau = self.tau_learner(x_raw, std).exp()
            delta = self.delta_learner(x_raw, mean)

        # decoder input: tail of the (normalized) forcing window + zeros over the prediction window
        x_dec = torch.cat([x_enc[:, -self._label_len:, :], torch.zeros_like(x_enc[:, :self._pred_len, :])], dim=1)

        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out, tau=tau, delta=delta)
        enc_out = self.encoder_norm(enc_out)

        dec_out = self.dec_embedding(x_dec)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, tau=tau, delta=delta)
        dec_out = self.decoder_norm(dec_out)

        pred = self.head(self.dropout(dec_out[:, -self._pred_len:, :]))

        if self._with_shift == 'ws':
            raw_mean = x_raw.mean(1, keepdim=True)
            raw_std = torch.sqrt(torch.var(x_raw, dim=1, keepdim=True, unbiased=False) + 1e-5)
            shift = self.shift_learner(x_raw, raw_mean, raw_std)
            scale = shift[:, 0:1, :].exp()
            offset = shift[:, 1:2, :]
            pred['y_hat'] = pred['y_hat'] * scale + offset

        if self._peak_decoder:
            # block maximum over each 24-step block of the prediction window (the reference's `busy_decoder`)
            peak, index = F.max_pool1d(pred['y_hat'].transpose(1, 2), kernel_size=24, return_indices=True)
            pred['y_hat_peak'] = peak.transpose(1, 2)
            pred['y_hat_peak_index'] = index.transpose(1, 2)

        return pred
