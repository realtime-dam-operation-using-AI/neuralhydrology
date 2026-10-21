"""Extreme-Adaptive Transformer (Exformer) backbone for NeuralHydrology.

This is a port of the reference implementation that accompanies "Extreme Adaptive Transformer for Time Series
Forecasting" (https://github.com/.../Exformer, files ``models/my_method/dozerformer_Linear.py``,
``models/my_method/dozerformer_EncDec.py``, ``models/my_method/build_model_util.py``,
``models/my_method/Attentions/attention_masking.py`` and ``models/REVIN.py``).

The reference code is written against the Informer/MICN-style ``(seq_len, label_len, pred_len)`` pipeline and
expects the *target series itself* to be one of the input channels. NeuralHydrology instead feeds a window of
forcings and asks for the target at the last ``predict_last_n`` steps. Three adaptations were therefore
necessary; they are the only deviations from the reference implementation:

1. **Output mapping.** The reference model predicts every input channel and then keeps channel 0 (the stream
   sensor). Here the target is not an input channel, so the per-channel predictions are mapped to the target
   variables with a NeuralHydrology head (``modelzoo/head.py``), which keeps the probabilistic heads working.
2. **Extreme labels.** The reference derives the binary "extreme" label from a channel of the input window
   (``data/data_loader.py``: ``anomaly = data_x[:, :, 1:2] > anorm_thres``, the rain sensor). Here the label is
   derived the same way from one of the dynamic inputs, selected with ``exformer_extreme_variable`` and
   thresholded with ``exformer_extreme_threshold`` (in units of the standardized input).
3. **Attention kernel.** The reference calls ``flash_sparse_attn``. The mask is reproduced exactly, but it is
   applied with ``torch.nn.functional.scaled_dot_product_attention``, so no extra dependency is required.
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


class RevIN(nn.Module):
    """Reversible instance normalization (port of ``models/REVIN.py``)."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._mean = None
        self._stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            dim2reduce = tuple(range(1, x.ndim - 1))
            self._mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self._stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self._stdev + self._mean
        raise NotImplementedError(f"Unknown RevIN mode: {mode}")


class _MovingAverage(nn.Module):
    """Moving average with replicate padding (port of ``build_model_util.moving_avg``)."""

    def __init__(self, kernel_size: int):
        super(_MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecompMulti(nn.Module):
    """Multi-kernel seasonal/trend decomposition (port of ``build_model_util.series_decomp_multi``)."""

    def __init__(self, kernel_sizes: List[int]):
        super(SeriesDecompMulti, self).__init__()
        # nn.ModuleList (the reference uses a plain list, which keeps the modules off the device)
        self.moving_avg = nn.ModuleList([_MovingAverage(kernel) for kernel in kernel_sizes])

    def forward(self, x: torch.Tensor):
        means, residuals = [], []
        for func in self.moving_avg:
            mean = func(x)
            means.append(mean)
            residuals.append(x - mean)
        return sum(residuals) / len(residuals), sum(means) / len(means)


class DimensionInvariantEmbedding(nn.Module):
    """Channel-independent value embedding (port of ``build_model_util.DI_embedding``)."""

    def __init__(self, embed_dim: int):
        super(DimensionInvariantEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(3, 1), padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, seq_length, n_channels] -> [batch, embed_dim, seq_length, n_channels]
        return torch.tanh(self.conv1(x))


class TimeSeriesSegment(nn.Module):
    """Patchify along the time axis (port of ``build_model_util.TS_Segment``)."""

    def __init__(self, seq_len: int, seg_len: int):
        super(TimeSeriesSegment, self).__init__()
        self.seg_len = seg_len
        self.seg_num = math.ceil(seq_len / seg_len)
        self.pad_len = self.seg_num * seg_len - seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, embed_dim, seq_length, n_channels] -> [batch, embed_dim, seg_num, seg_len, n_channels]
        batch_size, embed_dim, _, n_channels = x.shape
        if self.pad_len != 0:
            pad = x.new_zeros(batch_size, embed_dim, self.pad_len, n_channels)
            x = torch.cat([x, pad], dim=2)
        return x.reshape(batch_size, embed_dim, self.seg_num, self.seg_len, n_channels)

    def concat(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, embed_dim, seg_num, seg_len, n_channels] -> [batch, embed_dim, seq_length, n_channels]
        batch_size, embed_dim, seg_num, seg_len, n_channels = x.shape
        x = x.reshape(batch_size, embed_dim, seg_num * seg_len, n_channels)
        if self.pad_len != 0:
            x = x[:, :, :seg_num * seg_len - self.pad_len, :]
        return x


def build_dozer_mask(seq_len: int, local_window: int, stride: int, device: torch.device) -> torch.Tensor:
    """Local + stride component of the Dozer mask (port of ``attention_masking.build_dozer_mask``)."""
    i = torch.arange(seq_len, device=device)[:, None]
    j = torch.arange(seq_len, device=device)[None, :]
    d = (i - j).abs()

    mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)
    if local_window:
        mask |= (d <= local_window // 2)
    if stride:
        mask |= (d % (stride + 1) == 0)
    return mask


def build_extreme_adaptive_mask(labels: torch.Tensor, dozer_mask: torch.Tensor, mask_type: str) -> torch.Tensor:
    """Build the extreme-adaptive attention mask.

    Port of ``attention_masking.SparseMask.generate_mask``, restricted to the mask types that the reference
    scripts actually use.

    Parameters
    ----------
    labels : torch.Tensor
        Binary extreme labels per segment, shape ``[batch, seg_num]``.
    dozer_mask : torch.Tensor
        Local + stride mask of shape ``[seg_num, seg_num]``.
    mask_type : str
        One of ``dozer`` (local + stride only, i.e. the Dozerformer baseline), ``dozer_ext_0`` (the
        extreme-adaptive mask of the Exformer paper), ``dozer_ext_only``, ``extreme_mask`` or ``full_mask``.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``[batch, seg_num, seg_num]``, where ``True`` means "attention allowed".
    """
    batch_size, seg_num = labels.shape
    is_extreme = labels > 0  # [batch, seg_num]

    if mask_type == 'full_mask':
        return torch.ones(batch_size, seg_num, seg_num, dtype=torch.bool, device=labels.device)
    if mask_type == 'dozer':
        return dozer_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # content masks
    same_label = labels.unsqueeze(2).eq(labels.unsqueeze(1))  # ExtremeMask
    extreme_extreme = is_extreme.unsqueeze(2) & is_extreme.unsqueeze(1)  # ExtremeOnlyMask
    dozer_b = dozer_mask.unsqueeze(0).expand(batch_size, -1, -1)

    if mask_type == 'extreme_mask':
        mask = same_label
    elif mask_type == 'dozer_ext_only':
        mask = extreme_extreme | dozer_b
    elif mask_type == 'dozer_ext_0':
        # ConditionalExtremeDozerMask: normal query rows use the dozer mask restricted to same-label keys,
        # extreme query rows attend to all extreme keys.
        normal_rows = dozer_b & same_label
        mask = torch.where(is_extreme.unsqueeze(-1), extreme_extreme, normal_rows)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    # a query must always be able to attend to itself, otherwise the softmax over an all-masked row is NaN
    eye = torch.eye(seg_num, dtype=torch.bool, device=labels.device).unsqueeze(0)
    return mask | eye


def labels_to_segments(labels: torch.Tensor, patch_size: int, patch_thres: int, seg_num: int) -> torch.Tensor:
    """Majority vote of the per-timestep extreme labels within each patch.

    Port of ``dozerformer_EncDec.labels_to_segments``, extended to pad the sequence when ``seq_length`` is not a
    multiple of ``patch_size`` (the reference assumes it always is).
    """
    batch_size, seq_len = labels.shape
    pad_len = seg_num * patch_size - seq_len
    if pad_len > 0:
        labels = torch.cat([labels, labels.new_zeros(batch_size, pad_len)], dim=1)
    counts = labels.reshape(batch_size, seg_num, patch_size).sum(dim=-1)
    return (counts >= patch_thres).to(labels.dtype)


class ExtremeAdaptiveAttention(nn.Module):
    """Sparse attention with the Local + Stride + Extreme mask of the Exformer paper.

    The reference implementation (``Attentions/flash_dozer.py``) builds exactly this mask and hands it to
    ``flash_sparse_attn``. Here it is applied as a masked softmax, which needs no extra dependency and produces
    the same attention distribution.
    """

    def __init__(self, local_window: int, stride: int, mask_type: str, dropout: float):
        super(ExtremeAdaptiveAttention, self).__init__()
        self.local_window = local_window
        self.stride = stride
        self.mask_type = mask_type
        self.dropout_p = dropout
        self._dozer_mask = None

    def _get_dozer_mask(self, seg_num: int, device: torch.device) -> torch.Tensor:
        if self._dozer_mask is None or self._dozer_mask.shape[0] != seg_num or self._dozer_mask.device != device:
            self._dozer_mask = build_dozer_mask(seg_num, self.local_window, self.stride, device)
        return self._dozer_mask

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        # queries/keys/values: [batch, seg_num, n_heads, head_dim], labels: [batch, seg_num]
        _, seg_num, _, _ = queries.shape

        mask = build_extreme_adaptive_mask(labels, self._get_dozer_mask(seg_num, queries.device), self.mask_type)
        mask = mask.unsqueeze(1)  # [batch, 1, seg_num, seg_num], True = attention allowed

        # [batch, seg_num, n_heads, head_dim] -> [batch, n_heads, seg_num, head_dim]
        out = F.scaled_dot_product_attention(queries.transpose(1, 2),
                                             keys.transpose(1, 2),
                                             values.transpose(1, 2),
                                             attn_mask=mask,
                                             dropout_p=self.dropout_p if self.training else 0.0)
        return out.transpose(1, 2).contiguous()


class ExtremeAdaptiveAttentionLayer(nn.Module):
    """Projection wrapper around :class:`ExtremeAdaptiveAttention` (port of ``DozerAttentionLayer``)."""

    def __init__(self, attention: nn.Module, d_model: int, n_heads: int):
        super(ExtremeAdaptiveAttentionLayer, self).__init__()
        d_keys = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, seg_num, _ = x.shape
        queries = self.query_projection(x).view(batch_size, seg_num, self.n_heads, -1)
        keys = self.key_projection(x).view(batch_size, seg_num, self.n_heads, -1)
        values = self.value_projection(x).view(batch_size, seg_num, self.n_heads, -1)
        out = self.inner_attention(queries, keys, values, labels).view(batch_size, seg_num, -1)
        return self.out_projection(out)


class _DepthwiseConv(nn.Module):
    """Depthwise convolution inside the feed-forward block (port of ``Transformer_EncDec.DWConv``)."""

    def __init__(self, dim: int):
        super(_DepthwiseConv, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class ExformerEncoderLayer(nn.Module):
    """Encoder layer (port of ``Attentions/Transformer_EncDec.EncoderLayer``)."""

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int, dropout: float):
        super(ExformerEncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dwconv = _DepthwiseConv(d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(x, labels))
        y = x = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dwconv(y)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Exformer(BaseModel):
    """Extreme-Adaptive Transformer (Exformer).

    A channel-independent, patch-based encoder whose self-attention is restricted to a sparse mask with three
    components: a **local** band, a **stride** (periodic) component and an **extreme** component that links
    time steps carrying the same "extreme" label. The seasonal part of the input is passed through this encoder
    while the trend part is mapped linearly, and both are added before the head.

    The model is configured with the following options:

    * ``exformer_patch_size``: number of time steps per patch (attention runs over patches, not time steps).
    * ``exformer_embed_dim``: per-time-step embedding size. The attention dimension is
      ``exformer_embed_dim * exformer_patch_size`` and must be divisible by ``exformer_n_heads``.
    * ``exformer_n_heads``, ``exformer_encoder_depth``, ``exformer_d_ff``, ``exformer_dropout``.
    * ``exformer_local_window`` / ``exformer_stride``: the local and stride components of the mask.
    * ``exformer_mask``: which mask to build (``dozer_ext_0`` is the extreme-adaptive mask of the paper,
      ``dozer`` is the Dozerformer baseline without the extreme component).
    * ``exformer_patch_thres``: number of extreme time steps needed to mark a whole patch as extreme.
    * ``exformer_extreme_variable`` / ``exformer_extreme_threshold``: which dynamic input defines the extreme
      label and the threshold (in standardized units) above which a time step counts as extreme.
    * ``exformer_moving_avg``: kernel sizes of the seasonal/trend decomposition.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    module_parts = ['embedding_net', 'value_embedding', 'encoder', 'head']

    def __init__(self, cfg: Config):
        super(Exformer, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)
        self._n_channels = self.embedding_net.output_size
        self._seq_len = cfg.seq_length
        self._pred_len = cfg.predict_last_n
        if not isinstance(self._pred_len, int):
            raise ValueError("Exformer does not support multiple frequencies.")

        patch_size = cfg.exformer_patch_size
        embed_dim = cfg.exformer_embed_dim
        d_model = embed_dim * patch_size
        if d_model % cfg.exformer_n_heads != 0:
            raise ValueError(f"exformer_embed_dim * exformer_patch_size ({d_model}) must be divisible by "
                             f"exformer_n_heads ({cfg.exformer_n_heads}).")

        # which dynamic input carries the extreme signal
        self._extreme_variable = cfg.exformer_extreme_variable
        self._extreme_threshold = cfg.exformer_extreme_threshold
        self._patch_thres = cfg.exformer_patch_thres
        self._patch_size = patch_size

        self.revin = RevIN(self._n_channels)
        self.decomposition = SeriesDecompMulti(cfg.exformer_moving_avg)

        self.value_embedding = DimensionInvariantEmbedding(embed_dim)
        self.segment = TimeSeriesSegment(self._seq_len, patch_size)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, embed_dim, self.segment.seg_num, patch_size, self._n_channels))
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList([
            ExformerEncoderLayer(attention=ExtremeAdaptiveAttentionLayer(ExtremeAdaptiveAttention(
                local_window=cfg.exformer_local_window,
                stride=cfg.exformer_stride,
                mask_type=cfg.exformer_mask,
                dropout=cfg.exformer_dropout),
                                                                         d_model=d_model,
                                                                         n_heads=cfg.exformer_n_heads),
                                 d_model=d_model,
                                 d_ff=cfg.exformer_d_ff * patch_size,
                                 dropout=cfg.exformer_dropout) for _ in range(cfg.exformer_encoder_depth)
        ])

        # seasonal and trend projections from the input window onto the prediction window
        self.seasonal_temporal = nn.Linear(self._seq_len, self._pred_len)
        self.seasonal_channel = nn.Conv2d(in_channels=embed_dim, out_channels=1, kernel_size=(1, 1))
        self.trend_temporal = nn.Linear(self._seq_len, self._pred_len)

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=self._n_channels, n_out=self.output_size)

    def _extreme_labels(self, data: dict) -> torch.Tensor:
        """Binary extreme label per time step, derived from one of the (standardized) dynamic inputs."""
        x_d = data['x_d']
        if self._extreme_variable not in x_d:
            raise ValueError(f"exformer_extreme_variable '{self._extreme_variable}' is not a dynamic input. "
                             f"Available: {sorted(x_d.keys())}")
        signal = x_d[self._extreme_variable]  # [batch, seq_length, 1]
        signal = torch.nan_to_num(signal, nan=0.0)
        return (signal[:, :, 0] > self._extreme_threshold).to(signal.dtype)

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Exformer model.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, predict_last_n, number of target variables].
                - `extreme_fraction`: fraction of patches that were marked as extreme, for diagnostics.
        """
        # [seq_length, batch, n_channels] -> [batch, seq_length, n_channels]
        x = self.embedding_net(data).transpose(0, 1)
        batch_size = x.shape[0]

        labels = self._extreme_labels(data)
        segment_labels = labels_to_segments(labels, self._patch_size, self._patch_thres, self.segment.seg_num)

        x = self.revin(x, 'norm')
        seasonal, trend = self.decomposition(x)

        # --- seasonal branch: patch embedding + extreme-adaptive attention ---
        embeddings = self.value_embedding(seasonal.unsqueeze(1))  # [batch, embed_dim, seq_length, n_channels]
        patches = self.segment(embeddings)  # [batch, embed_dim, seg_num, seg_len, n_channels]
        identity = patches
        patches = patches + self.positional_embedding

        # [batch, embed_dim, seg_num, seg_len, n_channels] -> [(batch n_channels), seg_num, (seg_len embed_dim)]
        patches = patches.permute(0, 4, 2, 3, 1).reshape(batch_size * self._n_channels, self.segment.seg_num, -1)
        patches = self.pre_norm(patches)

        # the same window-level labels apply to every channel of that window
        channel_labels = segment_labels.repeat_interleave(self._n_channels, dim=0)
        for layer in self.encoder:
            patches = layer(patches, channel_labels)
        patches = self.post_norm(patches)

        # back to [batch, embed_dim, seg_num, seg_len, n_channels]
        patches = patches.reshape(batch_size, self._n_channels, self.segment.seg_num, self._patch_size,
                                  -1).permute(0, 4, 2, 3, 1)
        encoder_output = patches + identity

        encoder_output = self.segment.concat(encoder_output)  # [batch, embed_dim, seq_length, n_channels]
        encoder_output = encoder_output.permute(0, 1, 3, 2)  # [batch, embed_dim, n_channels, seq_length]
        seasonal_predict = self.seasonal_temporal(encoder_output)  # [batch, embed_dim, n_channels, pred_len]
        seasonal_predict = self.seasonal_channel(seasonal_predict)  # [batch, 1, n_channels, pred_len]
        seasonal_predict = seasonal_predict.squeeze(1).transpose(1, 2)  # [batch, pred_len, n_channels]

        # --- trend branch: linear projection ---
        trend_predict = self.trend_temporal(trend.transpose(1, 2)).transpose(1, 2)

        prediction = self.revin(seasonal_predict + trend_predict, 'denorm')

        pred = self.head(self.dropout(prediction))
        pred['extreme_fraction'] = segment_labels.mean()
        return pred
