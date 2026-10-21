"""PeakFocus backbone for NeuralHydrology.

Port of ``models/proposed_model.py`` from the PeakFocus repository (https://github.com/.../PeakFocus), the
model proposed in "PeakFocus: A Unified Multi-Scale Framework Focused on Peak Localization and Intensity
Regression".

The architecture has three parts:

* an **encoder** -- value/position embedding followed by a stack of residual MLP blocks and a linear projection
  from the input window onto the prediction window;
* **MSM-PL** (Multi-Scale Mixing Peak Locator) -- a small convolutional pyramid that produces, per predicted
  time step, a *peak logit* and a peak feature vector;
* **LAD** (Location-Aware Decoder) -- a cross-attention decoder whose keys and values are
  ``tanh(peak_features) * sigmoid(encoder_output)``, so that the value regression is conditioned on where the
  peaks are.

Two adaptations were necessary for the NeuralHydrology setting:

1. **De-normalization.** The reference instance-normalizes the input window and rescales the output with the
   statistics of input channel 0 (the target's own history). The target is not an input channel here, so the
   instance normalization is optional (``peakfocus_instance_norm``, default False) and the output is left in
   the units that NeuralHydrology's own scaler defines. The prediction is produced by a NeuralHydrology head (``modelzoo/head.py``) instead of the
   reference's ``nn.Linear(d_model, c_out)``, which keeps the probabilistic heads working.
2. **Decoder input / temporal embedding.** The reference feeds future calendar features to the decoder and to
   MSM-PL. NeuralHydrology does not pass calendar features to models, so the decoder is fed the forcings of the
   prediction window (which are available in this hindcast setting) and MSM-PL's temporal embedding is dropped.

The peak logits are returned as ``peak_logits``. They are only trained if the ``peak_focal`` and/or
``peak_weighted_mse`` regularization terms are listed in ``regularization`` -- see
``neuralhydrology/training/regularization.py``.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.peak_transformer import AttentionLayer, DataEmbedding, DSAttention
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class ResidualMLP(nn.Module):
    """Two-layer MLP with a residual link (port of ``proposed_model.MultiLayerPerceptron``)."""

    def __init__(self, dim: int, dropout: float):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.drop(F.relu(self.fc1(x))))


class MultiScaleMixingPeakLocator(nn.Module):
    """MSM-PL: multi-scale peak localization head.

    Port of ``proposed_model.Multi_Scale_Mixing_Peak_Locator`` without the (unavailable) temporal embedding.

    Parameters
    ----------
    d_model : int
        Feature dimension.
    n_out : int
        Number of peak logits per time step (one per target variable).
    dropout : float
        Dropout applied to the peak features before the projection.
    n_scales : int
        Number of pyramid levels. 0 disables the pyramid (linear projection only), 1 keeps only the original
        resolution, 2 is the setting of the paper, 3 adds a third level.
    """

    def __init__(self, d_model: int, n_out: int, dropout: float, n_scales: int = 2):
        super(MultiScaleMixingPeakLocator, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout_out = nn.Dropout(dropout)
        self.n_scales = n_scales

        if n_scales == 0:
            self.linear_proj = nn.Linear(d_model, d_model)
        else:
            self.process_s0 = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU())
            if n_scales >= 2:
                self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
                self.process_s1 = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU())
            if n_scales >= 3:
                self.pool2 = nn.AvgPool1d(kernel_size=5, stride=2, padding=2)
                self.process_s2 = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU())

        self.projection = nn.Linear(d_model, n_out)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)

        if self.n_scales == 0:
            features = self.linear_proj(x)
            return self.projection(self.dropout_out(features)), features

        s0 = x.permute(0, 2, 1)  # [batch, d_model, pred_len]
        if self.n_scales == 1:
            fused = self.process_s0(s0)
        elif self.n_scales == 2:
            f0 = self.process_s0(s0)
            f1 = self.process_s1(self.pool1(s0))
            fused = f0 + F.interpolate(f1, size=f0.shape[2], mode='linear', align_corners=True)
        else:
            s1 = self.pool1(s0)
            f0 = self.process_s0(s0)
            f1 = self.process_s1(s1)
            f2 = self.process_s2(self.pool2(s1))
            f1 = f1 + F.interpolate(f2, size=f1.shape[2], mode='linear', align_corners=True)
            fused = f0 + F.interpolate(f1, size=f0.shape[2], mode='linear', align_corners=True)

        features = fused.permute(0, 2, 1)
        return self.projection(self.dropout_out(features)), features


class LocationAwareDecoderLayer(nn.Module):
    """LAD layer: cross-attention gated by the peak features.

    Port of ``proposed_model.Location_Aware_Decoder_Layer``.
    """

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int, dropout: float, use_peak_guidance: bool):
        super(LocationAwareDecoderLayer, self).__init__()
        self.attention = attention
        self.use_peak_guidance = use_peak_guidance
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cross: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
        if self.use_peak_guidance:
            # Context Gate Fusion: the peak features select which encoder context the value head sees
            attn_input = torch.tanh(peak) * torch.sigmoid(cross)
        else:
            attn_input = cross

        x = x + self.dropout(self.attention(x, attn_input, attn_input))
        x = self.norm1(x)

        y = x
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class PeakFocus(BaseModel):
    """PeakFocus: joint peak localization and intensity regression.

    The model configuration is specified in the config file using the following options:

    * ``peakfocus_d_model``, ``peakfocus_n_heads``, ``peakfocus_d_ff``, ``peakfocus_d_layers``,
      ``peakfocus_dropout``: the usual encoder/decoder dimensions.
    * ``peakfocus_mlp_layers``: number of residual MLP blocks in the encoder backbone.
    * ``peakfocus_if_msm_pl``: if False, MSM-PL is replaced by a plain linear head (ablation ``w/o MSM-PL``).
    * ``peakfocus_if_lad``: if False, the decoder cross-attends to the encoder output directly instead of to
      the peak-gated context (ablation ``w/o LAD``).
    * ``peakfocus_n_scales``: number of MSM-PL pyramid levels (2 in the paper).
    * ``peakfocus_instance_norm``: if True, the input window is instance-normalized (per window and channel) as
      in the reference implementation. The reference undoes this normalization on the output with the statistics
      of the target's own history, which is not an input here, so with ``True`` the model loses the absolute
      magnitude of the forcings (a window with one large storm looks like a window with light rain). Default
      False: the globally standardized inputs of NeuralHydrology are used as they are.

    To actually train the peak head, add ``peak_focal`` (and optionally ``peak_weighted_mse``) to the
    ``regularization`` list in the config. Without them the peak head is present but unsupervised, which
    reproduces the ``value_loss_weight=1, peak_loss_weight=0`` ablation of the reference.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    module_parts = ['embedding_net', 'enc_embedding', 'encoder_backbone', 'peak_head', 'value_head', 'head']

    def __init__(self, cfg: Config):
        super(PeakFocus, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)
        self._n_channels = self.embedding_net.output_size
        self._seq_len = cfg.seq_length
        self._pred_len = cfg.predict_last_n
        if not isinstance(self._pred_len, int):
            raise ValueError("PeakFocus does not support multiple frequencies.")
        self._instance_norm = cfg.peakfocus_instance_norm

        d_model = cfg.peakfocus_d_model
        dropout = cfg.peakfocus_dropout

        self.enc_embedding = DataEmbedding(self._n_channels, d_model, dropout)
        self.encoder_backbone = nn.Sequential(*[ResidualMLP(d_model, dropout) for _ in range(cfg.peakfocus_mlp_layers)])
        self.temporal_proj = nn.Linear(self._seq_len, self._pred_len)

        self._if_msm_pl = cfg.peakfocus_if_msm_pl
        self._if_lad = cfg.peakfocus_if_lad and self._if_msm_pl
        if self._if_msm_pl:
            self.peak_head = MultiScaleMixingPeakLocator(d_model=d_model,
                                                         n_out=len(cfg.target_variables),
                                                         dropout=dropout,
                                                         n_scales=cfg.peakfocus_n_scales)
        else:
            self.peak_head = MultiScaleMixingPeakLocator(d_model=d_model,
                                                         n_out=len(cfg.target_variables),
                                                         dropout=dropout,
                                                         n_scales=0)

        self.dec_embedding = DataEmbedding(self._n_channels, d_model, dropout)
        self.value_head = nn.ModuleList([
            LocationAwareDecoderLayer(AttentionLayer(DSAttention(mask_flag=False, attention_dropout=dropout), d_model,
                                                     cfg.peakfocus_n_heads),
                                      d_model=d_model,
                                      d_ff=cfg.peakfocus_d_ff,
                                      dropout=dropout,
                                      use_peak_guidance=self._if_lad) for _ in range(cfg.peakfocus_d_layers)
        ])

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=d_model, n_out=self.output_size)

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the PeakFocus model.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, predict_last_n, number of target variables].
                - `peak_logits`: peak-localization logits of shape [batch size, predict_last_n, number of
                  target variables]. Trained by the `peak_focal` regularization term.
        """
        # [seq_length, batch, n_channels] -> [batch, seq_length, n_channels]
        x_enc = self.embedding_net(data).transpose(0, 1)

        # optional instance normalization of the input window (see class docstring)
        if self._instance_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # encoder: embedding -> residual MLPs -> projection onto the prediction window
        enc_out = self.encoder_backbone(self.enc_embedding(x_enc))
        enc_out = self.temporal_proj(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch, pred_len, d_model]

        # MSM-PL
        peak_logits, peak_features = self.peak_head(enc_out)

        # LAD: the decoder is fed the forcings of the prediction window
        dec_out = self.dec_embedding(x_enc[:, -self._pred_len:, :])
        for layer in self.value_head:
            dec_out = layer(dec_out, enc_out, peak_features)

        pred = self.head(self.dropout(dec_out))
        pred['peak_logits'] = peak_logits
        return pred
