import math
from typing import Dict

import pandas as pd
import torch

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.utils.config import Config


class BaseRegularization(torch.nn.Module):
    """Base class for regularization terms.

    Regularization terms subclass this class by implementing the `forward` method.

    Parameters
    ----------
    cfg: Config
        The run configuration.
    name: str
        The name of the regularization term.
    weight: float, optional.
        The weight of the regularization term. Default: 1.
    """

    def __init__(self, cfg: Config, name: str, weight: float = 1.0):
        super(BaseRegularization, self).__init__()
        self.cfg = cfg
        self.name = name
        self.weight = weight

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the regularization term.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Dictionary of predicted variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        ground_truth : Dict[str, torch.Tensor]
            Dictionary of ground truth variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        other_model_data : Dict[str, torch.Tensor]
            Dictionary of all remaining keys-value pairs in the prediction dictionary that are not directly linked to 
            the model predictions but can be useful for regularization purposes, e.g. network internals, weights etc.
            
        Returns
        -------
        torch.Tensor
            The regularization value.
        """
        raise NotImplementedError


class TiedFrequencyMSERegularization(BaseRegularization):
    """Regularization that penalizes inconsistent predictions across frequencies.

    This regularization can only be used if at least two frequencies are predicted. For each pair of adjacent
    frequencies f and f', where f is a higher frequency than f', it aggregates the f-predictions to f' and calculates
    the mean squared deviation between f' and aggregated f.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    weight: float, optional.
        Weight of the regularization term. Default: 1.

    Raises
    ------
    ValueError
        If the run configuration only predicts one frequency.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(TiedFrequencyMSERegularization, self).__init__(cfg, name='tie_frequencies', weight=weight)
        self._frequencies = sort_frequencies(
            [f for f in cfg.use_frequencies if cfg.predict_last_n[f] > 0 and f not in cfg.no_loss_frequencies])

        if len(self._frequencies) < 2:
            raise ValueError("TiedFrequencyMSERegularization needs at least two frequencies.")

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                *args) -> torch.Tensor:
        """Calculate the sum of mean squared deviations between adjacent predicted frequencies.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Dictionary containing ``y_hat_{frequency}`` for each frequency.
        ground_truth : Dict[str, torch.Tensor]
            Dictionary continaing ``y_{frequency}`` for each frequency.

        Returns
        -------
        torch.Tensor
            The sum of mean squared deviations for each pair of adjacent frequencies.
        """

        loss = 0
        for idx, freq in enumerate(self._frequencies):
            if idx == 0:
                continue
            frequency_factor = int(get_frequency_factor(self._frequencies[idx - 1], freq))
            freq_pred = prediction[f'y_hat_{freq}']
            mean_freq_pred = freq_pred.view(freq_pred.shape[0], freq_pred.shape[1] // frequency_factor,
                                            frequency_factor, -1).mean(dim=2)
            lower_freq_pred = prediction[f'y_hat_{self._frequencies[idx - 1]}'][:, -mean_freq_pred.shape[1]:]
            loss = loss + torch.mean((lower_freq_pred - mean_freq_pred)**2)

        return loss


class ForecastOverlapMSERegularization(BaseRegularization):
    """Squared error regularization for penalizing differences between hindcast and forecast models.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(ForecastOverlapMSERegularization, self).__init__(cfg, name='forecast_overlap', weight=weight)

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_output: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Calculate the squared difference between hindcast and forecast model during overlap.

        Does not work with multi-frequency models.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Not used.
        ground_truth : Dict[str, torch.Tensor]
            Not used.
        other_model_output : Dict[str, Dict[str, torch.Tensor]]
            Dictionary containing ``y_forecast_overlap`` and ``y_hindcast_overlap``, which are
            both dictionaries containing keys to relevant model outputs.

        Returns
        -------
        torch.Tensor
            The sum of mean squared deviations between overlapping portions of hindcast and forecast models.

        Raises
        ------
        ValueError if y_hindcast_overlap or y_forecast_overlap is not present in model output.
        """
        loss = 0
        if 'y_hindcast_overlap' not in other_model_output or not other_model_output['y_hindcast_overlap']:
            raise ValueError('y_hindcast_overlap is not present in the model output.')
        if 'y_forecast_overlap' not in other_model_output or not other_model_output['y_forecast_overlap']:
            raise ValueError('y_forecast_overlap is not present in the model output.')
        for key in other_model_output['y_hindcast_overlap']:
            hindcast = other_model_output['y_hindcast_overlap'][key]
            forecast = other_model_output['y_forecast_overlap'][key]
            loss += torch.mean((hindcast - forecast)**2)
        return loss


def _peak_labels(y: torch.Tensor, window: int, threshold: float) -> torch.Tensor:
    """Binary peak labels of a target window.

    A time step is a peak if it is the maximum of its +/- ``window`` neighborhood and lies above ``threshold``.
    This replaces the offline ``findpeaks`` call of the PeakFocus reference implementation, which is not
    applicable here because NeuralHydrology builds its windows on the fly.

    Parameters
    ----------
    y : torch.Tensor
        Target values of shape ``[batch, predict_last_n, n_targets]``. May contain NaN.
    window : int
        Half-width of the neighborhood a peak has to dominate.
    threshold : float
        Minimum (standardized) target value of a peak.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``[batch, predict_last_n, n_targets]`` with 1 at peaks and 0 elsewhere. NaN
        target steps are never labeled as peaks.
    """
    finite = ~torch.isnan(y)
    filled = torch.where(finite, y, torch.full_like(y, -float('inf')))
    if window > 0 and y.shape[1] > 1:
        pooled = torch.nn.functional.max_pool1d(filled.transpose(1, 2),
                                                kernel_size=2 * window + 1,
                                                stride=1,
                                                padding=window).transpose(1, 2)
    else:
        pooled = filled
    return ((filled >= pooled) & finite & (filled > threshold)).to(y.dtype)


def _soften(labels: torch.Tensor, sigma: float, tolerance: int) -> torch.Tensor:
    """Spread hard peak labels over a tolerance window with a Gaussian profile.

    Port of ``exp_peak_detect_based_on_long_term_forecasting.create_soft_peak_labels``.
    """
    if tolerance <= 0:
        return labels
    soft = labels
    for distance in range(1, tolerance + 1):
        weight = math.exp(-(distance**2) / (2 * sigma**2))
        shifted_right = torch.nn.functional.pad(labels, (0, 0, distance, 0))[:, :labels.shape[1], :]
        shifted_left = torch.nn.functional.pad(labels, (0, 0, 0, distance))[:, distance:, :]
        soft = torch.maximum(soft, weight * torch.maximum(shifted_right, shifted_left))
    return soft


class PeakFocalRegularization(BaseRegularization):
    """Focal loss on a model's peak-localization logits.

    This is the peak-classification term of the PeakFocus "triple hybrid objective"
    (``peak_loss_weight * FocalLoss(peak_logits, soft_peak_labels)``). It requires the model to return
    ``peak_logits`` (see :py:class:`neuralhydrology.modelzoo.peakfocus.PeakFocus`).

    The peak labels are derived from the target window itself, using ``peak_label_window`` and
    ``peak_label_threshold``, and softened with ``peak_soft_label_sigma`` / ``peak_soft_label_tolerance``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    weight : float, optional
        Weight of the regularization term. Default: 1.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(PeakFocalRegularization, self).__init__(cfg, name='peak_focal', weight=weight)
        self._window = cfg.peak_label_window
        self._threshold = cfg.peak_label_threshold
        self._sigma = cfg.peak_soft_label_sigma
        self._tolerance = cfg.peak_soft_label_tolerance
        self._alpha = cfg.peak_focal_alpha
        self._gamma = cfg.peak_focal_gamma

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'peak_logits' not in other_model_data:
            raise ValueError("The 'peak_focal' regularization requires a model that returns 'peak_logits'.")
        logits = other_model_data['peak_logits']
        y = ground_truth['y']

        targets = _soften(_peak_labels(y, self._window, self._threshold), self._sigma, self._tolerance)
        mask = ~torch.isnan(y)
        if not torch.any(mask):
            return logits.sum() * 0.0

        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = torch.where(targets > 0.5, self._alpha, 1 - self._alpha)
        loss = alpha_t * (1 - p_t).pow(self._gamma) * bce
        return loss[mask].mean()


class PeakWeightedMSERegularization(BaseRegularization):
    """Mean squared error restricted to the peaks of the target window.

    This is the intensity term of the PeakFocus "triple hybrid objective" (``tp_mse_loss_weight``): the same
    squared error as the main loss, but averaged only over the time steps that the peak labels mark, so that
    peak magnitude is not diluted by the non-peak majority.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    weight : float, optional
        Weight of the regularization term. Default: 1.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(PeakWeightedMSERegularization, self).__init__(cfg, name='peak_weighted_mse', weight=weight)
        self._window = cfg.peak_label_window
        self._threshold = cfg.peak_label_threshold
        self._sigma = cfg.peak_soft_label_sigma
        self._tolerance = cfg.peak_soft_label_tolerance

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        y_hat, y = prediction['y_hat'], ground_truth['y']
        weights = _soften(_peak_labels(y, self._window, self._threshold), self._sigma, self._tolerance)
        weights = torch.where(torch.isnan(y), torch.zeros_like(weights), weights)
        total = weights.sum()
        if total == 0:
            return y_hat.sum() * 0.0
        squared_error = (torch.nan_to_num(y_hat[..., :y.shape[-1]], nan=0.0) - torch.nan_to_num(y, nan=0.0))**2
        return (squared_error * weights).sum() / total
