"""Generate the 30 NeuralHydrology run configurations (10 models x 3 input sets) used by the comparison study.

All runs share the same data split, optimizer, loss, batch size, sequence length and early-stopping rule; only the
model block (and the handful of keys that a model architecture *requires*) differs. Re-run this script after changing
anything in ``COMMON`` or ``MODEL_SPECS`` -- the notebooks read the generated ``configs/<model>_<set>.yml`` files.
"""
from pathlib import Path

import yaml

from prepare_data import INPUT_SETS, STATION_COLS

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"

# --- settings shared by every run -------------------------------------------------------------------------------
COMMON = {
    "run_dir": None,
    "train_basin_file": "YDDam_basin.txt",
    "validation_basin_file": "YDDam_basin.txt",
    "test_basin_file": "YDDam_basin.txt",
    # 20 years of hourly data (2006-2025): 12 / 4 / 4 years
    "train_start_date": "01/01/2006",
    "train_end_date": "31/12/2017",
    "validation_start_date": "01/01/2018",
    "validation_end_date": "31/12/2021",
    "test_start_date": "01/01/2022",
    "test_end_date": "31/12/2025",
    "seed": 111,
    "device": "cuda:0",
    # validation after every epoch (needed for early stopping); NSE/KGE of the full validation period are logged
    "validate_every": 1,
    "validate_n_random_basins": 1,
    "metrics": ["NSE", "KGE"],
    "head": "regression",
    "output_activation": "linear",
    # optimisation: identical for all models (model-specific tuning is left for future work)
    "optimizer": "Adam",
    "loss": "MSE",
    "learning_rate": {0: 1e-3},
    "batch_size": 256,
    "epochs": 50,
    "clip_gradient_norm": 1,
    # early stopping: at least 30 epochs, at most 50, stop after 5 epochs without validation improvement.
    # (with early_stopping the library also enables ReduceLROnPlateau: LR x 0.5 after 3 epochs without improvement)
    "early_stopping": True,
    "patience_early_stopping": 5,
    "minimum_epochs_before_early_stopping": 30,
    "patience_dynamic_learning_rate": 3,
    "factor_dynamic_learning_rate": 0.5,
    # 30 days of hourly look-back, prediction of the current hour
    "seq_length": 720,
    "predict_last_n": 1,
    "num_workers": 4,
    "log_interval": 100,
    "log_tensorboard": False,
    "log_n_figures": 0,
    "save_weights_every": 1,
    "save_validation_results": False,
    # data
    "dataset": "hourlycsv",
    "data_dir": "../../data/YDDam_paper",
    "forcings": ["hourly"],
    "target_variables": ["inflow"],
    "clip_targets_to_zero": ["inflow"],
}

# --- model blocks -------------------------------------------------------------------------------------------------
# 'nh_model' is the key understood by neuralhydrology.modelzoo.get_model; 'name' is the label used in the paper.
MODEL_SPECS = {
    "lstm": {
        "name": "LSTM",
        "nh_model": "cudalstm",
        "cfg": {"hidden_size": 64, "initial_forget_bias": 3, "output_dropout": 0.4},
    },
    "gru": {
        "name": "GRU",
        "nh_model": "gru",
        "cfg": {"hidden_size": 64, "output_dropout": 0.4},
    },
    "transformer": {
        "name": "Transformer",
        "nh_model": "transformer",
        "cfg": {
            # the transformer works in the embedding space, so the inputs are first mapped to 64 dimensions
            "dynamics_embedding": {"type": "fc", "hiddens": [64], "activation": "tanh", "dropout": 0.0},
            "transformer_nheads": 4,
            "transformer_nlayers": 2,
            "transformer_dim_feedforward": 128,
            "transformer_dropout": 0.1,
            "transformer_positional_dropout": 0.1,
            "transformer_positional_encoding_type": "sum",
            "output_dropout": 0.4,
        },
    },
    "mtslstm": {
        "name": "MTS-LSTM",
        "nh_model": "mtslstm",
        "cfg": {
            "hidden_size": 64,
            "initial_forget_bias": 3,
            "output_dropout": 0.4,
            # daily branch: 365 days from the daily csv; hourly branch: last 14 days (336 h) from the hourly csv
            "use_frequencies": ["1D", "1h"],
            "seq_length": {"1D": 365, "1h": 336},
            "predict_last_n": {"1D": 1, "1h": 24},
            "shared_mtslstm": False,
            "transfer_mtslstm_states": {"h": "linear", "c": "linear"},
            "regularization": ["tie_frequencies"],
            "forcings": ["hourly", "daily"],
        },
    },
    "mclstm": {
        "name": "MC-LSTM",
        "nh_model": "mclstm",
        "cfg": {
            "hidden_size": 64,
            "initial_forget_bias": 3,
            # mass conservation needs mass input and target in the same depth unit (mm/h), without normalization
            "target_variables": ["inflow_mm"],
            "clip_targets_to_zero": ["inflow_mm"],
        },
    },
    "odelstm": {
        "name": "ODE-LSTM",
        "nh_model": "odelstm",
        "cfg": {
            "hidden_size": 64,
            "initial_forget_bias": 3,
            "output_dropout": 0.4,
            "use_frequencies": ["1D", "1h"],
            "seq_length": {"1D": 365, "1h": 336},
            "predict_last_n": {"1D": 1, "1h": 24},
            "ode_method": "euler",
            "ode_num_unfolds": 4,
            "ode_random_freq_lower_bound": "5D",
            "regularization": ["tie_frequencies"],
        },
    },
    "mamba": {
        "name": "Mamba",
        "nh_model": "mamba",
        "cfg": {"hidden_size": 64, "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2, "output_dropout": 0.4},
    },
    "xlstm": {
        "name": "xLSTM",
        "nh_model": "x_lstm",
        "cfg": {
            "hidden_size": 64,
            "xlstm_num_blocks": 2,
            "xlstm_slstm_at": [1],
            "xlstm_heads": 4,
            "xlstm_kernel_size": 4,
            "xlstm_proj_factor": 1.3,
            "output_dropout": 0.4,
        },
    },
    "peakfocus": {
        "name": "PeakFocus",
        "nh_model": "peakfocus",
        "cfg": {
            "peakfocus_d_model": 64,
            "peakfocus_n_heads": 4,
            "peakfocus_d_ff": 128,
            "peakfocus_d_layers": 1,
            "peakfocus_mlp_layers": 2,
            "peakfocus_dropout": 0.1,
            "peakfocus_n_scales": 2,
            "peakfocus_if_msm_pl": True,
            "peakfocus_if_lad": True,
            # the reference's instance normalization needs the target history for de-normalization (not an input here)
            "peakfocus_instance_norm": False,
            "output_dropout": 0.4,
            # the peak head is supervised on the last 24 h of the window; only the last step is evaluated
            "predict_last_n": 24,
            "regularization": [["peak_focal", 1.0], ["peak_weighted_mse", 1.0]],
            "peak_label_window": 3,
            "peak_label_threshold": 1.0,
            "peak_soft_label_sigma": 2.0,
            "peak_soft_label_tolerance": 1,
            "peak_focal_alpha": 0.25,
            "peak_focal_gamma": 2.0,
        },
    },
    "exformer": {
        "name": "Exformer",
        "nh_model": "exformer",
        "cfg": {
            "exformer_patch_size": 24,  # one patch = one day -> 30 patches per 720 h window
            "exformer_embed_dim": 4,  # attention dimension = 4 x 24 = 96
            "exformer_n_heads": 4,
            "exformer_encoder_depth": 2,
            "exformer_d_ff": 32,
            "exformer_dropout": 0.1,
            "exformer_local_window": 2,
            "exformer_stride": 7,
            "exformer_mask": "dozer_ext_0",
            "exformer_patch_thres": 1,
            "exformer_moving_avg": [13, 25],
            "exformer_extreme_threshold": 2.0,
            "output_dropout": 0.4,
        },
    },
}

MODEL_ORDER = list(MODEL_SPECS)


def dynamic_inputs_for(model: str, input_set: str) -> dict:
    """Input-related keys for one (model, input set) combination."""
    cols = list(INPUT_SETS[input_set])
    keys = {}
    if model == "mtslstm":
        # with two forcing products the csv columns get the suffix _<forcing>
        keys["dynamic_inputs"] = {"1D": [f"{c}_daily" for c in cols], "1h": [f"{c}_hourly" for c in cols]}
    elif model == "mclstm":
        # one scalar mass input (un-normalized), the remaining variables as (standardized) auxiliary inputs
        mass = "rf_stn_mean" if input_set == "B" else "rf_avg"
        aux = STATION_COLS if input_set in ("B", "C") else []
        if input_set in ("A", "C"):
            aux = aux + ["rf_avg_aux"]
        keys["mass_inputs"] = [mass]
        keys["dynamic_inputs"] = aux
        keys["custom_normalization"] = {
            mass: {"centering": "none", "scaling": "none"},
            "inflow_mm": {"centering": "none", "scaling": "none"},
        }
    elif model == "exformer":
        # the extreme label is derived from one scalar rainfall series
        extreme = "rf_stn_mean" if input_set == "B" else "rf_avg"
        if extreme not in cols:
            cols = cols + [extreme]
        keys["dynamic_inputs"] = cols
        keys["exformer_extreme_variable"] = extreme
    else:
        keys["dynamic_inputs"] = cols
    return keys


def build_config(model: str, input_set: str) -> dict:
    spec = MODEL_SPECS[model]
    cfg = {"experiment_name": f"{model}_{input_set}", **COMMON, "model": spec["nh_model"], **spec["cfg"]}
    cfg.update(dynamic_inputs_for(model, input_set))
    return cfg


def write_configs(config_dir: Path = CONFIG_DIR) -> list:
    config_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model in MODEL_ORDER:
        for input_set in INPUT_SETS:
            path = config_dir / f"{model}_{input_set}.yml"
            with open(path, "w") as fp:
                yaml.safe_dump(build_config(model, input_set), fp, sort_keys=False, default_flow_style=False)
            written.append(path)
    return written


if __name__ == "__main__":
    for path in write_configs():
        print("wrote", path.relative_to(HERE))
