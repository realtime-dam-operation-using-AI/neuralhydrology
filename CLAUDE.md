# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NeuralHydrology is a Python library (built on PyTorch) for training neural networks (mostly LSTM variants) on
hydrological time series data — primarily rainfall-runoff / streamflow prediction. Runs are entirely
config-driven via YAML files: users typically don't write Python code to train a model, they write a config
and the library dispatches to the right dataset/model/trainer classes based on its contents.

## Environment setup & commands

Uses `uv` for environment management (see CONTRIBUTING.rst).

```bash
uv sync --all-groups          # create .venv with all deps (dev + docs) installed
```

Run the full test suite:

```bash
uv run pytest --cov=neuralhydrology            # what CI runs (see .github/workflows/pytest-ci.yml)
uv run python -m pytest test                   # equivalent, no coverage
```

Run a single test / test file:

```bash
uv run python -m pytest test/test_config_runs.py -k "test_daily_regression_nan_targets"
```

`test/test_config_runs.py` contains slow, full train+evaluate integration tests parametrized over many
model/dataset/forcing combinations (via fixtures in `test/conftest.py`). Use `--smoke-test` to cut this down to
one representative combination per fixture when iterating locally:

```bash
uv run python -m pytest test --smoke-test
```

Code style: [yapf](https://github.com/google/yapf) with config in `.style.yapf` (Google style, 120 column
limit). Format changed files with yapf before committing if possible.

Build docs locally (from `docs/`):

```bash
uv run make html   # open docs/build/html/index.html
```

CLI entry points (installed via pyproject.toml `[project.scripts]`), all wrapping `neuralhydrology/nh_run.py`:

```bash
nh-run train --config-file /path/to/config.yml
nh-run continue_training --run-dir /path/to/run_dir [--config-file ...]
nh-run finetune --config-file /path/to/finetune_config.yml
nh-run evaluate --run-dir /path/to/run_dir [--period test|validation|train] [--epoch N]
nh-schedule-runs         # neuralhydrology/nh_run_scheduler.py — queue/schedule multiple runs
nh-results-ensemble      # neuralhydrology/utils/nh_results_ensemble.py — ensemble multiple runs' predictions
```

## Architecture

### The config object is the spine of everything

`neuralhydrology/utils/config.py` (`Config` class) parses a run's YAML file and exposes every config key as a
typed attribute (e.g. `cfg.model`, `cfg.dataset`, `cfg.dynamic_inputs`, `cfg.hidden_size`). Almost every
constructor in the codebase (`get_model(cfg)`, `get_dataset(cfg, ...)`, `BaseTrainer(cfg)`, model `__init__`,
dataset `__init__`) takes this single `Config` instance and derives its behavior entirely from it — there's no
separate argument plumbing. When adding a new config option, update `Config` and document it in
`docs/source/usage/config.rst` (CONTRIBUTING.rst explicitly calls this out).

A run directory (`cfg.run_dir`) is created per experiment and holds `config.yml` (the resolved config, so runs
are reproducible), per-epoch model checkpoints, `train_data/train_data_scaler.yml` (feature means/stds computed
during training), and evaluation outputs (`test/model_epochXXX/...`). `continue_training` and `finetune` modes
both start from an existing run's `config.yml` and layer a new config file on top of it
(`Config.update_config()`).

### Three orthogonal "zoos", selected by string keys in the config

1. **Model zoo** (`neuralhydrology/modelzoo/`) — selected by `cfg.model`. `get_model()` in
   `modelzoo/__init__.py` is a big if/elif dispatch on `cfg.model.lower()` (not a registry like datasets — new
   models must be added here by hand). All models subclass `BaseModel` (`modelzoo/basemodel.py`), which handles
   `output_size` (accounting for probabilistic heads like GMM/CMAL/UMAL) and defines the `forward()` /
   `pre_model_hook()` contract. `modelzoo/template.py` documents the contract for adding a new model: constructor
   takes only `cfg`, `forward()` takes a dict with `x_d` (and optionally `x_s`, `x_one_hot`, each with `_{freq}`
   suffixes when `use_frequencies` has multiple entries) and must return a dict containing `y_hat` (or
   `y_hat_{freq}` per frequency) plus typically `h_n`/`c_n`. Output heads live in `modelzoo/head.py` and should be
   used rather than hand-rolling an output layer, so the probabilistic heads keep working. `module_parts` on a
   model class names the submodules eligible for `finetune_modules` during finetuning.

2. **Dataset zoo** (`neuralhydrology/datasetzoo/`) — selected by `cfg.dataset`, resolved through
   `DatasetRegistry` (`datasetzoo/datasetregistry.py`), a real registry (unlike the model zoo). Built-in datasets
   are registered in `datasetzoo/__init__.py`; new dataset classes can also be registered at runtime via
   `register_dataset(key, cls)` without editing the library. All datasets subclass `BaseDataset`
   (`datasetzoo/basedataset.py`, ~900 lines — the shared preprocessing/windowing/scaling engine), and only need to
   implement `_load_basin_data(basin)` (time-indexed DataFrame of forcings/target) and `_load_attributes()`
   (basin-indexed DataFrame of static attributes) — see `datasetzoo/template.py`. Most concrete datasets (e.g.
   `camelsus.py`, `camelsgb.py`, `lamah.py`) are thin: they implement those two loaders plus a module-level
   loader function reusable outside the class, and BaseDataset does the rest (train/val/test period slicing,
   per-basin or per-period date ranges, multi-frequency resampling, scaling/caching, PyTorch `Dataset.__getitem__`
   windowing).

3. **Loss/head combination** — `cfg.head` (regression, gmm, umal, cmal) determines both the model's output size
   (`BaseModel.__init__`) and which loss function trains it (`training/loss.py`); `training/train.py`'s
   `start_training()` validates `cfg.head` is one of these before building the trainer.

### Training and evaluation flow

`training/basetrainer.py` (`BaseTrainer`) is the training loop: builds the model via `get_model(cfg)`, builds
train/validation datasets via `get_dataset(cfg, ...)`, runs the epoch loop, checkpointing, early stopping
(`earlystopper.py`), regularization (`regularization.py`), and periodic validation. `evaluation/tester.py`
(`Tester`) and `evaluation/evaluate.py` (`start_evaluation()`) mirror this for inference: load a checkpoint, run
the model over a period, and hand off to `evaluation/metrics.py` (NSE, KGE, FDC signatures, probabilistic
metrics like CRPS) and `evaluation/plots.py`. `evaluation/signatures.py` computes hydrological signatures (FDC
slope, baseflow index, etc.) used both as evaluation metrics and as potential additional features.

### Multi-frequency / multi-timescale models

`cfg.use_frequencies` (e.g. `['1D', '1H']`) triggers multi-timescale training (models like `MTSLSTM`,
`ODELSTM`). This is the source of the `_{freq}` suffix convention on input/output dict keys mentioned above —
grep for `use_frequencies` when touching model I/O or dataset windowing code, since single- and multi-frequency
code paths often diverge.

### Extending the library

Both `modelzoo/template.py` and `datasetzoo/template.py` are living documentation of the extension contract, not
just boilerplate — read them (and the corresponding notebook in `examples/02-Adding-Models` /
`examples/03-Adding-Datasets`) before adding a new model or dataset. New models must be wired into `get_model()`
in `modelzoo/__init__.py`; new datasets can either be added to `datasetzoo/__init__.py` or registered externally
via `register_dataset()`.

## Tests

- `test/test_config_runs.py` — end-to-end train+evaluate runs against YAML configs in `test/test_configs/`,
  parametrized across models/datasets/forcings/frequencies via `test/conftest.py` fixtures. These are the
  closest thing to integration tests for the whole pipeline.
- `test/test_data/` contains a small bundled CAMELS-US subset (4 basins) used by these integration tests — no
  external data download needed to run the test suite.
- `test_datautils.py`, `test_metrics.py`, `test_mclstm.py`, `test_custom_lstm.py`, `test_uncertainty.py` are
  more focused unit tests.
