"""Unit tests for the DailyCSV and HourlyCSV dataset classes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.dailycsv import DailyCSV, load_csv_attributes, load_csv_timeseries
from neuralhydrology.datasetzoo.hourlycsv import HourlyCSV
from neuralhydrology.utils.config import Config

BASINS = ['basin_A', 'basin_AB']  # 'basin_A' is a prefix of 'basin_AB' to check for exact file matching


def _write_timeseries(path: Path, index: pd.DatetimeIndex, columns: list, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((len(index), len(columns))), index=index, columns=columns)
    df.index.name = 'date'
    df.to_csv(path)


@pytest.fixture
def csv_data_dir(tmp_path: Path) -> Path:
    """Create a small csv dataset with daily and hourly data, a subdir level and static attributes."""
    data_dir = tmp_path / 'data'
    daily_index = pd.date_range('2000-01-01', '2001-12-31', freq='D')
    hourly_index = pd.date_range('2000-01-01', '2000-03-31 23:00', freq='h')
    for i, basin in enumerate(BASINS):
        # daily data, with the 'files directly in the frequency folder' layout
        _write_timeseries(data_dir / 'time_series' / 'input' / 'daily' / f'{basin}_input.csv',
                          daily_index, ['prcp', 'temp'],
                          seed=i)
        _write_timeseries(data_dir / 'time_series' / 'target' / 'daily' / f'{basin}.csv',
                          daily_index, ['qobs'],
                          seed=10 + i)
        # hourly data, with an additional sub-directory level (as in the user's examples)
        _write_timeseries(data_dir / 'time_series' / 'input' / 'hourly' / 'qobs' / f'{basin}_input.csv',
                          hourly_index, ['prcp', 'temp'],
                          seed=20 + i)
        _write_timeseries(data_dir / 'time_series' / 'target' / 'hourly' / 'qobs' / f'{basin}_target.csv',
                          hourly_index, ['qobs'],
                          seed=30 + i)

    attributes = pd.DataFrame({'basin_id': BASINS, 'area': [10.0, 25.0], 'elev_mean': [300.0, 800.0]})
    (data_dir / 'attributes').mkdir()
    attributes.to_csv(data_dir / 'attributes' / 'attributes.csv', index=False)

    basin_file = tmp_path / 'basins.txt'
    basin_file.write_text('\n'.join(BASINS))
    return data_dir


def _make_config(data_dir: Path, run_dir: Path, **overrides) -> Config:
    cfg = {
        'experiment_name': 'csv_test',
        'run_dir': run_dir,
        'train_dir': run_dir / 'train_data',
        'train_basin_file': data_dir.parent / 'basins.txt',
        'validation_basin_file': data_dir.parent / 'basins.txt',
        'test_basin_file': data_dir.parent / 'basins.txt',
        'train_start_date': '01/01/2000',
        'train_end_date': '31/12/2000',
        'validation_start_date': '01/01/2001',
        'validation_end_date': '31/12/2001',
        'test_start_date': '01/01/2001',
        'test_end_date': '31/12/2001',
        'dataset': 'dailycsv',
        'data_dir': data_dir,
        'dynamic_inputs': ['prcp', 'temp'],
        'target_variables': ['qobs'],
        'static_attributes': ['area', 'elev_mean'],
        'model': 'cudalstm',
        'head': 'regression',
        'hidden_size': 8,
        'seq_length': 10,
        'predict_last_n': 1,
        'batch_size': 4,
        'epochs': 1,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'learning_rate': 1e-3,
        'device': 'cpu',
    }
    cfg.update(overrides)
    return Config(cfg)


def test_daily_csv_with_static_attributes(csv_data_dir, tmp_path):
    cfg = _make_config(csv_data_dir, tmp_path / 'run')
    ds = get_dataset(cfg, is_train=True, period='train')
    assert isinstance(ds, DailyCSV)
    assert len(ds) > 0

    sample = ds[0]
    assert set(sample['x_d'].keys()) == {'prcp', 'temp'}
    assert sample['x_d']['prcp'].shape == (10, 1)
    # two static attributes, standardized with the attribute means/stds computed on the training set
    assert sample['x_s'].shape == (2,)
    assert set(ds.scaler['attribute_means'].index) == {'area', 'elev_mean'}
    for basin in BASINS:
        assert ds._attributes[basin].shape == (2,)


def test_daily_csv_missing_attribute_raises(csv_data_dir, tmp_path):
    cfg = _make_config(csv_data_dir, tmp_path / 'run', static_attributes=['area', 'does_not_exist'])
    with pytest.raises(ValueError, match='does_not_exist'):
        get_dataset(cfg, is_train=True, period='train')


def test_daily_csv_without_attributes_folder(csv_data_dir, tmp_path):
    """Static attributes are optional: without 'static_attributes' the attributes folder is never touched."""
    for f in (csv_data_dir / 'attributes').iterdir():
        f.unlink()
    (csv_data_dir / 'attributes').rmdir()
    cfg = _make_config(csv_data_dir, tmp_path / 'run', static_attributes=[])
    ds = get_dataset(cfg, is_train=True, period='train')
    assert 'x_s' not in ds[0]


def test_hourly_csv_with_subdir_and_attributes(csv_data_dir, tmp_path):
    cfg = _make_config(csv_data_dir,
                       tmp_path / 'run',
                       dataset='hourlycsv',
                       forcings=['hourly'],
                       train_start_date='01/01/2000',
                       train_end_date='29/02/2000',
                       seq_length=24)
    ds = get_dataset(cfg, is_train=True, period='train')
    assert isinstance(ds, HourlyCSV)
    assert ds.frequencies == ['1h']
    sample = ds[0]
    assert sample['x_d']['prcp'].shape == (24, 1)
    assert sample['x_s'].shape == (2,)


def test_hourly_csv_requires_hourly_forcing(csv_data_dir, tmp_path):
    cfg = _make_config(csv_data_dir, tmp_path / 'run', dataset='hourlycsv', forcings=['daily'])
    with pytest.raises(ValueError, match='no hourly forcings'):
        get_dataset(cfg, is_train=True, period='train')


def test_hourly_csv_upsamples_daily_forcings(csv_data_dir, tmp_path):
    """Non-hourly forcings are loaded from the daily folder, upsampled and suffixed with the forcing name."""
    cfg = _make_config(csv_data_dir,
                       tmp_path / 'run',
                       dataset='hourlycsv',
                       forcings=['hourly', 'daily'],
                       dynamic_inputs=['prcp_hourly', 'temp_daily'],
                       train_start_date='01/01/2000',
                       train_end_date='29/02/2000',
                       seq_length=24)
    ds = get_dataset(cfg, is_train=True, period='train')
    df = ds._load_basin_data('basin_A')
    assert {'prcp_hourly', 'temp_hourly', 'prcp_daily', 'temp_daily', 'qobs'} <= set(df.columns)
    # target variables are never suffixed, daily values are forward-filled within a day
    assert 'qobs_hourly' not in df.columns
    day = df.loc['2000-01-05', 'temp_daily']
    assert len(day) == 24 and day.nunique() == 1


def test_load_csv_timeseries_exact_basin_match(csv_data_dir):
    df = load_csv_timeseries(csv_data_dir, 'basin_A', kind='input', frequency='daily')
    assert df.index.name == 'date'
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ['prcp', 'temp']

    with pytest.raises(FileNotFoundError):
        load_csv_timeseries(csv_data_dir, 'basin_', kind='input', frequency='daily')
    with pytest.raises(OSError):
        load_csv_timeseries(csv_data_dir, 'basin_A', kind='input', frequency='monthly')

    # a second matching file (in a sub-directory) makes the basin ambiguous
    extra = csv_data_dir / 'time_series' / 'input' / 'daily' / 'other' / 'basin_A.csv'
    extra.parent.mkdir()
    extra.write_text('date,prcp\n2000-01-01,1.0\n2000-01-02,2.0\n')
    with pytest.raises(ValueError, match='Multiple'):
        load_csv_timeseries(csv_data_dir, 'basin_A', kind='input', frequency='daily')


def test_load_csv_attributes(csv_data_dir):
    df = load_csv_attributes(csv_data_dir, basins=['basin_AB'])
    assert list(df.index) == ['basin_AB']
    assert df.loc['basin_AB', 'area'] == 25.0
    with pytest.raises(ValueError):
        load_csv_attributes(csv_data_dir, basins=['unknown'])


def test_missing_target_file_only_allowed_for_evaluation(csv_data_dir, tmp_path):
    (csv_data_dir / 'time_series' / 'target' / 'daily' / 'basin_A.csv').unlink()
    # no static attributes, because training on a single basin below would give attributes with zero std
    cfg = _make_config(csv_data_dir, tmp_path / 'run', static_attributes=[])

    with pytest.raises(FileNotFoundError):
        get_dataset(cfg, is_train=True, period='train')

    # evaluation: the dataset is created and the targets are NaN for the basin without target file
    train_ds = get_dataset(cfg, is_train=True, period='train', basin='basin_AB')
    ds = get_dataset(cfg, is_train=False, period='test', basin='basin_A', scaler=train_ds.scaler)
    assert np.isnan(ds[0]['y'].numpy()).all()
