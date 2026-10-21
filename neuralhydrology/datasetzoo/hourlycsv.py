from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import xarray

from neuralhydrology.datasetzoo import dailycsv
from neuralhydrology.utils.config import Config


class HourlyCSV(dailycsv.DailyCSV):
    """Data set class for hourly time series stored as plain csv files (one file per basin).

    This class extends :py:class:`neuralhydrology.datasetzoo.dailycsv.DailyCSV` to hourly in- and output data. The
    ``data_dir`` has to be organized as follows::

        data_dir/
        ├── time_series/
        │   ├── input/
        │   │   ├── hourly/[<subdir>/]<basin>_input.csv    (or <basin>.csv)
        │   │   └── daily/...                              (optional, daily forcings upsampled to hourly)
        │   └── target/
        │       └── hourly/[<subdir>/]<basin>_target.csv   (or <basin>.csv)
        └── attributes/                                   (optional, only needed for ``static_attributes``)
            └── *.csv

    The config argument ``forcings`` is optional and defaults to ``['hourly']``. At least one entry has to end with
    ``hourly`` (e.g., ``hourly``, ``nwp_hourly``); these are loaded from ``time_series/input/hourly``. Any other entry
    is treated as a daily forcing product, loaded from ``time_series/input/daily`` and upsampled to hourly resolution
    by forward-filling. As for the daily dataset, variables get the suffix ``_<forcing>`` if more than one forcing
    product is used. Static attributes are read from ``data_dir/attributes/*.csv`` (see
    :py:func:`neuralhydrology.datasetzoo.dailycsv.load_csv_attributes`).

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used.
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    """

    _frequency = 'hourly'

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(HourlyCSV, self).__init__(cfg=cfg,
                                        is_train=is_train,
                                        period=period,
                                        basin=basin,
                                        additional_features=additional_features,
                                        id_to_int=id_to_int,
                                        scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and target data from csv files."""
        if not any(f.endswith('hourly') for f in self._forcings):
            raise ValueError("Forcings include no hourly forcings set. At least one entry of 'forcings' has to end "
                             "with 'hourly'.")
        return super(HourlyCSV, self)._load_basin_data(basin)

    def _load_forcing(self, basin: str, forcing: str) -> pd.DataFrame:
        """Load one forcing product: hourly products from csv, daily products upsampled to hourly resolution."""
        if forcing.endswith('hourly'):
            return load_hourly_input_csv(self.cfg.data_dir, basin, forcing)

        # load daily forcings and upsample to hourly
        df = dailycsv.load_daily_input_csv(self.cfg.data_dir, basin, forcing)
        return df.resample('1h').ffill()


def load_hourly_input_csv(data_dir: Path, basin: str, forcings: str = None) -> pd.DataFrame:
    """Load the hourly input (forcings) csv file of one basin.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory, containing the 'time_series/input/hourly' folder.
    basin : str
        Basin identifier.
    forcings : str, optional
        Name of the forcing product. If a folder ``time_series/input/hourly/<forcings>`` exists, the file is searched
        in that folder only. Otherwise, the whole 'hourly' folder is searched recursively.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    """
    return dailycsv.load_csv_timeseries(data_dir, basin, kind='input', frequency='hourly', subdir=forcings)


def load_hourly_target_csv(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the hourly target csv file of one basin.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory, containing the 'time_series/target/hourly' folder.
    basin : str
        Basin identifier.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the target variable(s).
    """
    return dailycsv.load_csv_timeseries(data_dir, basin, kind='target', frequency='hourly')
