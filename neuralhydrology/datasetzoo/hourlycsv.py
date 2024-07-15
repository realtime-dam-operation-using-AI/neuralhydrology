import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo import dailycsv
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class HourlyCSV(dailycsv.DailyCSV):
    """Data set class for the DailyCSV Dataset
    
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

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: list = [],
                 id_to_int: dict = {},
                 scaler: dict = {}):
        self._warn_slow_loading = True
        super(HourlyCSV, self).__init__(cfg=cfg,
                                             is_train=is_train,
                                             period=period,
                                             basin=basin,
                                             additional_features=additional_features,
                                             id_to_int=id_to_int,
                                             scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        # get forcings
        dfs = []
        if not any(f.endswith('hourly') for f in self.cfg.forcings):
            raise ValueError('Forcings include no hourly forcings set.')
        for forcing in self.cfg.forcings:
            if forcing[-7:] == 'hourly':
                df = self.load_hourly_data(basin, forcing)
            else:
                # load daily CAMELS forcings and upsample to hourly
                df = dailycsv.load_daily_weather_csv(self.cfg.data_dir, basin, forcing)
                df = df.resample('1H').ffill()
            if len(self.cfg.forcings) > 1:
                # rename columns
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns if 'qobs' not in col.lower()})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)

        # collapse all input features to a single list, to check for 'QObs(mm/d)'.
        all_features = self.cfg.target_variables
        if isinstance(self.cfg.dynamic_inputs, dict):
            for val in self.cfg.dynamic_inputs.values():
                all_features = all_features + val
        elif isinstance(self.cfg.dynamic_inputs, list):
            all_features = all_features + self.cfg.dynamic_inputs

        return df

    def load_hourly_data(self, basin: str, forcings: str) -> pd.DataFrame:
        """Load a single set of daily forcings and discharge.
        
        Parameters
        ----------
        basin : str
            Identifier of the basin for which to load data.
        forcings : str
            Name of the forcings set to load.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with forcings and discharge values for the specified basin.
        """
        df = load_hourly_weather_csv(self.cfg.data_dir, self.cfg.target_variables, basin, forcings)

            # add discharge
        df = df.join(load_hourly_inflow_csv(self.cfg.data_dir, self.cfg.target_variables, basin))

        return df


def load_hourly_weather_csv(data_dir: Path, input_dir: Path, basin: str, forcings: str) -> pd.DataFrame:
    """Load hourly input data set.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory
    basin : str
        
    forcings : str

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    """
    forcing_path = data_dir / 'time_series' /  'input' / 'hourly'
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob(f'{input_dir[0]}/*.csv'))
    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {forcing_path}')

    return pd.read_csv(file_path, index_col=['date'], parse_dates=['date'])


def load_hourly_inflow_csv(data_dir: Path, input_dir: Path, basin: str) -> pd.DataFrame:
    """Load hourly target data set.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory
    basin : str
        
    Returns
    -------
    pd.Series
        Time-index Series of the discharge values (mm/hour)
    """
    forcing_path = data_dir / 'time_series' /  'target' / 'hourly'
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob(f'{input_dir[0]}/*.csv'))
    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {files}')

    return pd.read_csv(file_path, index_col=['date'], parse_dates=['date'])