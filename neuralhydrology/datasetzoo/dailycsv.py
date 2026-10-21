import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import xarray

from neuralhydrology.datasetzoo import genericdataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

# column names that are recognized as the datetime column of a csv file (checked in this order). If none of these
# exists, the first column of the file is used.
_DATE_COLUMN_CANDIDATES = ('date', 'datetime', 'time')


class DailyCSV(BaseDataset):
    """Data set class for daily time series stored as plain csv files (one file per basin).

    This dataset reads the column names directly from the csv files, so any variable name that exists as a column in
    the input (or target) file can be used in ``dynamic_inputs``, ``target_variables``, ``evolving_attributes``, etc.
    The ``data_dir`` has to be organized as follows::

        data_dir/
        ├── time_series/
        │   ├── input/
        │   │   └── daily/[<subdir>/]<basin>_input.csv     (or <basin>.csv)
        │   └── target/
        │       └── daily/[<subdir>/]<basin>_target.csv    (or <basin>.csv)
        └── attributes/                                   (optional, only needed for ``static_attributes``)
            └── *.csv

    Input and target files are searched recursively below ``time_series/input/daily`` and ``time_series/target/daily``,
    so an arbitrary sub-directory level (e.g., named after the target variable or the forcing product) is allowed. The
    file stem has to match the basin id exactly (``<basin>.csv``, ``<basin>_input.csv`` or ``<basin>_target.csv``).
    Each csv file needs a datetime column (``date``, ``datetime`` or ``time``; otherwise the first column is used) and
    one column per variable. Missing values have to be stored as empty fields or ``NaN``, not as ``-999`` or similar.

    The config argument ``forcings`` is optional for this dataset and defaults to ``['daily']``. If multiple forcing
    products are listed (e.g., ``forcings: [obs, nwp]``), each product is loaded from the sub-directory
    ``time_series/input/daily/<forcing>`` (falling back to the recursive search if that sub-directory does not exist)
    and the variable names get the suffix ``_<forcing>``, following the convention of the CAMELS datasets. Target
    variables are never suffixed.

    Static attributes are read from ``data_dir/attributes/*.csv`` in the same format as the
    :py:class:`neuralhydrology.datasetzoo.genericdataset.GenericDataset`: basin-indexed csv files (first column =
    basin id) with one attribute per column. See :py:func:`load_csv_attributes` for details on multiple files.

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

    # name of the frequency folder below 'time_series/input' and 'time_series/target'
    _frequency = 'daily'

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # 'forcings' is optional for the csv datasets. Must be resolved before the parent class loads the data.
        self._forcings = cfg.forcings if cfg.as_dict().get('forcings') else [self._frequency]
        super(DailyCSV, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and target data from csv files."""
        dfs = []
        for forcing in self._forcings:
            df = self._load_forcing(basin, forcing)

            # rename columns
            if len(self._forcings) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)

        if df.columns.has_duplicates:
            duplicates = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate input columns {duplicates} for basin {basin}. If multiple forcing products "
                             "share variable names, list them separately in 'forcings' so they get a suffix.")

        return self._add_targets(df, basin)

    def _load_forcing(self, basin: str, forcing: str) -> pd.DataFrame:
        """Load the input csv file of one forcing product for one basin."""
        return load_daily_input_csv(self.cfg.data_dir, basin, forcing)

    def _add_targets(self, df: pd.DataFrame, basin: str) -> pd.DataFrame:
        """Join the target csv file to the input data.

        During training, a missing target file is an error. During evaluation, a missing target file only triggers a
        warning, as `BaseDataset` adds all-NaN target columns so that predictions can still be made.
        """
        try:
            targets = load_csv_timeseries(self.cfg.data_dir, basin, kind='target', frequency=self._frequency)
        except FileNotFoundError:
            if self.is_train:
                raise
            LOGGER.warning(f"No target file found for basin {basin}. Targets will be filled with NaNs.")
            return df

        overlap = [col for col in targets.columns if col in df.columns]
        if overlap:
            raise ValueError(f"Columns {overlap} exist in both the input and the target file of basin {basin}.")

        # outer join, so that periods covered by only one of the two files are kept (filled with NaNs).
        return pd.concat([df, targets], axis=1).sort_index()

    def _load_attributes(self) -> pd.DataFrame:
        """Load static catchment attributes from 'data_dir/attributes/*.csv'."""
        return load_csv_attributes(self.cfg.data_dir, basins=self.basins)


def load_csv_attributes(data_dir: Path, basins: List[str] = None) -> pd.DataFrame:
    """Load static attributes from csv files.

    The attributes are read from ``data_dir/attributes``, which has to contain one or more csv files. The first column
    of each file is interpreted as the basin id (read as string), all other columns as attributes. This is the same
    format as used by the :py:class:`neuralhydrology.datasetzoo.genericdataset.GenericDataset`, and the loading is
    delegated to :py:func:`neuralhydrology.datasetzoo.genericdataset.load_attributes`.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory. This folder must contain an 'attributes' folder with one or multiple csv files.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of
        all basins are returned.

    Returns
    -------
    pandas.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns. If the attributes folder contains multiple
        files, they are either concatenated along the attributes (if the files refer to the same basins) or along the
        basins (if the files contain the same attributes).

    Raises
    ------
    FileNotFoundError
        If the attributes folder is not found or does not contain any csv files.
    ValueError
        If an attributes file contains duplicate basin or attribute names, multiple files are found that have no
        overlap, or there are no attributes for a basin specified in `basins`.
    """
    return genericdataset.load_attributes(Path(data_dir), basins=basins)


def load_csv_timeseries(data_dir: Path, basin: str, kind: str, frequency: str, subdir: str = None) -> pd.DataFrame:
    """Load the csv time series file of one basin.

    The file is searched recursively below ``data_dir/time_series/<kind>/<frequency>`` (or below
    ``data_dir/time_series/<kind>/<frequency>/<subdir>`` if `subdir` is passed and that directory exists). Exactly one
    file with the stem ``<basin>`` or ``<basin>_<kind>`` has to exist.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory, containing the 'time_series' folder.
    basin : str
        Basin identifier. Must match the file stem exactly (excluding the optional ``_<kind>`` suffix).
    kind : {'input', 'target'}
        Whether to load the input (forcings) or the target file.
    frequency : {'daily', 'hourly'}
        Name of the frequency folder.
    subdir : str, optional
        Optional sub-directory (e.g., the name of a forcing product). Ignored if it does not exist.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame (index name 'date') containing all columns of the csv file.

    Raises
    ------
    OSError
        If the ``time_series/<kind>/<frequency>`` folder does not exist.
    FileNotFoundError
        If no csv file exists for the basin.
    ValueError
        If multiple csv files match the basin, or the file contains duplicate timestamps.
    """
    if kind not in ('input', 'target'):
        raise ValueError(f"kind must be 'input' or 'target', not {kind}.")

    search_dir = Path(data_dir) / 'time_series' / kind / frequency
    if not search_dir.is_dir():
        raise OSError(f"{search_dir} does not exist")
    if subdir and (search_dir / subdir).is_dir():
        search_dir = search_dir / subdir

    valid_stems = {basin, f"{basin}_{kind}"}
    files = sorted(f for f in search_dir.rglob('*.csv') if f.stem in valid_stems)
    if not files:
        raise FileNotFoundError(f"No {kind} file for basin {basin} found below {search_dir}. Expected a file named "
                                f"'{basin}.csv' or '{basin}_{kind}.csv'.")
    if len(files) > 1:
        raise ValueError(f"Multiple {kind} files found for basin {basin}: {[str(f) for f in files]}")

    return _read_timeseries_csv(files[0])


def load_daily_input_csv(data_dir: Path, basin: str, forcings: str = None) -> pd.DataFrame:
    """Load the daily input (forcings) csv file of one basin.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory, containing the 'time_series/input/daily' folder.
    basin : str
        Basin identifier.
    forcings : str, optional
        Name of the forcing product. If a folder ``time_series/input/daily/<forcings>`` exists, the file is searched in
        that folder only. Otherwise, the whole 'daily' folder is searched recursively.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    """
    return load_csv_timeseries(data_dir, basin, kind='input', frequency='daily', subdir=forcings)


def load_daily_target_csv(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the daily target csv file of one basin.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory, containing the 'time_series/target/daily' folder.
    basin : str
        Basin identifier.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the target variable(s).
    """
    return load_csv_timeseries(data_dir, basin, kind='target', frequency='daily')


def _read_timeseries_csv(file_path: Path) -> pd.DataFrame:
    """Read a csv file into a DataFrame with a DatetimeIndex named 'date'."""
    df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        raise ValueError(f"{file_path} needs at least a datetime column and one data column.")

    date_col = next((c for c in _DATE_COLUMN_CANDIDATES if c in df.columns), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = 'date'

    if df.index.has_duplicates:
        raise ValueError(f"{file_path} contains duplicate timestamps.")

    return df
