import pandas as pd
import glob
import numpy as np
from abc import ABC, abstractmethod


class IDataReader(ABC):
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass


class DataReader(IDataReader):
    """
    Reads all CSVs from a given directory, takes the Timestamp, Latitude, Longitude columns
    and saves them to a numpy array of matrices, where each matrix describes a single trajectory.
    A single row consists of: timestamp, latitude, longitude.
    """
    def __init__(self, folder_dir: str, files_limit: int):
        self.dir = folder_dir
        self.files_limit = files_limit
        self._get_all_filenames()

    def _get_all_filenames(self) -> None:
        self.filenames = glob.glob(self.dir)

    def _split_to_trajectories(self, df: pd.DataFrame) -> list[np.ndarray]:
        dfs = dict(tuple(df.groupby("Trip")))
        trajectories = []
        for single_df in dfs.values():
            trajectory_df = single_df[["Timestamp(ms)", "Latitude[deg]", "Longitude[deg]"]]
            trajectory_array = np.array(trajectory_df)
            trajectories.append(trajectory_array)
        return trajectories

    def _read_dfs(self) -> list[pd.DataFrame]:
        dfs = [pd.read_csv(filename)
               for i, filename in enumerate(self.filenames)
               if i < self.files_limit]
        return dfs

    def get_data(self) -> np.ndarray:
        dfs = self._read_dfs()
        dataset = []
        for df in dfs:
            dataset += self._split_to_trajectories(df)
        return np.array(dataset, dtype=object)
