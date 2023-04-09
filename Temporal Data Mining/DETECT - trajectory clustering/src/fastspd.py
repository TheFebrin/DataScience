import numpy as np
import pickle
from typing import Union


class FastSPD:
    def __init__(
            self,
            p_s: float = 0.001,
            p_t: float = 1000,
            out_dir: str = "./VED_dataset/compressed_data.pkl"
            ):
        """
        Attributes:
            p_s (float): limits the space range between points
                that will be squashed into a single point
            p_t (float): limits the time range (in milliseconds) between points
                that will be squashed into a single point
            out_dir (string): directory where processed data should be stored
        """
        self.p_s = p_s
        self.p_t = p_t
        self.out_dir = out_dir

    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        dist = np.sqrt(
            (x[1] - y[1]) ** 2 +
            (x[2] - y[2]) ** 2
        )
        return dist

    def _get_candidates(
        self,
        start_point: float,
        next_points: np.ndarray,
        dist_limit: float
    ) -> np.ndarray:
        n = len(next_points)
        for i in range(n):
            distance = self._distance(start_point, next_points[i])
            if distance > dist_limit:
                return next_points[:i+1]
        return next_points

    def compress(
        self,
        s: np.ndarray,
    ) -> np.ndarray:
        """
        Attributes:
            s (np.ndarray): Trajectory in a form of a matrix
        """
        i = 0
        res_s = []
        while i < len(s) - 1:
            if self._distance(s[i], s[i+1]) > self.p_s:
                i += 1
                continue
            candidates = self._get_candidates(
                start_point=s[i],
                next_points=s[i+1:],
                dist_limit=self.p_s
            )
            if candidates[-1][0] - s[i][0] > self.p_t:
                candidates = np.append(candidates, [s[i]], axis=0)
                res_s = res_s + [np.mean(candidates, axis=0)]
                i += len(candidates)
            i += 1
        res_s = [s[0]] + res_s + [s[-1]]
        return np.array(res_s)
    
    def compress_and_save(
        self,
        trajectories: np.ndarray
    ):
        preprocessed_data = []
        for t in trajectories:
            preprocessed_data.append(self.compress(t))
        with open(self.out_dir, 'wb') as f:
            pickle.dump(preprocessed_data, f)

    def load_compressed(self, dir: Union[None, str] = None) -> list[np.ndarray]:
        if dir is None:
            dir = self.out_dir
        with open(dir, "rb") as f:
            compressed_data = pickle.load(f)
        return compressed_data
