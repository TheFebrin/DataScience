import numpy as np


class FastSPD:
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        dist = np.sqrt(
            (x[1] - y[1]) ** 2 +
            (x[2] - y[2]) ** 2
        )
        return dist

    def get_candidates(
        self,
        start_point: float,
        next_points: np.ndarray,
        dist_limit: float
    ) -> np.ndarray:
        n = len(next_points)
        for i in range(n):
            distance = self.distance(start_point, next_points[i])
            if distance > dist_limit:
                return next_points[:i+1]
        return next_points

    def compress(
            self,
            s: np.ndarray,
            p_s: float = 0.001,
            p_t: float = 1000
    ) -> np.ndarray:
        """
        Attributes:
            s (np.ndarray): Trajectory in a form of a matrix
            p_s (float): limits the space range between points
                that will be squashed into a single point
            p_t (float): limits the time range (in milliseconds) between points
                that will be squashed into a single point
        """
        i = 0
        res_s = []
        while i < len(s) - 1:
            if self.distance(s[i], s[i+1]) > p_s:
                i += 1
                continue
            candidates = self.get_candidates(
                start_point=s[i],
                next_points=s[i+1:],
                dist_limit=p_s
            )
            if candidates[-1][0] - s[i][0] > p_t:
                candidates = np.append(candidates, [s[i]], axis=0)
                res_s = res_s + [np.mean(candidates, axis=0)]
                i += len(candidates)
            i += 1
        res_s = [s[0]] + res_s + [s[-1]]
        return np.array(res_s)
