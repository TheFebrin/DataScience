import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(trajectory: np.ndarray):
    X = trajectory[:, 1]
    Y = trajectory[:, 2]
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y)
    plt.title("Trajectory with {} points".format(len(X)))
    plt.show()
