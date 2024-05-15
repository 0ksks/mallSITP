import sys

sys.path.append("/Users/admin/Desktop/LEARN/商场sitp/ori/codes")
import numpy as np
from AE.datasets import AEDataset
from typing import Literal


def get_points(type: Literal["raw", "norm"]) -> np.ndarray:
    """
    (amount, dim)
    """
    data = AEDataset(type)
    return data[:].squeeze().numpy()


def get_labels(path: str) -> np.ndarray:
    """
    1-dim
    """
    return np.load(path)


def plot_radar(points: np.ndarray, labels: np.ndarray, texts=None) -> None:
    import matplotlib.pyplot as plt

    class Datum:
        def __init__(self, point_, label_):
            self.point = point_
            self.label = label_

    labelSet = np.unique(labels)
    N = points.shape[1]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    data: list[Datum] = []
    pointMax = 0
    if not texts:
        texts = list(map(str, range(N)))

    for label in labelSet:
        point = points[labels == label].mean(axis=0)
        tmp = max(point)
        if tmp > pointMax:
            pointMax = tmp
        point = np.concatenate((point, [point[0]]))
        data.append(Datum(point, label))
    _, axs = plt.subplots(2, 7, subplot_kw=dict(polar=True))
    axs = axs.flatten()
    ax = axs[-1]
    for idx, datum in enumerate(data):
        values = datum.point
        axs[idx].fill(angles, values, alpha=0.25)
        axs[idx].set_yticks([])
        axs[idx].set_xticks([])
    ax.set_ylim(0, pointMax * 1.2)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    labels = get_labels("codes/ENCODE/cluster/second/labels/2_(9.00e-03,13).npy")
    points = get_points("norm")
    points += 2
    plot_radar(points, labels)
