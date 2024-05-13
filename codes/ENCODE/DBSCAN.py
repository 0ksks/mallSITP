from encode import get_encoded


def plot_elbow(k, points):
    from sklearn.neighbors import NearestNeighbors as NN
    import matplotlib.pyplot as plt

    knn = NN(n_neighbors=k).fit(points)
    distance, _ = knn.kneighbors(points)
    distance = distance[:, -1]
    distance.sort()
    x = list(range(distance.shape[0]))
    plt.plot(x, distance)
    plt.show()


if __name__ == "__main__":
    import time
    from sklearn.cluster import dbscan
    from itertools import product
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    minPtsStep = 10
    epsLen = 4
    epsRange = np.linspace(0.009, 0.012, epsLen)
    minPtsRange = range(100, 200, minPtsStep)
    # 网格搜索的精度

    grids = product(epsRange, minPtsRange)
    minPtsLen = len(list(minPtsRange))
    points = get_encoded().detach().numpy()

    start = time.time()
    for idx, (eps, minPts) in enumerate(grids):
        end = time.time()
        print(
            f"{eps:.2e} {minPts} ({idx+1}/{epsLen*minPtsLen}) running {round(end-start,3)}"
        )
        _, labels = dbscan(points, eps=eps, min_samples=minPts, metric="euclidean")
        dirfig = "code/ENCODE/cluster/second/figs"
        if not os.path.exists(dirfig):
            os.makedirs(dirfig)
        dirnpy = "code/ENCODE/cluster/second/labels"
        if not os.path.exists(dirnpy):
            os.makedirs(dirnpy)
        plt.scatter(*points.T, c=labels, alpha=0.5, s=0.5, cmap="jet")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"({eps:.2e},{minPts})")
        plt.axis("off")
        plt.savefig(f"{dirfig}/{idx+1+40}_({eps:.2e},{minPts}).png", dpi=300)
        np.save(f"{dirnpy}/{idx+1+40}_({eps:.2e},{minPts}).npy", labels)
