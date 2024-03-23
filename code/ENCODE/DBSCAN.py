from sklearn.cluster import dbscan
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from encode import get_encoded,tensor2points

points = tensor2points(get_encoded())
rn = 4
fig,axs = plt.subplots(int(rn/2),2)
epsRange = range(rn)
labels = []
for eps in epsRange:
    epsave = eps
    eps = eps/rn+0.0001
    print("DBSCANing",round(eps,2))
    _,predLabel = dbscan(points,eps=eps)
    labels.append((eps,predLabel))
