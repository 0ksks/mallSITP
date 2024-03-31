import sys
sys.path.append("/Users/admin/Desktop/商场sitp/ori/codes")
from AE.datasets import AEDataset
# read data
data = AEDataset("norm").data
# PCA
from sklearn.decomposition import PCA
pca = PCA(2)
pca.fit(data)
transformed_data = pca.transform(data)
# plot
import matplotlib.pyplot as plt
plt.scatter(*transformed_data.T,s=0.1)
plt.show()