import sys
sys.path.append("/Users/admin/Desktop/商场sitp/ori/codes")
import numpy as np
def get_train_test(test_size = 0.2):
    """
    Args:
        test_size (float, optional): Defaults to 0.2.

    Returns:
        tuple: (X_train, X_test, y_trian, y_test)
    """
    from read_complete import get_ori_data
    complete = get_ori_data()
    points = complete["points"]
    labels = complete["labels"]
    print("loading selector")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(points, labels, test_size=test_size)
    return (X_train, X_test, y_train, y_test)

def train_test_knn(test_size = 0.2, n_neighbours = 4):
    print("loading KNN")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    knn = KNeighborsClassifier(n_neighbors=n_neighbours)
    X_train, X_test, y_train, y_test = get_train_test(test_size)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"{accuracy = }")

def get_knn():
    print("loading KNN")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=4)
    from read_complete import get_ori_data
    complete = get_ori_data()
    points = complete["points"]
    labels = complete["labels"]
    print("training")
    knn.fit(points, labels)
    return knn

def gen_grid(arr,range_=(-3,3,10)):
    arr = arr.astype(np.float64)
    def gen_probability(range_):
        start,end,amount = range_
        offset = (end-start)/(amount-1)/2
        newRange = [-float("inf"),] + list(np.linspace(start+offset,end+offset,amount))[:-1] + [float("inf"),]
        from scipy.stats import norm
        probability = []
        for i in range(len(newRange)-1):
            probability.append(norm.cdf(newRange[i+1])-norm.cdf(newRange[i]))
        return probability
    from itertools import product
    nan_indices = np.where(np.isnan(arr))[0]
    grid = np.array(list(product(np.linspace(*range_), repeat=len(nan_indices))))
    probability = np.prod(np.array(list(product(gen_probability(range_), repeat=len(nan_indices)))),axis=1)
    grids = np.full((len(grid), len(arr)), np.nan)
    for i, index in enumerate(nan_indices):
        grids[:, index] = grid[:, i]
    valid_indices = np.where(~np.isnan(arr))[0]
    for i in valid_indices:
        grids[:, i] = arr[i]
    return grids,probability

def pred_by_grid(arrs,range_=(-3,3,10)):
    from ENCODE.encode import tensor2points
    from AE.training import get_trained_model
    from collections import defaultdict
    import torch
    predLabels = []
    predX = []
    n = len(list(arrs))
    ae = get_trained_model()
    knn = get_knn()
    for idx,arr in enumerate(arrs):
        grids,probs = gen_grid(arr,range_)
        encoded = tensor2points(ae(torch.tensor(grids).to(torch.float)))
        print(f"predicting {idx+1}/{n}")
        labels = knn.predict(encoded)
        probLabel = defaultdict(float)
        probGrid = defaultdict(list)
        for label,prob,grid in zip(labels,probs,grids):
            probLabel[label] += prob
            probGrid[label].append(grid)
        maxProbGrid = -1
        for k,v in probGrid.items():
            if probLabel[k]>maxProbGrid:
                maxProbGrid = probLabel[k]
                mostGrid = v
        mostGrid = np.array(mostGrid).mean(axis=0)
        probLabel = probLabel.items()
        mostLabel = max(probLabel,key=lambda x:x[1])[0]
        predLabels.append(mostLabel)
        predX.append(mostGrid)
    return np.array(predX),np.array(predLabels)
    

if __name__ == "__main__":
    from classifier.read_incomplete import get_uni_inc,get_bi_inc
    import pandas as pd
    
    data1 = get_uni_inc()
    pk1 = data1["PK"]
    data1 = data1.drop("PK",axis=1)
    predX1,labels1 = pred_by_grid(data1.values,range_=(-3,3,100))
    data1 = pd.DataFrame(predX1,columns=data1.columns)
    data1["label"] = labels1
    data1["PK"] = pk1.values
    data1.to_csv("codes/classifier/data/uni_inc.csv",index=False)
    
    data2 = get_bi_inc()
    pk2 = data2["PK"]
    data2 = data2.drop("PK",axis=1)
    predX2,labels2 = pred_by_grid(data2.values,range_=(-3,3,100))
    data2 = pd.DataFrame(predX2,columns=data2.columns)
    data2["label"] = labels2
    data2["PK"] = pk2.values
    data2.to_csv("codes/classifier/data/bi_inc.csv",index=False)