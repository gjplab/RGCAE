from sklearn.cluster import KMeans, spectral_clustering
from sklearn.neighbors import KNeighborsClassifier
from . import metrics
import numpy as np

def classify(H_train, H_test, gt_train, gt_test, count=1):
    pred_all = []
    for i in range(count):
        # 分类任务
        knn = KNeighborsClassifier()
        knn.fit(H_train, gt_train)
        pred = knn.predict(H_test)
        pred_all.append(pred)
    gt = np.reshape(gt_test, np.shape(pred))
    if np.min(gt) == 1:
        gt -= 1
    acc_avg = get_avg_acc(gt, pred_all, count)
    return acc_avg

def get_avg_acc(y_true, y_pred, count):
    acc_array = np.zeros(count)
    for i in range(count):
        acc_array[i] = metrics.acc(y_true, y_pred[i])
    acc_avg = acc_array.mean()
    return acc_avg



