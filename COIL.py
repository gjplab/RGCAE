from utils.Dataset import Dataset
from model_3views import model
from utils.print_result import get_result
from utils.parse_option_3views import parse_option
from sklearn.model_selection import StratifiedShuffleSplit
import os
import numpy as np
import timeit
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    name = 'COIL20_3views'
    data = Dataset('COIL20_3views')

    args = parse_option()
    x1, x2, x3, gt = data.load_data()
    x1 = data.normalize(x1, 0)
    x2 = data.normalize(x2, 0)
    x3 = data.normalize(x3, 0)
    # construct KNN graph
    x1_graph, x2_graph, x3_graph = data.load_graph1(x1, x2, x3, args.k, args.gamma)

    H, gt= model(x1, x2, x3, x1_graph, x2_graph, x3_graph, gt,
                  args.dims_ae1, args.dims_ae2, args.dims_ae3, args.dims_ae1_graph, args.dims_ae2_graph, args.dims_ae3_graph,
                  args.act, args.batch_size, args.epochs_pre, args.epochs_fine,
                  args.lr_pre, args.lr_fine, args.lr_joint, args.alpha, args.beta)
    acc_all = []
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
    for train_index, test_index in split.split(H, gt):
        H_train, H_test = H[train_index], H[test_index]
        H_train = H_train[:, :H.shape[1]]
        H_test = H_test[:, :H.shape[1]]
        gt_train, gt_test = gt[train_index], gt[test_index]
        acc = get_result(H_train, H_test, gt_train, gt_test)
        acc_all.append(round(acc, 4))
    acc_avg = round(np.mean(acc_all), 4)
    acc_std = round(np.std(acc_all), 4)
    print(acc_all)
    print('classifying h      : acc_avg = {:.4f}  acc_std = {:.4f} '.format(acc_avg, acc_std))
