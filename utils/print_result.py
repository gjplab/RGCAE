from utils.classify import classify
import warnings

warnings.filterwarnings('ignore')


def get_result(H_train, H_test, gt_train, gt_test, count=1):
    acc_avg = classify(H_train, H_test, gt_train, gt_test, count=count)
    return acc_avg*100
