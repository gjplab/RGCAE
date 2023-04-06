import math

def next_batch1(X1, X2, X3, gt, batch_size):
    tot = X1.shape[0]
    total = int(math.ceil(tot / batch_size))  # fix the last batch
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_gt = gt[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2, batch_x3, batch_gt, (i+1))


