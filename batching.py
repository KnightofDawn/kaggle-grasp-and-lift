#!/usr/bin/env python

import numpy as np
import random


# return slices that chunk a time-series into windows
# of size window_size
def get_series_window_slices(num_datapoints, window_size):
    slices = []
    num_windows = num_datapoints - window_size + 1
    for i in range(0, num_windows, 1):
        slices.append(slice(i, i + window_size))

    return slices


# permute the time-windows of all time-series to
# give a shuffled list of time-windows over all
# time series
def get_permuted_windows(series_list, window_size, rand=True):
    series_slices = []
    for i, series in enumerate(series_list):
        slices = get_series_window_slices(series.shape[1],
                                          window_size)
        # need to mark each slice with the series it came from
        for s in slices:
            series_slices.append((i, s))

    # wish to iterate over the windows in random order for training
    if rand:
        random.shuffle(series_slices)
    return series_slices


def normalize_window(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    return (X - X_min) / (X_max - X_min)


# splits the list of window indices and slices into batches
# and grabs the fixed-length windows from the corresponding
# slice from that time-series
def batch_iterator(bs, W, X, y=None, normalize_window=False):
    if not W:
        raise StopIteration
    window_size = W[0][1].stop - W[0][1].start
    # total number of batches for this data set and batch size
    N = (len(W) + bs - 1) / bs
    for i in range(N):
        Wb = W[i * bs:(i + 1) * bs]

        X_batch_list, y_batch_list = [], []
        # index: which time series to take the window from
        # s:     the slice to take from that time series
        for index, s in Wb:
            if normalize_window:
                X_window = normalize_window(X[index][:, s])
            else:
                X_window = X[index][:, s]
            X_batch_list.append(X_window)
            if y is not None:
                y_batch_list.append(y[index][:, s][:, -1])

        # reshape to (batch_size, num_channels, window_size)
        X_batch = np.vstack(X_batch_list).reshape(-1,
                                                  X[0].shape[0], window_size)
        if y_batch_list:
            y_batch = np.vstack(y_batch_list)
        else:
            y_batch = None

        yield X_batch, y_batch


def compute_geometric_mean(mat_list):
    log_mat_sum = np.sum(np.log(np.array(mat_list)), axis=0)
    log_mat_sum /= len(mat_list)
    geom_mean = np.power(np.e, log_mat_sum)

    return geom_mean


if __name__ == '__main__':
    data = np.arange(5 * 10).reshape(5, 10)
    slices = get_series_window_slices(data.shape[1], 3)
    print data
    print slices
    print('slicing:')
    for s in slices:
        print data[:, s]
