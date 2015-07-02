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
def get_permuted_windows(series_list, window_size):
    series_slices = []
    for i, series in enumerate(series_list):
        slices = get_series_window_slices(series.shape[1],

                                          window_size)
        # need to mark each slice with the series it came from
        for s in slices:
            series_slices.append((i, s))

    # wish to iterate over the windows in random order
    random.shuffle(series_slices)
    return series_slices


if __name__ == '__main__':
    data = np.arange(5 * 10).reshape(5, 10)
    slices = get_series_window_slices(data.shape[1], 3)
    print data
    print slices
    print('slicing:')
    for s in slices:
        print data[:, s]
