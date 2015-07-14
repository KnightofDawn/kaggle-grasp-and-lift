#!/usr/bin/env python

import numpy as np
import pandas as pd

from time import time, strftime


def get_current_time():
    return strftime('%Y-%m-%d_%H:%M:%S')


def load_subject_train(subj_id):
    data_list, events_list = [], []
    for series_id in range(1, 9):
        fname_data = 'data/train/subj%d_series%d_data.csv' % (
            subj_id, series_id)
        fname_events = 'data/train/subj%d_series%d_events.csv' % (
            subj_id, series_id)

        data = pd.read_csv(fname_data)
        events = pd.read_csv(fname_events)

        channel_names = data.columns[1:]
        data_list.append(data[channel_names].values.T)

        event_names = events.columns[1:]
        events_list.append(events[event_names].values.T.astype(np.int32))

    return data_list, events_list


def load_subject_test(subj_id):
    data_list, ids_list = [], []
    for series_id in range(9, 11):
        fname_data = 'data/test/subj%d_series%d_data.csv' % (
            subj_id, series_id)

        data = pd.read_csv(fname_data)
        channel_names = data.columns[1:]
        id_col = data.columns[0]

        data_list.append(data[channel_names].values.T)
        ids_list.append(data[id_col].values.T)

    return data_list, ids_list


def preprocess(subj_id, train_data, test_data):
    train_data = [data.astype(np.float32) for data in train_data]
    test_data = [data.astype(np.float32) for data in test_data]

    print('normalizing...')
    # subtract the mean from all time series
    train_mean = np.mean(np.hstack(train_data), axis=1).reshape(-1, 1)
    train_data = [data - train_mean for data in train_data]

    # divide all time series by the standard deviation
    train_std = np.std(np.hstack(train_data), axis=1).reshape(-1, 1)
    train_data = [data / train_std for data in train_data]

    # apply the same transform to valid/test
    test_data = [data - train_mean for data in test_data]
    test_data = [data / train_std for data in test_data]

    return train_data, test_data


# split the time series into training, validation, and test
def split_train_test_data(data_list, events_list, val_size=2, rand=False):
    # randomly choose val_size time series for validation
    if rand:
        val_ind = np.random.choice(8, size=val_size, replace=False)
    # just use the last two time series for validation
    else:
        val_ind = np.arange(8 - val_size, 8)

    train_data, valid_data = [], []
    train_events, valid_events = [], []
    # separate the time series into training and validation
    for i in range(8):
        if i not in val_ind:
            train_data.append(data_list[i])
            train_events.append(events_list[i])
        else:
            valid_data.append(data_list[i])
            valid_events.append(events_list[i])

    return train_data, train_events, valid_data, valid_events


# time the loading of the h5py files and print the shapes of the time series
# arrays
def verify_data():
    for subject in range(1, 13):
        # load the training data for each subject, time it, and print the shape
        t0 = time()
        data_list, events_list = load_subject_train(subject)
        print('loaded training data for subject %d in %.2f s' %
              (subject, time() - t0))
        print('verifying training data for subject %d...' % (subject))
        for i, (data, events) in enumerate(zip(data_list, events_list),
                                           start=1):
            print('  series %d:' % (i))
            print('    data.shape = %r' % (data.shape,))
            print('    events.shape = %r' % (events.shape,))

        # load the test data for each subject, time it, and print the shape
        t0 = time()
        data_list, ids_list = load_subject_test(subject)
        print('loaded test data for subject %d in %.2f s' %
              (subject, time() - t0))
        print('verifying test data for subject %d...' % (subject))
        for i, (data, ids) in enumerate(zip(data_list, ids_list),
                                        start=1):
            print('  series %d:' % (i))
            print('    data.shape = %r' % (data.shape,))
            print('    ids.shape = %r' % (ids.shape))


def main():
    #generate_data()
    verify_data()


if __name__ == '__main__':
    main()
