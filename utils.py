#!/usr/bin/env python

import cPickle as pickle
import numpy as np

TRAIN_FILE = 'data/processed/train.pickle'
TEST_FILE = 'data/processed/test.pickle'


# read the data from a single series given the filename
def read_data_series(filename):
    datapoints = []
    with open(filename, 'r') as ifile:
        for i, line in enumerate(ifile, start=1):
            # ignore the header
            if i == 1:
                continue
            dtext = line.strip().split(',')
            dvalues = map(int, dtext[1:])
            datapoints.append(dvalues)

    # [sensors (32), time (N)]
    data = np.vstack(datapoints).transpose()

    assert data.shape[0] == 32, 'need 32 sensor readings per timeframe'
    return data


# read the events from a single series given the filename
def read_events_series(filename):
    eventpoints = []
    with open(filename, 'r') as ifile:
        for i, line in enumerate(ifile, start=1):
            if i == 1:
                continue
            etext = line.strip().split(',')
            evalues = map(int, etext[1:])
            eventpoints.append(evalues)

    # [actions (6), time (N)]
    events = np.vstack(eventpoints).transpose()

    assert events.shape[0] == 6, 'need 6 labels per timeframe'
    return events


# read the data for a range of subjects and series
def read_data(subj_range, series_range, train):
    data_subjects, events_subjects = [], []
    for subj_id in subj_range:
        data_series, events_series = [], []
        print('reading data for subject %d...' % (subj_id))
        for series_id in series_range:
            print('  series %d...' % (series_id))
            if train:
                data_file = 'data/train/subj%d_series%d_data.csv' % (subj_id, series_id)
                events_file = 'data/train/subj%d_series%d_events.csv' % (subj_id, series_id)

                data = read_data_series(data_file)
                events = read_events_series(events_file)
                assert data.shape[1] == events.shape[1], 'need an equal number of timeframes'

            else:
                data_file = 'data/test/subj%d_series%d_data.csv' % (subj_id, series_id)
                events_file = 'data/test/subj%d_series%d_events.csv' % (subj_id, series_id)

                data = read_data_series(data_file)
                events = None  # no events for test data

            data_series.append(data)
            if events is not None:
                events_series.append(events)

        data_subjects.append(data_series)
        if events_series:
            events_subjects.append(events_series)

    assert type(data_subjects[0][0]) == np.ndarray, 'list elements should be numpy arrays'
    if events_subjects:
        assert type(events_subjects[0][0]) == np.ndarray, 'list elements should be numpy arrays'

    return data_subjects, events_subjects


# read the training and test data from disk
def load_data():
    print('getting training data...')
    print('loading training data from %s...' % (TRAIN_FILE))
    with open(TRAIN_FILE, 'rb') as ofile:
        train_data, train_events = pickle.load(ofile)

    print('getting test data...')
    print('loading test data from %s...' % (TEST_FILE))
    with open(TEST_FILE, 'rb') as ofile:
        test_data = pickle.load(ofile)

    return train_data, train_events, test_data


# dump the training and test data to disk
def dump_data():
    # there are 12 subjects
    subj_range = range(1, 13)
    # series [1...8] are training data
    train_range = range(1, 9)
    # series [9, 10] are test data
    test_range = range(9, 11)

    print('getting training data...')
    train_data, train_events = read_data(subj_range, train_range, train=True)
    print('writing training data to %s...' % (TRAIN_FILE))
    with open(TRAIN_FILE, 'wb') as ofile:
        pickle.dump((train_data, train_events), ofile, protocol=pickle.HIGHEST_PROTOCOL)

    print('getting test data...')
    test_data, _ = read_data(subj_range, test_range, train=False)
    print('writing test data to %s...' % (TEST_FILE))
    with open(TEST_FILE, 'wb') as ofile:
        pickle.dump(test_data, ofile, protocol=pickle.HIGHEST_PROTOCOL)

    
def main():
    #dump_data()
    train_data, train_events, test_data = load_data()


if __name__ == '__main__':
    main()
