#!/usr/bin/env python

import h5py
import numpy as np

from time import time


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
def dump_data(subj_range, series_range, train):
    for subj_id in subj_range:
        print('reading data for subject %d...' % (subj_id))
        for series_id in series_range:
            print('  series %d...' % (series_id))
            if train:
                in_data_file = 'data/train/subj%d_series%d_data.csv' % \
                               (subj_id, series_id)
                in_events_file = 'data/train/subj%d_series%d_events.csv' % \
                                 (subj_id, series_id)

                data = read_data_series(in_data_file)
                events = read_events_series(in_events_file)
                assert data.shape[1] == events.shape[1], \
                    ('need an equal number of timeframes')

                # write the data and events to a single h5py file
                out_file = in_data_file.replace('train', 'processed').replace(
                    '_data', '').replace(
                    'csv', 'h5')
                h5f_data = h5py.File(out_file, 'w')
                h5f_data.create_dataset('data', data=data)
                h5f_data.create_dataset('events', data=events)
                h5f_data.close()
            else:
                in_data_file = 'data/test/subj%d_series%d_data.csv' % \
                               (subj_id, series_id)

                data = read_data_series(in_data_file)

                out_file = in_data_file.replace('test', 'processed').replace(
                    '_data', '').replace(
                    'csv', 'h5')
                h5f_data = h5py.File(out_file, 'w')
                h5f_data.create_dataset('data', data=data)
                h5f_data.close()

    print('done')


# read the training and test data from disk
def load_series(subj_id, series_id):
    # it will be faster to load from h5py than pickle for such large arrays
    in_file = 'data/processed/subj%d_series%d.h5' % (subj_id, series_id)
    h5f_data = h5py.File(in_file, 'r')

    data = h5f_data['data'][:]
    if 'events' in h5f_data:
        events = h5f_data['events'][:]
    else:
        events = None

    h5f_data.close()
    return data, events


# load a range of time series for a given subject
def load_subject(subj_id, series_range):
    data_list, events_list = [], []
    for series_id in series_range:
        data, events = load_series(subj_id, series_id)
        data_list.append(data)
        events_list.append(events)

    return data_list, events_list


# dump the training and test data to disk
def generate_data():
    # there are 12 subjects
    subj_range = range(1, 13)
    # series [1...8] are training data
    train_range = range(1, 9)
    # series [9, 10] are test data
    test_range = range(9, 11)

    print('getting training data...')
    dump_data(subj_range, train_range, train=True)

    print('getting test data...')
    dump_data(subj_range, test_range, train=False)


# time the loading of the h5py files and print the shapes of the time series
# arrays
def verify_data():
    for subject in range(1, 13):
        # load the training data for each subject, time it, and print the shape
        t0 = time()
        data_list, events_list = load_subject(subject, range(1, 9))
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
        data_list, events_list = load_subject(subject, range(9, 11))
        print('loaded test data for subject %d in %.2f s' %
              (subject, time() - t0))
        print('verifying test data for subject %d...' % (subject))
        for i, (data, events) in enumerate(zip(data_list, events_list),
                                           start=1):
            print('  series %d:' % (i))
            print('    data.shape = %r' % (data.shape,))
            print('    events = %r' % (events))


def main():
    #generate_data()
    verify_data()


if __name__ == '__main__':
    main()
