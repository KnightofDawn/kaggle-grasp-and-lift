#!/usr/bin/env python

import batching
import utils


def main():
    subj_id = 10
    window_size = 1000
    print('loading time series for subject %d...' % (subj_id))
    data_list, events_list = utils.load_subject(subj_id)

    train_data, train_events, valid_data, valid_events, test_data = \
        utils.split_train_test_data(data_list, events_list,
                                    val_size=2, rand=False)
    train_slices = batching.get_permuted_windows(train_data, window_size)
    valid_slices = batching.get_permuted_windows(valid_data, window_size)

    print('using %d time series for training' % (len(train_data)))
    print('using %d time series for validation' % (len(valid_data)))

if __name__ == '__main__':
    main()
