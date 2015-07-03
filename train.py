#!/usr/bin/env python

import numpy as np

from lasagne import layers
from sklearn.metrics import roc_auc_score

import batching
import iter_funcs
import utils

from convnet import build_model


def main():
    subj_id = 10
    window_size = 1000
    print('loading time series for subject %d...' % (subj_id))
    data_list, events_list = utils.load_subject(subj_id)

    print('creating train and validation sets...')
    train_data, train_events, valid_data, valid_events, test_data = \
        utils.split_train_test_data(data_list, events_list,
                                    val_size=2, rand=False)
    print('using %d time series for training' % (len(train_data)))
    print('using %d time series for validation' % (len(valid_data)))

    print('creating fixed-size time-windows of size %d' % (window_size))
    train_slices = batching.get_permuted_windows(train_data, window_size)
    valid_slices = batching.get_permuted_windows(valid_data, window_size)

    print('building model...')
    l_out = build_model(8, 32, window_size, 6)

    all_layers = layers.get_all_layers(l_out)
    print('this network has %d learnable parameters' %
          (layers.count_params(l_out)))
    for layer in all_layers:
        print('Layer %s has output shape %r' %
              (layer.name, layer.output_shape))

    max_epochs = 5
    lr, mntm = 0.001, 0.9
    print('compiling theano functions...')
    train_iter = iter_funcs.create_iter_funcs_train(lr, mntm, l_out)
    valid_iter = iter_funcs.create_iter_funcs_valid(l_out)

    data = np.zeros((8, 32, window_size), dtype=np.float32)
    labels = np.random.randint(0, 2, size=(8, 6)).astype(np.int32)

    #train_loss, train_output = train_iter(data, labels)
    valid_loss, valid_output = valid_iter(data, labels)
    valid_roc = roc_auc_score(labels, valid_output)
    print('valid loss: %.6f' % (valid_loss))
    print('valid roc:  %.6f' % (valid_roc))

if __name__ == '__main__':
    main()
