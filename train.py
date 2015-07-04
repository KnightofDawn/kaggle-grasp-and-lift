#!/usr/bin/env python

import numpy as np

from lasagne import layers
from sklearn.metrics import roc_auc_score

import csp
import batching
import iter_funcs
import utils

from convnet import build_model


def main():
    subj_id = 10
    window_size = 100
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
    print('there are %d windows for training' % (len(train_slices)))
    print('there are %d windows for validation' % (len(valid_slices)))

    print('computing common spatial patterns...')
    csp_transform = csp.compute_transform(subj_id, nfilters=4)
    print('csp transform shape = %r' % (csp_transform.shape,))
    train_data = csp.apply_transform(train_data, csp_transform, nwin=250)
    valid_data = csp.apply_transform(valid_data, csp_transform, nwin=250)

    train_data = [data.astype(np.float32) for data in train_data]
    valid_data = [data.astype(np.float32) for data in valid_data]
    train_events = [events.astype(np.int32) for events in train_events]
    valid_events = [events.astype(np.int32) for events in valid_events]

    batch_size = 16
    # remember to change the number of channels when there is csp!!!
    num_channels = 4
    num_actions = 6
    print('building model...')
    #l_out = build_model(batch_size, num_channels, window_size, num_actions)
    l_out = build_model(None, num_channels, window_size, num_actions)

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
    
    for epoch in range(max_epochs):
        print('epoch: %d' % (epoch))
        print('  training...')
        train_losses, training_outputs = [], []
        num_batches = len(train_slices) / batch_size + 1
        for i, (Xb, yb) in enumerate(batching.batch_iterator(batch_size,
                                                        train_slices,
                                                        train_data,
                                                        train_events)):
            if i < 70000:
                continue
            train_loss, train_output = train_iter(Xb, yb)
            if i % 10000 == 0:
                print('    processing training minibatch %d of %d with loss %.6f...' %
                      (i, num_batches, train_loss))
            train_losses.append(train_loss)
            for v in train_output:
                training_outputs.append(v)
        avg_train_loss = np.mean(train_losses)
        training_outputs = np.hstack(training_outputs)
        # yb needs to be all labels in training_data
        train_roc = roc_auc_score(yb, training_outputs)
        print('    train loss: %.6f' % (avg_train_loss))
        print('    train roc:  %.6f' % (train_roc))

        print('validation...')
        valid_losses, valid_outputs = [], []
        num_batches = len(valid_slices) / batch_size + 1
        for i, (Xb, yb) in enumerate(batching.batch_iterator(batch_size,
                                                             valid_slices,
                                                             valid_data,
                                                             valid_events)):
            if i % 10000 == 0:
                print('    processing validation minibatch %d of %d...' %
                      (i, num_batches))
            valid_loss, valid_output = valid_iter(Xb, yb)
            valid_losses.append(valid_loss)
            for v in valid_output:
                valid_outputs.append(v)
        avg_valid_loss = np.mean(valid_losses)
        valid_outputs = np.hstack(valid_outputs)
        valid_roc = roc_auc_score(yb, valid_outputs)
        print('    valid loss: %.6f' % (avg_valid_loss))
        print('    valid roc:  %.6f' % (valid_roc))


if __name__ == '__main__':
    main()
