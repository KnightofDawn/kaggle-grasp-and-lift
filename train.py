#!/usr/bin/env python

import cPickle as pickle
import numpy as np
import theano

from lasagne import layers
from scipy.signal import butter, lfilter
from sklearn.metrics import roc_auc_score
from time import time

import csp
import batching
import iter_funcs
import utils

from convnet import build_model


def main():
    subj_id = 1
    window_size = 100
    weights_file = 'data/nets/subj1_weights.pickle'
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

    train_data = [1e-6 * data for data in train_data]
    valid_data = [1e-6 * data for data in valid_data]
    b, a = butter(5, np.array([7, 30]) / 250., btype='bandpass')
    train_data = [lfilter(b, a, data) for data in train_data]
    valid_data = [lfilter(b, a, data) for data in valid_data]
    #print train_data[7][:, 0]
    print('computing common spatial patterns...')
    csp_transform = csp.compute_transform(subj_id, nfilters=4)
    #print csp_transform
    print('csp transform shape = %r' % (csp_transform.shape,))
    train_data = csp.apply_transform(train_data, csp_transform, nwin=250)
    valid_data = csp.apply_transform(valid_data, csp_transform, nwin=250)

    #print train_data[7][:, 0]
    train_data = [data.astype(np.float32) for data in train_data]
    valid_data = [data.astype(np.float32) for data in valid_data]

    train_mean = np.mean(np.hstack(train_data), axis=1).reshape(-1, 1)
    train_data = [data - train_mean for data in train_data]
    train_std = np.std(np.hstack(train_data), axis=1).reshape(-1, 1)
    train_data = [data / train_std for data in train_data]

    valid_data = [data - train_mean for data in valid_data]
    valid_data = [data / train_std for data in valid_data]

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

    max_epochs = 50
    lr = theano.shared(np.cast['float32'](0.001))
    mntm = 0.9
    patience = 10
    print('compiling theano functions...')
    train_iter = iter_funcs.create_iter_funcs_train(lr, mntm, l_out)
    valid_iter = iter_funcs.create_iter_funcs_valid(l_out)

    best_weights = None
    best_valid_loss = np.inf
    best_epoch = 0
    try:
        for epoch in range(max_epochs):
            print('epoch: %d' % (epoch))
            print('  training...')
            train_losses, training_outputs, training_inputs = [], [], []
            num_batches = len(train_slices) / batch_size + 1
            t_train_start = time()
            for i, (Xb, yb) in enumerate(batching.batch_iterator(batch_size,
                                                                 train_slices,
                                                                 train_data,
                                                                 train_events)):
                train_loss, train_output = train_iter(Xb, yb)
                if (i + 1) % 10000 == 0:
                    print('    processed training minibatch %d of %d...' %
                          (i, num_batches))
                train_losses.append(train_loss)
                assert len(yb) == len(train_output)
                for input, output in zip(yb, train_output):
                    training_inputs.append(input)
                    training_outputs.append(output)
            avg_train_loss = np.mean(train_losses)

            training_inputs = np.hstack(training_inputs)
            training_outputs = np.hstack(training_outputs)
            train_roc = roc_auc_score(training_inputs, training_outputs)

            train_duration = time() - t_train_start
            print('    train loss: %.6f' % (avg_train_loss))
            print('    train roc:  %.6f' % (train_roc))
            print('    duration:   %.2f s' % (train_duration))

            print('validation...')
            valid_losses, valid_outputs, valid_inputs = [], [], []
            num_batches = len(valid_slices) / batch_size + 1
            t_valid_start = time()
            for i, (Xb, yb) in enumerate(batching.batch_iterator(batch_size,
                                                                 valid_slices,
                                                                 valid_data,
                                                                 valid_events)):
                valid_loss, valid_output = valid_iter(Xb, yb)
                if (i + 1) % 10000 == 0:
                    print('    processing validation minibatch %d of %d...' %
                          (i, num_batches))
                valid_losses.append(valid_loss)
                assert len(yb) == len(valid_output)
                for input, output in zip(yb, valid_output):
                    valid_inputs.append(input)
                    valid_outputs.append(output)
            avg_valid_loss = np.mean(valid_losses)
            valid_inputs = np.hstack(valid_inputs)
            valid_outputs = np.hstack(valid_outputs)
            valid_roc = roc_auc_score(valid_inputs, valid_outputs)
            valid_duration = time() - t_valid_start
            print('    valid loss: %.6f' % (avg_valid_loss))
            print('    valid roc:  %.6f' % (valid_roc))
            print('    duration:   %.2f s' % (valid_duration))

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_weights = layers.get_all_param_values(l_out)

            if epoch > best_epoch + patience:
                best_epoch = epoch
                new_lr = 0.5 * lr.get_value()
                lr.set_value(np.cast['float32'](new_lr))
                print('setting learning rate to %.6f' % (new_lr))

    except KeyboardInterrupt:
        print('caught Ctrl-C, stopping training...')

    with open(weights_file, 'wb') as ofile:
        print('saving best weights to %s' % (weights_file))
        pickle.dump(best_weights, ofile, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
