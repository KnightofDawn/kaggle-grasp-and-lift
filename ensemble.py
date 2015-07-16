#!/usr/bin/env python

import cPickle as pickle
import numpy as np

import batching
import iter_funcs
import utils

from lasagne import layers
from os.path import join, basename
from sklearn.metrics import roc_auc_score, log_loss
from time import time


def ensemble(subj_id, models, weights_files, window_sizes,
             do_train=True, do_valid=True):

    print('loading time series for subject %d...' % (subj_id))
    data_list, events_list = utils.load_subject_train(subj_id)

    print('creating train and validation sets...')
    train_data, train_events, valid_data, valid_events = \
        utils.split_train_test_data(data_list, events_list,
                                    val_size=2, rand=False)
    print('using %d time series for training' % (len(train_data)))
    print('using %d time series for validation' % (len(valid_data)))

    train_data, valid_data = \
        utils.preprocess(subj_id, train_data, valid_data)

    batch_size = 4096
    num_channels = 32
    num_actions = 6

    ensemble_predictions_train, ensemble_predictions_valid = [], []
    for weights, model, window_size in zip(weights_files, models, window_sizes):
        print('creating fixed-size time-windows of size %d' % (window_size))
        # the training windows should be in random order
        train_slices = batching.get_permuted_windows(train_data, window_size, rand=True)
        valid_slices = batching.get_permuted_windows(valid_data, window_size, rand=True)
        print('there are %d windows for training' % (len(train_slices)))
        print('there are %d windows for validation' % (len(valid_slices)))

        build_model = model.build_model
        print('building model...')
        l_out = build_model(None, num_channels,
                            window_size, num_actions)

        print('loading model weights from %s' % (weights))
        with open(weights, 'rb') as ifile:
            src_layers = pickle.load(ifile)
        dst_layers = layers.get_all_params(l_out)
        for i, (src_weights, dst_layer) in enumerate(
                zip(src_layers, dst_layers)):
            dst_layer.set_value(src_weights)

        print('compiling theano functions...')
        # only need the validation iter because we're not training
        valid_iter = iter_funcs.create_iter_funcs_valid(l_out)

        if do_train:
            train_losses, training_outputs, training_inputs = [], [], []
            num_batches = (len(train_slices) + batch_size - 1) / batch_size
            t_train_start = time()
            print('  predicting on training data...')
            for i, (Xb, yb) in enumerate(
                batching.batch_iterator(batch_size,
                                        train_slices,
                                        train_data,
                                        train_events,
                                        window_norm=True)):
                train_loss, train_output = \
                    valid_iter(Xb, yb)
                if np.isnan(train_loss):
                    print('nan loss encountered in minibatch %d' % (i))
                    continue
                if (i + 1) % 100 == 0:
                    print('    processed training minibatch %d of %d...' %
                          (i + 1, num_batches))

                train_losses.append(train_loss)
                assert len(yb) == len(train_output)
                for input, output in zip(yb, train_output):
                    training_inputs.append(input)
                    training_outputs.append(output)
            avg_train_loss = np.mean(train_losses)

            training_inputs = np.hstack(training_inputs)
            training_outputs = np.hstack(training_outputs)
            train_roc = roc_auc_score(training_inputs, training_outputs)
            ensemble_predictions_train.append(training_outputs)
            train_duration = time() - t_train_start
            print('    train loss: %.6f' % (avg_train_loss))
            print('    train roc:  %.6f' % (train_roc))
            print('    duration:   %.2f s' % (train_duration))

        if do_valid:
            valid_losses, valid_outputs, valid_inputs = [], [], []
            num_batches = (len(valid_slices) + batch_size - 1) / batch_size
            t_valid_start = time()
            print('  predicting on validation data...')
            for i, (Xb, yb) in enumerate(
                batching.batch_iterator(batch_size,
                                        valid_slices,
                                        valid_data,
                                        valid_events,
                                        window_norm=True)):
                valid_loss, valid_output = \
                    valid_iter(Xb, yb)
                if np.isnan(valid_loss):
                    print('nan loss encountered in minibatch %d' % (i))
                    continue
                if (i + 1) % 100 == 0:
                    print('    processed validation minibatch %d of %d...' %
                          (i + 1, num_batches))

                valid_losses.append(valid_loss)
                assert len(yb) == len(valid_output)
                for input, output in zip(yb, valid_output):
                    valid_inputs.append(input)
                    valid_outputs.append(output)
            avg_valid_loss = np.mean(valid_losses)

            padding = np.zeros((window_size - 1, 6), dtype=np.int32)
            for pad in padding:
                valid_inputs.append(pad)
                valid_outputs.append(pad)

            valid_inputs = np.vstack(valid_inputs)
            valid_outputs = np.vstack(valid_outputs)
            print valid_outputs.shape
            valid_roc = roc_auc_score(valid_inputs, valid_outputs)

            ensemble_predictions_valid.append(valid_outputs)
            valid_duration = time() - t_valid_start
            print('    valid loss: %.6f' % (avg_valid_loss))
            print('    valid roc:  %.6f' % (valid_roc))
            print('    duration:   %.2f s' % (valid_duration))

    print('ensemble results for subject %d' % (subj_id))
    if do_train:
        avg_predictions_train = batching.compute_geometric_mean(ensemble_predictions_train)
        train_loss = log_loss(train_events, avg_predictions_train)
        train_roc = roc_auc_score(train_events, avg_predictions_train)
        print('    train loss: %.6f' % (train_loss))
        print('    train roc:  %.6f' % (train_roc))

    if do_valid:
        for a in ensemble_predictions_valid:
            print type(a), a.shape
        avg_predictions_valid = batching.compute_geometric_mean(ensemble_predictions_valid)
        valid_loss = log_loss(valid_events, avg_predictions_valid)
        valid_roc = roc_auc_score(valid_events, avg_predictions_valid)
        print('    valid loss: %.6f' % (valid_loss))
        print('    valid roc:  %.6f' % (valid_roc))


def main():
    do_train = False
    do_valid = True
    root_dir = join('data', 'nets')
    weights_file_patterns = ['subj%d_weights_deep_nocsp_wn_extra.pickle',
                             'subj%d_weights_deep_nocsp_wn_regions.pickle']
    weights_file_patterns = [join(root_dir, p) for p in weights_file_patterns]
    window_sizes = [2000, 1600]
    import convnet_deep_drop
    import convnet_regions
    models = [convnet_deep_drop, convnet_regions]
    subjects = range(6, 7)
    for subj_id in subjects:
        weights_files = [pattern % subj_id for pattern in weights_file_patterns]
        print('creating ensemble for subject %d using weights:' % (subj_id))
        assert len(models) == len(weights_files) == len(window_sizes)
        for m, wf, ws in zip(models, weights_files, window_sizes):
            print('  %s (%d): %s' % (m.__name__, ws, basename(wf)))
        ensemble(subj_id, models, weights_files, window_sizes,
                 do_train=do_train, do_valid=do_valid)


if __name__ == '__main__':
    main()
