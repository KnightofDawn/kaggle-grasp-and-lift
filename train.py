#!/usr/bin/env python

import cPickle as pickle
import numpy as np
import theano
import sys

from lasagne import layers
from os.path import join
from sklearn.metrics import roc_auc_score
from time import time

import batching
import iter_funcs
import utils

#from convnet import build_model
#from convnet_small import build_model
#from convnet_deep import build_model
#from convnet_deep_drop import build_model
#from convnet_deep_scale import build_model
#from convnet_regions import build_model
#from convnet_very_deep_drop import build_model
from convnet_regions_two_equal import build_model


def train_model(subj_id, window_size, subsample, max_epochs, patience):
    root_dir = join('data', 'nets')
    # the file from which to load pre-trained weights
    #init_file = join(root_dir,
    #                 'subj%d_weights_deep_nocsp_wn.pickle' % (
    #                     subj_id))
    init_file = None
    # the file to which the learned weights will be written
    weights_file = join(root_dir,
                        'subj%d_weights_deep_nocsp_wn_two_equal_dozer.pickle' % (
                            subj_id))
    print('loading time series for subject %d...' % (subj_id))
    data_list, events_list = utils.load_subject_train(subj_id)

    print('creating train and validation sets...')
    train_data, train_events, valid_data, valid_events = \
        utils.split_train_test_data(data_list, events_list,
                                    val_size=2, rand=False)
    print('using %d time series for training' % (len(train_data)))
    print('using %d time series for validation' % (len(valid_data)))

    print('creating fixed-size time-windows of size %d' % (window_size))
    # the training windows should be in random order
    train_slices = batching.get_permuted_windows(train_data, window_size,
                                                 rand=True)
    valid_slices = batching.get_permuted_windows(valid_data, window_size,
                                                 rand=True)
    print('there are %d windows for training' % (len(train_slices)))
    print('there are %d windows for validation' % (len(valid_slices)))

    batch_size = 64
    num_channels = 32
    num_actions = 6
    train_data, valid_data = \
        utils.preprocess(subj_id, train_data, valid_data)

    print('building model...')
    l_out = build_model(None, num_channels,
                        window_size, num_actions)

    all_layers = layers.get_all_layers(l_out)
    print('this network has %d learnable parameters' %
          (layers.count_params(l_out)))
    for layer in all_layers:
        print('Layer %s has output shape %r' %
              (layer.name, layer.output_shape))

    if init_file is not None:
        print('loading model weights from %s' % (init_file))
        with open(init_file, 'rb') as ifile:
            src_layers = pickle.load(ifile)
        dst_layers = layers.get_all_params(l_out)
        for i, (src_weights, dst_layer) in enumerate(
                zip(src_layers, dst_layers)):
            print('loading pretrained weights for %s' % (dst_layer.name))
            dst_layer.set_value(src_weights)
    else:
        print('all layers will be trained from random initialization')

    lr = theano.shared(np.cast['float32'](0.001))
    #lr = theano.shared(np.cast['float32'](0.0001))
    mntm = 0.9
    print('compiling theano functions...')
    train_iter = iter_funcs.create_iter_funcs_train(lr, mntm, l_out)
    valid_iter = iter_funcs.create_iter_funcs_valid(l_out)

    best_weights = None
    best_valid_loss = np.inf
    best_epoch = 0
    print('starting training for subject %d at %s' % (
        subj_id, utils.get_current_time()))
    try:
        for epoch in range(max_epochs):
            print('epoch: %d' % (epoch))
            train_losses, training_outputs, training_inputs = [], [], []
            num_batches = (len(train_slices) + batch_size - 1) / batch_size
            t_train_start = time()
            for i, (Xb, yb) in enumerate(
                batching.batch_iterator(batch_size,
                                        train_slices,
                                        train_data,
                                        train_events,
                                        window_norm=False)):
                t_batch_start = time()
                # hack for faster debugging
                #if i < 70000:
                #    continue
                train_loss, train_output = \
                    train_iter(Xb, yb)
                batch_duration = time() - t_batch_start
                if np.isnan(train_loss):
                    print('nan loss encountered in minibatch %d' % (i))
                    continue
                if i % 10 == 0:
                    eta = batch_duration * (num_batches - i)
                    print('  training...  (ETA = %02d:%02d)\r'
                          % divmod(eta, 60)),
                    sys.stdout.flush()
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
            print('')
            print('    train loss: %.6f' % (avg_train_loss))
            print('    train roc:  %.6f' % (train_roc))
            print('    duration:   %.2f s' % (train_duration))

            valid_losses, valid_outputs, valid_inputs = [], [], []
            num_batches = (len(valid_slices) + batch_size - 1) / batch_size
            t_valid_start = time()
            for i, (Xb, yb) in enumerate(
                batching.batch_iterator(batch_size,
                                        valid_slices,
                                        valid_data,
                                        valid_events,
                                        window_norm=False)):
                #augmented_valid_losses, augmented_valid_outputs = [], []
                #for offset in range(0, subsample):
                #    valid_loss, valid_output = \
                #        valid_iter(Xb[:, :, offset::subsample], yb)
                #    augmented_valid_losses.append(valid_loss)
                #    augmented_valid_outputs.append(valid_output)
                #valid_loss = np.mean(augmented_valid_losses)
                #valid_output = batching.compute_geometric_mean(
                #    augmented_valid_outputs)
                t_batch_start = time()
                valid_loss, valid_output = \
                    valid_iter(Xb, yb)
                if np.isnan(valid_loss):
                    print('nan loss encountered in minibatch %d' % (i))
                    continue
                batch_duration = time() - t_batch_start
                if i % 10 == 0:
                    eta = batch_duration * (num_batches - i)
                    print('  validation...  (ETA = %02d:%02d)\r'
                          % divmod(eta, 60)),
                    sys.stdout.flush()

                valid_losses.append(valid_loss)
                assert len(yb) == len(valid_output)
                for input, output in zip(yb, valid_output):
                    valid_inputs.append(input)
                    valid_outputs.append(output)

            # allow training without validation
            if valid_losses:
                avg_valid_loss = np.mean(valid_losses)
                valid_inputs = np.hstack(valid_inputs)
                valid_outputs = np.hstack(valid_outputs)
                valid_roc = roc_auc_score(valid_inputs, valid_outputs)
                valid_duration = time() - t_valid_start
                print('')
                print('    valid loss: %.6f' % (avg_valid_loss))
                print('    valid roc:  %.6f' % (valid_roc))
                print('    duration:   %.2f s' % (valid_duration))
            else:
                print('    no validation...')

            # if we are not doing validation we always want the latest weights
            if not valid_losses:
                best_epoch = epoch
                model_train_loss = avg_train_loss
                model_train_roc = train_roc
                model_valid_roc = -1.
                best_valid_loss = -1.
                best_weights = layers.get_all_param_values(l_out)
            elif avg_valid_loss < best_valid_loss:
                best_epoch = epoch
                model_train_roc = train_roc
                model_valid_roc = valid_roc
                model_train_loss = avg_train_loss
                best_valid_loss = avg_valid_loss
                best_weights = layers.get_all_param_values(l_out)

            if epoch > best_epoch + patience:
                break
                best_epoch = epoch
                new_lr = 0.5 * lr.get_value()
                lr.set_value(np.cast['float32'](new_lr))
                print('setting learning rate to %.6f' % (new_lr))

    except KeyboardInterrupt:
        print('caught Ctrl-C, stopping training...')

    with open(weights_file, 'wb') as ofile:
        print('saving best weights to %s' % (weights_file))
        pickle.dump(best_weights, ofile, protocol=pickle.HIGHEST_PROTOCOL)
    print('finished training for subject %d at %s' % (
        subj_id, utils.get_current_time()))

    return model_train_loss, best_valid_loss, model_train_roc, model_valid_roc


def main():
    subjects = range(2, 13)
    #subjects = range(1, 6)
    # the models that were underfitting
    #subjects = [1, 2, 4, 5]
    #subjects = [1]
    #subjects = [7, 8, 9, 11, 12]
    #subjects = [10]
    #subjects = range(6, 13)
    #subjects = [5, 6, 7, 8, 9, 11, 12, 10]
    #subjects = [2, 3, 4, 1]
    #subjects = [2]
    #subjects = range(6, 7)
    window_size = 2000
    #window_size = 1600
    subsample = 10
    #subsample = 5
    max_epochs = 15
    patience = 1
    #max_epochs = 5
    model_train_losses, model_valid_losses = [], []
    model_train_rocs, model_valid_rocs = [], []
    for subj_id in subjects:
        model_train_loss, model_valid_loss, model_train_roc, model_valid_roc =\
            train_model(subj_id, window_size, subsample, max_epochs, patience)
        print('\n%s subject %d %s' % ('*' * 10, subj_id, '*' * 10))
        print(' model training loss = %.5f' % (model_train_loss))
        print(' model valid loss    = %.5f' % (model_valid_loss))
        print(' model training roc  = %.5f' % (model_train_roc))
        print(' model valid roc     = %.5f' % (model_valid_roc))
        print('%s subject %d %s\n' % ('*' * 10, subj_id, '*' * 10))
        model_train_losses.append(model_train_loss)
        model_valid_losses.append(model_valid_loss)
        model_train_rocs.append(model_train_roc)
        model_valid_rocs.append(model_valid_roc)

    print('average loss over subjects {%s}:' %
          (' '.join([str(s) for s in subjects])))
    print('  training loss:   %.5f' % (np.mean(model_train_losses)))
    print('  validation loss: %.5f' % (np.mean(model_valid_losses)))
    print('  training roc:    %.5f' % (np.mean(model_train_rocs)))
    print('  validation roc:  %.5f' % (np.mean(model_valid_rocs)))


if __name__ == '__main__':
    main()
