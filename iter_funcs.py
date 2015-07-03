import lasagne
import theano
import theano.tensor as T

from lasagne import layers


def create_iter_funcs_train(lr, mntm, l_out):
    X = T.tensor4('x')
    y = T.itensor4('y')
    X_batch = T.tensor4('x_batch')
    y_batch = T.itensor4('y_batch')

    output_train = layers.get_output(l_out, X_batch)
    train_loss = T.mean(T.nnet.binary_crossentropy(output_train, y_batch))

    all_params = layers.get_all_params(l_out)
    updates = lasagne.updates.nesterov_momentum(
        train_loss, all_params, lr, mntm)

    train_iter = theano.function(
        inputs=[theano.Param(X_batch),
                theano.Param(y_batch)],
        outputs=[train_loss],
        updates=updates,
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return train_iter


def create_iter_funcs_valid(l_out):
    X = T.tensor4('x')
    y = T.itensor4('y')
    X_batch = T.tensor4('x_batch')
    y_batch = T.itensor4('y_batch')

    output_valid = layers.get_output(l_out, X_batch, deterministic=True)
    valid_loss = T.mean(T.nnet.categorical_crossentropy(output_valid, y_batch))

    valid_iter = theano.function(
        inputs=[theano.Param(X_batch),
                theano.Param(y_batch)],
        outputs=[valid_loss, output_valid],
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return valid_iter
