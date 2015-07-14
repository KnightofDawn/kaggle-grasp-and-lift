#!/usr/bin/env python

import numpy as np
from lasagne import layers


# given a one-dimensional signal, this layer subsamples the signal
# between the specified time-steps, ensuring that the final time-step
# that is sampled is always the final one in the input signal
class SubsampleLayer(layers.Layer):
    def __init__(self, incoming, window, **kwargs):
        super(SubsampleLayer, self).__init__(incoming, **kwargs)
        if not isinstance(window, slice):
            self.window = slice(*window)
        else:
            self.window = window

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        # compute the output window length based on the range of the window
        if self.window.stop is None and self.window.start is None:
            output_shape[2] = input_shape[2] / self.window.step
        elif self.window.stop is None:
            output_shape[2] = (input_shape[2] -
                               self.window.start) / self.window.step
        elif self.window.start is None:
            output_shape[2] = self.window.stop / self.window.step
        else:
            output_shape[2] = (self.window.stop -
                               self.window.start) / self.window.step
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        start = self.window.start
        stop = self.window.stop
        step = self.window.step
        return input[:, :, start:stop][:, :, ::-1][:, :, ::step][:, :, ::-1]


def run_tests():
    l_in = layers.InputLayer(shape=(64, 32, 2000))
    l_sample = SubsampleLayer(l_in, window=(None, 1000, 10))

    X = np.random.normal(0, 1, (64, 32, 2000))
    expected_output_shape = (64, 32, 100)
    actual_output_shape = l_sample.get_output_shape_for(X.shape)
    assert expected_output_shape == actual_output_shape, '%r != %r' % (
        expected_output_shape, actual_output_shape)
    expected_output = X[:, :, None:1000][:, :, ::-1][:, :, ::10][:, :, ::-1]
    actual_output = l_sample.get_output_for(X)
    assert expected_output_shape == actual_output.shape, '%r != %r' % (
        expected_output_shape, actual_output.shape)
    print expected_output.shape, actual_output.shape
    assert (expected_output == actual_output).all(), 'bad subsampling'

if __name__ == '__main__':
    run_tests()
