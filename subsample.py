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
            output_shape[2] = (input_shape[2] - self.window.start) / self.window.step
        elif self.window.start is None:
            output_shape[2] = self.window.stop / self.window.step
        else:
            output_shape[2] = (self.window.stop - self.window.start) / self.window.step
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        step = self.window.step
        # always take the signal value at the last time-step
        start = (input.shape[0] - 1) % step
        # account for the possible offset
        if self.window.start is not None:
            start += self.window.start
        stop = self.window.stop

        # slice across the time-axis
        return input[:, :, slice(start, stop, step)]
