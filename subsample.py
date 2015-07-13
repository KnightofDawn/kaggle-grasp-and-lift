from lasagne import layers


class SubsampleLayer(layers.Layer):
    def __init__(self, incoming, window, name=None, **kwargs):
        super(SubsampleLayer, self).__init__(incoming, **kwargs)
        if not isinstance(window, slice):
            self.window = slice(*window)
        else:
            self.window = window

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
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
        start = (input.shape[0] - 1) % step
        if self.window.start is not None:
            start += self.window.start
        stop = self.window.stop

        s = slice(start, stop, step)
        return input[:, :, s]
