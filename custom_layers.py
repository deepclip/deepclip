import theano.tensor as T
from slice_n_pad import *


class Sum_last_ax(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input.sum(axis=-1)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]


class Sum_ax1(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], 1, input_shape[2]


class Sum_ax2(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input.sum(axis=2)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], 1


class Max_ax2(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input.max(axis=2)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], 1


class Divide_to_one(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input / (input + 0.00000000001)

    def get_output_shape_for(self, input_shape):
        return input_shape




class Semi_soft(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input / T.sum(input, axis=1, keepdims=True)

    def get_output_shape_for(self, input_shape):
        return input_shape


class Soft_info_2cl(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input * (T.log2(2) + T.sum(input * T.log2(input), axis=2, keepdims=True))

    def get_output_shape_for(self, input_shape):
        return input_shape


class Minus_layer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return -input

    def get_output_shape_for(self, input_shape):
        return input_shape


class Regulator(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input * (1 - (T.sum(input + 0.000000001, axis=1, keepdims=True) / T.sum(T.sum(input + 0.000000001, axis=1, keepdims=True))))

    def get_output_shape_for(self, input_shape):
        return input_shape


class High_divx(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input / T.max(input + 0.000000001, axis=2, keepdims=True)

    def get_output_shape_for(self, input_shape):
        return input_shape


class High_divabs(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input / T.max(abs(input) + 0.000000001, axis=2, keepdims=True)

    def get_output_shape_for(self, input_shape):
        return input_shape
