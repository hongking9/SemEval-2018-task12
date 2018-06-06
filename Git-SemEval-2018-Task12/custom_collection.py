# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 17:26:38 2017

@author: ChoiHongSeok
"""
import theano.tensor as T
from lasagne.layers.base import MergeLayer
from lasagne.layers import Layer

__all__ = [
    "autocrop",
    "autocrop_array_shapes",
    "Custom_merge"
]

def autocrop(inputs, cropping):
    if cropping is None:
        return inputs
    else:
        ndim = inputs[0].ndim
        if not all(input.ndim == ndim for input in inputs):
            raise ValueError("Not all inputs are of the same "
                             "dimensionality. Got {0} inputs of "
                             "dimensionalities {1}.".format(
                                len(inputs),
                                [input.ndim for input in inputs]))
        shapes = [input.shape for input in inputs]
        shapes_tensor = T.as_tensor_variable(shapes)
        min_shape = T.min(shapes_tensor, axis=0)
        slices_by_input = [[] for i in range(len(inputs))]
        cropping = list(cropping)
        if ndim > len(cropping):
            cropping = list(cropping) + \
                         [None] * (ndim - len(cropping))
        for dim, cr in enumerate(cropping):
            if cr is None:
                slice_all = slice(None)
                for slices in slices_by_input:
                    slices.append(slice_all)
            else:
                sz = min_shape[dim]
                if cr == 'lower':
                    slc_lower = slice(None, sz)
                    for slices in slices_by_input:
                        slices.append(slc_lower)
                elif cr == 'upper':
                    slc_upper = slice(-sz, None)
                    for slices in slices_by_input:
                        slices.append(slc_upper)
                elif cr == 'center':
                    for sh, slices in zip(shapes, slices_by_input):
                        offset = (sh[dim] - sz) // 2
                        slices.append(slice(offset, offset+sz))
                else:
                    raise ValueError(
                        'Unknown crop mode \'{0}\''.format(cr))

        return [input[slices] for input, slices in
                zip(inputs, slices_by_input)]


def autocrop_array_shapes(input_shapes, cropping):
    if cropping is None:
        return input_shapes
    else:
        ndim = len(input_shapes[0])
        if not all(len(sh) == ndim for sh in input_shapes):
            raise ValueError("Not all inputs are of the same "
                             "dimensionality. Got {0} inputs of "
                             "dimensionalities {1}.".format(
                                len(input_shapes),
                                [len(sh) for sh in input_shapes]))
        result = []
        cropping = list(cropping)
        if ndim > len(cropping):
            cropping = list(cropping) + \
                         [None] * (ndim - len(cropping))

        for sh, cr in zip(zip(*input_shapes), cropping):
            if cr is None:
                result.append(sh)
            elif cr in {'lower', 'center', 'upper'}:
                min_sh = None if any(x is None for x in sh) else min(sh)
                result.append([min_sh] * len(sh))
            else:
                raise ValueError('Unknown crop mode \'{0}\''.format(cr))
        return [tuple(sh) for sh in zip(*result)]

class Custom_merge(MergeLayer):
    def __init__(self, incomings, axis=1, cropping=None, **kwargs):
        super(Custom_merge, self).__init__(incomings, **kwargs)
        self.axis = axis
        if cropping is not None:
            cropping = list(cropping)
            cropping[axis] = None
        self.cropping = cropping

    def get_output_shape_for(self, input_shapes):
        input_shapes = autocrop_array_shapes(input_shapes, self.cropping)
        output_shape = [next((s for s in sizes if s is not None), None)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            axis = self.axis if self.axis >= 0 else len(shape1) + self.axis
            return (len(shape1) == len(shape2) and
                    all(i == axis or s1 is None or s2 is None or s1 == s2
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "in the concatenation axis")
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        concat_size = None if any(s is None for s in sizes) else max(sizes)
        output_shape[self.axis] = concat_size
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        inputs = autocrop(inputs, self.cropping)
        return inputs[0]*inputs[1]

class Tensor_func_Layer(Layer):
    def __init__(self, incoming, t_func, **kwargs):
        super(Tensor_func_Layer, self).__init__(incoming, **kwargs)
        self.t_func = t_func
    def get_output_for(self, input, **kwargs):
        return self.t_func(input)

class Softmax_axis_Layer(Layer):
    def __init__(self, incoming, axis=1, **kwargs):
        super(Softmax_axis_Layer, self).__init__(incoming, **kwargs)
        self.axis = axis
    def get_output_for(self, input, **kwargs):
        return input/(T.sum(input,axis=self.axis,keepdims=True)+0.1**6)

class Sum_axis_Layer(Layer):
    def __init__(self, incoming, axis=1, **kwargs):
        super(Sum_axis_Layer, self).__init__(incoming, **kwargs)
        self.axis = axis
    def get_output_for(self, input, **kwargs):
        return T.sum(input,axis=self.axis)
    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape[:self.axis]) + list(input_shape[self.axis+1:]))

class INVLayer(Layer):
    def get_output_for(self, input, **kwargs):
        return 1.0/(input+0.1**5)#T.inv(input+0.1**5)
