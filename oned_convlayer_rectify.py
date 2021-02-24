# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import lasagne
from slice_n_pad import shape_convolutions2
from custom_layers import Sum_ax1



def setup_conv_layers(l_in, FS, BATCH_SIZE, SEQ_SIZE,  filters_per_convolution, filter_sizes, vocab, dropout_conv=0.0,  padding="valid"):
    cdrop_l = []
    pool_l = []
    done = []
    done2 = []
    for i in range(len(filter_sizes)):
        conv_l = lasagne.layers.Conv1DLayer(
            l_in,
            num_filters=int(filters_per_convolution),
            filter_size=int(filter_sizes[i]),
            stride=int(len(vocab)),
            pad=padding, # a string value, eg. padding == 'valid'
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Constant(0.01), b=None
        )

        max = lasagne.layers.FeaturePoolLayer(conv_l, pool_size=int(filters_per_convolution),pool_function=T.max)
        wta = lasagne.layers.FeatureWTALayer(max, pool_size=int(FS[i]), axis=2)
        cdrop_l.append(wta)
        pool_l.append(lasagne.layers.FeaturePoolLayer(conv_l, pool_size=int(filters_per_convolution),
                                                      pool_function=T.argmax))
        max = lasagne.layers.ElemwiseMergeLayer([max, wta], theano.tensor.add)
        max = lasagne.layers.ElemwiseMergeLayer([max, max], theano.tensor.mul)

        temp = shape_convolutions2(max, BATCH_SIZE, [int(f) for f in filter_sizes], [int(f) for f in FS], vocab, int(SEQ_SIZE), i)
        temp = Sum_ax1(temp)
        temp = lasagne.layers.ReshapeLayer(temp, (BATCH_SIZE, 1, int(SEQ_SIZE*len(vocab))))
        temp = lasagne.layers.ElemwiseMergeLayer([l_in, temp],
                                                 theano.tensor.mul)
        done.append(temp)
        done2.append(lasagne.layers.ReshapeLayer(temp, (BATCH_SIZE, int(SEQ_SIZE), len(vocab))))

    return cdrop_l, pool_l, done, done2


