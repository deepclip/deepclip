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
    for i in xrange(len(filter_sizes)):
        conv_l = lasagne.layers.Conv1DLayer(
            l_in,
            num_filters=filters_per_convolution,
            filter_size=filter_sizes[i],
            stride=len(vocab),
            pad=padding,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Constant(0.01), b=None
        )

        max = lasagne.layers.FeaturePoolLayer(conv_l, pool_size=(filters_per_convolution),pool_function=T.max)
        wta = lasagne.layers.FeatureWTALayer(max, pool_size=FS[i], axis=2)
        cdrop_l.append(wta)
        pool_l.append(lasagne.layers.FeaturePoolLayer(conv_l, pool_size=(filters_per_convolution),
                                                      pool_function=T.argmax))
	
	max = lasagne.layers.ElemwiseMergeLayer([max, wta],
                                                 theano.tensor.add)
	max = lasagne.layers.ElemwiseMergeLayer([max, max],
                                                 theano.tensor.mul)

        temp = shape_convolutions2(max, BATCH_SIZE, filter_sizes, FS, vocab, SEQ_SIZE, i)
        temp = Sum_ax1(temp)
        temp = lasagne.layers.ReshapeLayer(temp, (BATCH_SIZE, 1, SEQ_SIZE*len(vocab)))
        temp = lasagne.layers.ElemwiseMergeLayer([l_in, temp],
                                                 theano.tensor.mul)
        done.append(temp)
        done2.append(lasagne.layers.ReshapeLayer(temp, (BATCH_SIZE, SEQ_SIZE, len(vocab))))

    return cdrop_l, pool_l, done, done2


