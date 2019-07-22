# -*- coding: utf-8 -*-

import lasagne


def slicpad_layers(layer, number_of_convs, filter_size, BATCH_SIZE, VOCAB, SEQ_SIZE, form='whole'):
    """ Slices and pads convolutional layer.
        Outputs: the padding layer """

    mini_seqs = []
    if form == 'whole':
        for i in range(0, len(VOCAB) * SEQ_SIZE - filter_size + 1, len(VOCAB)):
            slic0 = lasagne.layers.SliceLayer(layer, indices=slice(i, filter_size + i), axis=2)
            pad0 = lasagne.layers.PadLayer(slic0, width=[(i, len(VOCAB) * SEQ_SIZE - i - filter_size)], val=0,
                                           batch_ndim=2)
            pad0 = lasagne.layers.ReshapeLayer(pad0, (BATCH_SIZE, 1, len(VOCAB) * SEQ_SIZE))
            mini_seqs.append(pad0)

    pad0 = lasagne.layers.ConcatLayer(incomings=mini_seqs, axis=1)
    pad0 = lasagne.layers.ReshapeLayer(pad0, (BATCH_SIZE, number_of_convs, len(VOCAB) * SEQ_SIZE))

    return pad0


def shape_convolutions(pool, BATCH_SIZE, FILTER_SIZES, FS, ALL_F, VOCAB, SEQ_SIZE, form='whole'):
    """ Shapes the convolutional layer """
    convolutions = []
    if form == 'whole':
        for i in range(len(VOCAB) * SEQ_SIZE):
            convolutions.append(pool)

        pool = lasagne.layers.ConcatLayer(incomings=convolutions, axis=1)
        pool = lasagne.layers.DimshuffleLayer(pool, (0, 2, 1))
        pool = lasagne.layers.ReshapeLayer(pool, (BATCH_SIZE, ALL_F, len(VOCAB) * SEQ_SIZE))

        slatt_layers = []
        for i in xrange(len(FILTER_SIZES)):
            if i <= 1:
                if i == 0:
                    slatt_layers.append(lasagne.layers.SliceLayer(pool, indices=slice(0, FS[0]), axis=1))
                elif i == 1:
                    slatt_layers.append(lasagne.layers.SliceLayer(pool, indices=slice(FS[0], sum(FS[:i + 1])), axis=1))
            else:
                slatt_layers.append(
                    lasagne.layers.SliceLayer(pool, indices=slice(sum(FS[:i]), sum(FS[:i + 1])), axis=1))

        return slatt_layers
    
    
def shape_convolutions2(pool, BATCH_SIZE, FILTER_SIZES, FS, VOCAB, SEQ_SIZE, nr):
    """ Shapes the convolutional layer """
    convolutions = []
    for i in range(FILTER_SIZES[nr]):
        convolutions.append(pool)

    pool = lasagne.layers.ConcatLayer(incomings=convolutions, axis=1)
    pool = lasagne.layers.DimshuffleLayer(pool, (0, 2, 1))
    layer = lasagne.layers.ReshapeLayer(pool, (BATCH_SIZE, 1, FS[nr] * FILTER_SIZES[nr]))

    mini_seqs = []
    for i in range(FS[nr]):
        s = lasagne.layers.SliceLayer(layer, indices=slice(i * FILTER_SIZES[nr], i * FILTER_SIZES[nr] + FILTER_SIZES[nr]), axis=2)
        pad0 = lasagne.layers.PadLayer(s, width=[(i*len(VOCAB), len(VOCAB) * SEQ_SIZE - i*len(VOCAB) - FILTER_SIZES[nr])], val=0,
                                       batch_ndim=2)
        pad0 = lasagne.layers.ReshapeLayer(pad0, (BATCH_SIZE, 1, len(VOCAB) * SEQ_SIZE))
        mini_seqs.append(pad0)

    pad = lasagne.layers.ConcatLayer(incomings=mini_seqs, axis=1)
    pad = lasagne.layers.ReshapeLayer(pad, (BATCH_SIZE, FS[nr], len(VOCAB) * SEQ_SIZE))

    return pad
