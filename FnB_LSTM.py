# -*- coding: utf-8 -*-

import lasagne


def FnB_LSTM(inp_forw, inp_backw, N_LSTM, DROPOUT_LSTM, GRAD_CLIP, forget_b, ing=0.0, cellg=0.0, outg=0., learn_init=False, peepholes=False):
    l_forward_1 = lasagne.layers.LSTMLayer(
        inp_forw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.tanh,
        grad_clipping=GRAD_CLIP,
        backwards=False, 
        learn_init=False, 
        peepholes=False
    )

    l_backward_1 = lasagne.layers.LSTMLayer(
        inp_backw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        nonlinearity=lasagne.nonlinearities.tanh,  # (or linear)
        grad_clipping=GRAD_CLIP,
        backwards=True, 
        learn_init=False, 
        peepholes=False
    )

    l_fordrop1 = lasagne.layers.DropoutLayer(l_forward_1, p=DROPOUT_LSTM) 
    l_backdrop1 = lasagne.layers.DropoutLayer(l_backward_1, p=DROPOUT_LSTM)
    
    return l_fordrop1, l_backdrop1    

    
def FnB_LSTM_tanh(inp_forw, inp_backw, N_LSTM, DROPOUT_LSTM, GRAD_CLIP, forget_b, ing=0., cellg=0., outg=0., learn_init=False, peepholes=False):
    l_forward_1 = lasagne.layers.LSTMLayer(
        inp_forw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.tanh
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.tanh,
        grad_clipping=GRAD_CLIP,
        backwards=False, 
        learn_init=False, 
        peepholes=False
    )

    l_backward_1 = lasagne.layers.LSTMLayer(
        inp_backw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.tanh
        ),

        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        nonlinearity=lasagne.nonlinearities.tanh,
        grad_clipping=GRAD_CLIP,
        backwards=True, 
        learn_init=False, 
        peepholes=False
    )

    l_fordrop1 = lasagne.layers.DropoutLayer(l_forward_1, p=DROPOUT_LSTM) 
    l_backdrop1 = lasagne.layers.DropoutLayer(l_backward_1, p=DROPOUT_LSTM)
    
    return l_fordrop1, l_backdrop1    
    

def FnB_LSTM_softplus(inp_forw, inp_backw, N_LSTM, DROPOUT_LSTM, GRAD_CLIP, forget_b, ing=0., cellg=0., outg=0., learn_init=False, peepholes=False):
    l_forward_1 = lasagne.layers.LSTMLayer(
        inp_forw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.softplus
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.softplus,
        grad_clipping=GRAD_CLIP,
        backwards=False, 
        learn_init=False, 
        peepholes=False
    )

    l_backward_1 = lasagne.layers.LSTMLayer(
        inp_backw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.softplus
        ),

        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Uniform(),
            W_hid=lasagne.init.Uniform(),
            W_cell=lasagne.init.Uniform(), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),

        nonlinearity=lasagne.nonlinearities.softplus,
        grad_clipping=GRAD_CLIP,
        backwards=True, 
        learn_init=False, 
        peepholes=False
    )

    l_fordrop1 = lasagne.layers.DropoutLayer(l_forward_1, p=DROPOUT_LSTM) 
    l_backdrop1 = lasagne.layers.DropoutLayer(l_backward_1, p=DROPOUT_LSTM)
    
    return l_fordrop1, l_backdrop1   


def FnB_LSTM_softplusN(inp_forw, inp_backw, N_LSTM, DROPOUT_LSTM, GRAD_CLIP, forget_b, ing=0., cellg=0., outg=0., learn_init=False, peepholes=False):
    l_forward_1 = lasagne.layers.LSTMLayer(
        inp_forw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.softplus
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.softplus,
        grad_clipping=GRAD_CLIP,
        backwards=False, 
        learn_init=False, 
        peepholes=False
    )

    l_backward_1 = lasagne.layers.LSTMLayer(
        inp_backw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.softplus
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.softplus,
        grad_clipping=GRAD_CLIP,
        backwards=True, 
        learn_init=False, 
        peepholes=False
    )

    l_fordrop1 = lasagne.layers.DropoutLayer(l_forward_1, p=DROPOUT_LSTM) 
    l_backdrop1 = lasagne.layers.DropoutLayer(l_backward_1, p=DROPOUT_LSTM)
    
    return l_fordrop1, l_backdrop1   


def FnB_LSTM_N(inp_forw, inp_backw, N_LSTM, DROPOUT_LSTM, GRAD_CLIP, forget_b, ing=0., cellg=0., outg=0., learn_init=False, peepholes=False):
    l_forward_1 = lasagne.layers.LSTMLayer(
        inp_forw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.tanh,
        grad_clipping=GRAD_CLIP,
        backwards=False, 
        learn_init=False, 
        peepholes=False
    )

    l_backward_1 = lasagne.layers.LSTMLayer(
        inp_backw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.tanh,
        grad_clipping=GRAD_CLIP,
        backwards=True, 
        learn_init=False, 
        peepholes=False
    )

    l_fordrop1 = lasagne.layers.DropoutLayer(l_forward_1, p=DROPOUT_LSTM)
    l_backdrop1 = lasagne.layers.DropoutLayer(l_backward_1, p=DROPOUT_LSTM)
    
    return l_fordrop1, l_backdrop1   


def FnB_LSTMsig_N(inp_forw, inp_backw, N_LSTM, DROPOUT_LSTM, GRAD_CLIP, forget_b, ing=0., cellg=0., outg=0., learn_init=False, peepholes=False):
    l_forward_1 = lasagne.layers.LSTMLayer(
        inp_forw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        grad_clipping=GRAD_CLIP,
        backwards=False, 
        learn_init=False, 
        peepholes=False
    )

    l_backward_1 = lasagne.layers.LSTMLayer(
        inp_backw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        grad_clipping=GRAD_CLIP,
        backwards=True, 
        learn_init=False, 
        peepholes=False
    )

    l_fordrop1 = lasagne.layers.DropoutLayer(l_forward_1, p=DROPOUT_LSTM) 
    l_backdrop1 = lasagne.layers.DropoutLayer(l_backward_1, p=DROPOUT_LSTM)
    
    return l_fordrop1, l_backdrop1  

    
def FnB_LSTMtan_N(inp_forw, inp_backw, N_LSTM, DROPOUT_LSTM, GRAD_CLIP, forget_b, ing=0., cellg=0., outg=0., learn_init=False, peepholes=False):
    l_forward_1 = lasagne.layers.LSTMLayer(
        inp_forw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.tanh
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.tanh,
        grad_clipping=GRAD_CLIP,
        backwards=False, 
        learn_init=False, 
        peepholes=False
    )

    l_backward_1 = lasagne.layers.LSTMLayer(
        inp_backw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.tanh
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Normal(0.1),
            W_hid=lasagne.init.Normal(0.1),
            W_cell=lasagne.init.Normal(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.tanh,
        grad_clipping=GRAD_CLIP,
        backwards=True, 
        learn_init=False, 
        peepholes=False
    )

    l_fordrop1 = lasagne.layers.DropoutLayer(l_forward_1, p=DROPOUT_LSTM) 
    l_backdrop1 = lasagne.layers.DropoutLayer(l_backward_1, p=DROPOUT_LSTM)
    
    return l_fordrop1, l_backdrop1




def FnB_LSTMconst(inp_forw, inp_backw, N_LSTM, DROPOUT_LSTM, GRAD_CLIP, forget_b, ing=0., cellg=0., outg=0., learn_init=False, peepholes=False):
    l_forward_1 = lasagne.layers.LSTMLayer(
        inp_forw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Constant(0.1),
            W_hid=lasagne.init.Constant(0.1),
            W_cell=lasagne.init.Constant(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Constant(0.1),
            W_hid=lasagne.init.Constant(0.1),
            W_cell=lasagne.init.Constant(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Constant(0.1),
            W_hid=lasagne.init.Constant(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.tanh
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Constant(0.1),
            W_hid=lasagne.init.Constant(0.1),
            W_cell=lasagne.init.Constant(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.tanh,
        grad_clipping=GRAD_CLIP,
        backwards=False, 
        learn_init=False, 
        peepholes=False
    )

    l_backward_1 = lasagne.layers.LSTMLayer(
        inp_backw, 
        N_LSTM,
        ingate=lasagne.layers.Gate(
            W_in=lasagne.init.Constant(0.1),
            W_hid=lasagne.init.Constant(0.1),
            W_cell=lasagne.init.Constant(0.1), 
            b=lasagne.init.Constant(ing), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        forgetgate=lasagne.layers.Gate(
            W_in=lasagne.init.Constant(0.1),
            W_hid=lasagne.init.Constant(0.1),
            W_cell=lasagne.init.Constant(0.1), 
            b=lasagne.init.Constant(forget_b), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        cell=lasagne.layers.Gate(
            W_in=lasagne.init.Constant(0.1),
            W_hid=lasagne.init.Constant(0.1),
            W_cell=None, 
            b=lasagne.init.Constant(cellg), 
            nonlinearity=lasagne.nonlinearities.tanh
        ),
        outgate=lasagne.layers.Gate(
            W_in=lasagne.init.Constant(0.1),
            W_hid=lasagne.init.Constant(0.1),
            W_cell=lasagne.init.Constant(0.1), 
            b=lasagne.init.Constant(outg), 
            nonlinearity=lasagne.nonlinearities.sigmoid
        ),
        nonlinearity=lasagne.nonlinearities.tanh,
        grad_clipping=GRAD_CLIP,
        backwards=True, 
        learn_init=False, 
        peepholes=False
    )

    l_fordrop1 = lasagne.layers.DropoutLayer(l_forward_1, p=DROPOUT_LSTM) 
    l_backdrop1 = lasagne.layers.DropoutLayer(l_backward_1, p=DROPOUT_LSTM)
    
    return l_fordrop1, l_backdrop1
