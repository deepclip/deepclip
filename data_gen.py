# -*- coding: utf-8 -*-

import numpy as np
from constants import VOCAB


def induce_all_SNPs_at_all_positions(string):
    set_of_new_string = [string]

    VOCAB_gene = ['a', 'c', 'g', 't']

    for let in VOCAB_gene:
        for i in range(len(string) - 1):
            if i == 0:
                set_of_new_string.append(let + string[1:])

            if i > 0:
                set_of_new_string.append(string[:i] + let + string[i + 1:])

        set_of_new_string.append(string[:-1] + let)

    return set_of_new_string


def create_snip_of_weighted_seqs(b_cargsr, b_srseq, pat, prob):
    out = np.array(b_cargsr).reshape((len(b_cargsr), 200))
    seqs = np.array(b_srseq).reshape((len(b_cargsr), 200))

    short_seqs = []
    number = []
    for i in range(len(out)):
        indx = 0
        for ii in range(0, len(out[i]) - (pat * len(VOCAB)) + 1, 4):
            if np.sum(seqs[i][ii:ii + pat * len(VOCAB)]) < pat:
                continue
            if np.sum(seqs[i][ii:ii + pat * len(VOCAB)]) == pat:
                short_seqs.append(out[i][ii:ii + pat * len(VOCAB)])
                indx += 1
                number.append([indx, i])

    return np.array(short_seqs).reshape((len(short_seqs), 1, pat * len(VOCAB))), number


def onehot_binary(input_a, input_b, freq, vocab):
    """
        This functions takes two classes of input sequences and a vocabulary and 
        returns the onehot encoded sequences as a numpy array and their respective target values.
        Sequence length is assumed to be equal for all sequences.
        
        Inputs:     input_a (list of class 0 sequences)
                    input_b (list of class 1 sequences)
                    vocab (list of characters)
                    
        Outputs:    X (numpy array of one-hot encoded sequences)
                    y (numpy array of corresponding target values)
    """

    inputs = input_a + input_b
    y_out = np.asarray([[0]] * len(input_a) + [[1]] * len(input_b))

    #vals = (1 / np.array(freq)) / (1/np.min(freq))
    #freq = vals
    #print vals
    X_out = []
    for input in inputs:
        x = np.zeros((len(input), len(VOCAB)))
        for i in xrange(len(input)):

	    #if input[i].lower() in vocab:
	    #	x[i][vocab.index(input[i])] += freq[vocab.index(input[i])] #vals[vocab.index(input[i])] #vals[vocab.index(input[i])] #1 - freq[vocab.index(input[i])]

            for j in xrange(len(vocab)):
                if input[i].lower() in vocab[j].lower():
                    x[i][j] = 1
        X_out.append(x.flatten())

    X_out = np.asarray(X_out)
    X_out.reshape((len(inputs), 1, len(inputs[0]) * len(vocab)))
    return X_out, y_out


def onehot_encode(inputs, freq, vocab):
    """
        This functions takes a list of input sequences and a vocabulary and 
        returns the onehot encoded sequences as a numpy array.
        Sequence length is assumed to be equal for all sequences.
        
        Inputs:     inputs (list of sequences)
                    vocab (list of characters)
                    
        Outputs:    X (numpy array of one-hot encoded sequences)
    """

    #vals = (1 / np.array(freq)) / (1/np.min(freq))
    #print vals
    #freq = vals

    X_out = []
    for input in inputs:
        x = np.zeros((len(input), len(VOCAB)))
        for i in xrange(len(input)):
	    #if input[i].lower() in vocab:
            #    x[i][vocab.index(input[i])] += freq[vocab.index(input[i])] #vals[vocab.index(input[i])] #vals[vocab.index(input[i])] #1 - freq[vocab.index(input[i])]

   	    for j in xrange(len(vocab)):
                if input[i].lower() in vocab[j].lower():
                    x[i][j] = 1

        X_out.append(x.flatten())

    X_out = np.asarray(X_out)
    X_out.reshape((len(inputs), 1, len(inputs[0]) * len(vocab)))
    return X_out
