# -*- coding: utf-8 -*-

import numpy as np


def gradient_cm2(sr, a1):
    """
    This function adds the probabilities of a sequence being a class.
    The probabilities are derived from the CNN. The top row is the percent wise
    distribution of class 0 prediction and the bottom row is the same
    for class 1.
    """
    x = np.zeros((2, 2))
    for i in sr:
        x[0] += i
    for i in a1:
        x[1] += i

    x1 = np.vstack(np.array([x[0], x[1]]))
    x2 = np.vstack(np.array([x[0] / len(sr), x[1] / len(a1)]))
    # x2 = np.array(x2).transpose()
    # print '\n The added pure-predictions of the CNN:'
    # print x1

    # print '\n The percent-wise distribution of the model predictions:'
    # print x2

    return x1, x2


def bernoulli_cm2(sr, a1):
    """
    The predictions of the CNN is treated as Bernoulli trials
    ('success' or 'failure')
    """

    x = np.zeros((2, 2))
    for i in sr:
        x[0] += np.round(i)
    for i in a1:
        x[1] += np.round(i)

    x1 = np.vstack(np.array([x[0], x[1]]))
    x2 = np.vstack(np.array([x[0] / len(sr), x[1] / len(a1)]))
    # x2 = np.array(x2).transpose()

    print '\n The Bernoulli distribution:'
    print x1
    print '\n The percent-wise Bernoulli distribution:'
    print x2

    return x1, x2


def gradient_cm3(sr, a1, p):
    """
    This function adds the probabilities of a sequence being a class.
    The probabilities are derived from the CNN. The top row is the percentwise
    distribution of SRSF5 prediction and the bottom row is the same
    for hnRNPA1.
    """
    x = np.zeros((3, 3))
    for i in sr:
        x[0] += i
    for i in a1:
        x[1] += i
    for i in p:
        x[2] += i

    x2 = x / np.sum(x, 1).reshape((3, 1))
    x2 = np.array(x2).transpose()
    # print '\n The added pure-predictions of the CNN:'
    # print x1

    print '\n The percent-wise distribution of the model predictions:'
    print x2

    return x, x2


def bernoulli_cm3(sr, a1, p):
    """
    The predictions of the CNN is treated as Bernoulli trials
    ('succes' or 'failure')
    """

    x = np.zeros((3, 3))
    for i in sr:
        x[0][np.argmax(i)] += 1
    for i in a1:
        x[1][np.argmax(i)] += 1
    for i in p:
        x[2][np.argmax(i)] += 1

    x2 = x / np.sum(x, 1).reshape((3, 1))
    x2 = np.array(x2).transpose()
    # print '\n pure Bernoulli distribution:'
    # print x1

    print '\n The percent-wise Bernoulli distribution:'
    print x2

    return x, x2


def gradient_cm2_binary(sr, a1):
    """
    This function adds the probabilities of a sequence being a class.
    The probabilities are derived from the CNN. The top row is the percentwise
    distribution of SRSF5 prediction and the bottom row is the same
    for hnRNPA1.
    """
    x = np.zeros((2, 2))
    for i in sr:
        x[0] += np.array([1 - i[0], 1 - (1 - i[0])])
    for i in a1:
        x[1] += np.array([1 - i[0], 1 - (1 - i[0])])

    x1 = np.vstack(np.array([x[0], x[1]]))
    x2 = np.vstack(np.array([x[0] / len(sr), x[1] / len(a1)]))
    # x2 = np.array(x2).transpose()
    # print '\n The added pure-predictions of the CNN:'
    # print x1

    print '\n The percent-wise distribution of the model predictions:'
    print x2

    return x1, x2


def bernoulli_cm2_binary(sr, a1):
    """
    The predictions of the CNN is treated as Bernoulli trials 
    ('succes' or 'failure')
    """

    x = np.zeros((2, 2))
    for i in sr:
        x[0] += np.round(np.array([1 - i[0], 1 - (1 - i[0])]))
    for i in a1:
        x[1] += np.round(np.array([1 - i[0], 1 - (1 - i[0])]))

    x1 = np.vstack(np.array([x[0], x[1]]))
    x2 = np.vstack(np.array([x[0] / len(sr), x[1] / len(a1)]))
    # x2 = np.array(x2).transpose()

    print '\n Validation confusion matrix:'
    print x1
    print '\n Validation accuracy in percent:'
    print x2

    return x1, x2
