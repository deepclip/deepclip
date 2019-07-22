# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import auc


def get_auroc_data(y_test, y_preds, segments=1000):
    assert len(y_test) == len(y_preds)

    test = np.array(y_test)
    preds = np.array(y_preds)

    is_pos = (test == 1)
    is_neg = (test == 0)

    num_pos = is_pos.sum()
    num_neg = is_neg.sum()

    x0 = []
    y0 = []

    for i in range(segments):
        thr = 1.0 - float(i+1) / float(segments)

        sel = (preds > thr)
        pos = np.logical_and(sel, is_pos).sum()
        neg = np.logical_and(sel, is_neg).sum()

        x0.append(float(neg) / float(num_neg))
        y0.append(float(pos) / float(num_pos))

    return auc(x0, y0), [x0, y0]
