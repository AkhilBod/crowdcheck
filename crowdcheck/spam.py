"""Spot workers whose answers don't depend on the question.

A spammer's confusion matrix has identical rows: whatever the true answer,
they pick from the same distribution (all "cat", or random clicks). A
worker who is consistently wrong, say always swapping two classes, has very
different rows and is actually useful once you know their habit.

Score from Raykar & Yu (2012): average squared distance between rows,
scaled so a perfect worker scores 1 and a spammer scores 0.
"""

import numpy as np


def spammer_scores(confusion):
    m, k, _ = confusion.shape
    diffs = confusion[:, :, None, :] - confusion[:, None, :, :]
    pairs = (diffs ** 2).sum(axis=3)
    return pairs.sum(axis=(1, 2)) / (2 * k * (k - 1))
