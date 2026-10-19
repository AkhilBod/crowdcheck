"""Three ways to turn noisy crowd labels into one answer per task.

majority_vote        what most people do
gold_weighted_vote   grade workers on a few tasks you already know the answer to
dawid_skene          learn every worker's habits from the labels alone (Dawid & Skene, 1979)
"""

import numpy as np

from .labels import Labels


def majority_vote(labels: Labels, rng=None):
    """Most common label per task. Ties are broken at random, not toward class 0."""
    rng = rng or np.random.default_rng(0)
    counts = labels.counts()
    noise = rng.random(counts.shape) * 1e-6
    return np.argmax(counts + noise, axis=1)


def gold_weighted_vote(labels: Labels, gold: dict, min_gold=3):
    """Weight each vote by how often that worker got the known-answer tasks right.

    gold maps task -> true label. A worker's weight is the log-odds of their
    gold accuracy, so someone at chance counts for nothing and someone worse
    than chance counts against the label they picked. Workers who saw fewer
    than min_gold gold tasks get the average weight.
    """
    k = labels.n_classes
    right = np.zeros(labels.n_workers)
    seen = np.zeros(labels.n_workers)
    for t, w, l in zip(labels.task, labels.worker, labels.label):
        if t in gold:
            seen[w] += 1
            right[w] += l == gold[t]
    acc = (right + 1) / (seen + 2)  # Laplace smoothing
    chance = 1 / k
    weight = np.log(np.clip(acc, 1e-3, 1 - 1e-3) * (k - 1) / np.clip(1 - acc, 1e-3, None))
    weight[acc <= chance] = 0.0
    known = seen >= min_gold
    if known.any():
        weight[~known] = weight[known].mean()
    scores = np.zeros((labels.n_tasks, k))
    np.add.at(scores, (labels.task, labels.label), weight[labels.worker])
    out = np.argmax(scores + 1e-9 * labels.counts(), axis=1)
    for t, y in gold.items():
        out[t] = y
    return out


def dawid_skene(labels: Labels, iters=50, tol=1e-6, smoothing=0.01):
    """EM over a full confusion matrix per worker.

    Returns (answers, posteriors, confusion, priors). confusion[w, true, said]
    is the estimated chance worker w says `said` when the answer is `true`.
    """
    n, k, m = labels.n_tasks, labels.n_classes, labels.n_workers
    t, w, l = labels.task, labels.worker, labels.label

    counts = labels.counts()
    post = (counts + 1e-9) / (counts + 1e-9).sum(axis=1, keepdims=True)  # start from vote shares

    prev = None
    for _ in range(iters):
        # M-step: class priors and each worker's confusion matrix
        priors = post.mean(axis=0)
        conf = np.full((m, k, k), smoothing)
        np.add.at(conf, (w, slice(None), l), post[t])
        conf /= conf.sum(axis=2, keepdims=True)

        # E-step: posterior over the true class of each task
        log_post = np.tile(np.log(priors), (n, 1))
        np.add.at(log_post, t, np.log(conf[w, :, l]))
        log_post -= log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post)
        post /= post.sum(axis=1, keepdims=True)

        if prev is not None and np.abs(post - prev).max() < tol:
            break
        prev = post

    return np.argmax(post, axis=1), post, conf, priors
