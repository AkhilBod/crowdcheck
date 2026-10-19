"""A fake labeling job with the kinds of workers you actually get.

honest     right most of the time, mistakes spread across the other classes
swapper    honest, except they consistently mix up two particular classes
clicker    picks at random
lazy       picks the same class every time
Spammers (clicker, lazy) are fast, so they end up doing more of the tasks.
"""

import numpy as np

from .labels import Labels


def make_job(n_tasks=2000, n_classes=4, n_workers=60, redundancy=3,
             spam_share=0.2, swapper_share=0.1, spam_speed=3.0,
             class_weights=None, seed=0):
    rng = np.random.default_rng(seed)
    k = n_classes
    p = np.asarray(class_weights or [0.4] + [0.6 / (k - 1)] * (k - 1))
    truth = rng.choice(k, size=n_tasks, p=p)

    kinds = []
    n_spam = round(spam_share * n_workers)
    n_swap = round(swapper_share * n_workers)
    kinds += ["clicker" if i % 2 else "lazy" for i in range(n_spam)]
    kinds += ["swapper"] * n_swap
    kinds += ["honest"] * (n_workers - len(kinds))
    kinds = np.array(rng.permutation(kinds))

    conf = np.zeros((n_workers, k, k))
    for j, kind in enumerate(kinds):
        if kind == "clicker":
            conf[j] = 1 / k
        elif kind == "lazy":
            conf[j, :, rng.integers(k)] = 1.0
        else:
            acc = rng.uniform(0.65, 0.92)
            conf[j] = (1 - acc) / (k - 1)
            np.fill_diagonal(conf[j], acc)
            if kind == "swapper":
                a, b = rng.choice(k, 2, replace=False)
                conf[j, a] = conf[j, b] = (1 - 0.85) / (k - 1)
                conf[j, a, b] = conf[j, b, a] = 0.85
                conf[j, a, a] = conf[j, b, b] = 0.0
                conf[j] /= conf[j].sum(axis=1, keepdims=True)

    speed = np.where(np.isin(kinds, ["clicker", "lazy"]), spam_speed, 1.0)
    pick_p = speed / speed.sum()
    rows = []
    for i in range(n_tasks):
        for j in rng.choice(n_workers, size=redundancy, replace=False, p=pick_p):
            rows.append((i, j, rng.choice(k, p=conf[j, truth[i]])))
    return Labels.from_triples(rows, n_classes=k), truth, kinds
