"""A batch of crowd labels: who said what about which task."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Labels:
    task: np.ndarray    # task index for each label
    worker: np.ndarray  # worker index for each label
    label: np.ndarray   # class the worker picked, 0..n_classes-1
    n_tasks: int
    n_workers: int
    n_classes: int

    @classmethod
    def from_triples(cls, triples, n_classes=None):
        arr = np.asarray(triples, dtype=int).reshape(-1, 3)
        return cls(
            task=arr[:, 0], worker=arr[:, 1], label=arr[:, 2],
            n_tasks=int(arr[:, 0].max()) + 1,
            n_workers=int(arr[:, 1].max()) + 1,
            n_classes=n_classes or int(arr[:, 2].max()) + 1,
        )

    def counts(self):
        """counts[i, k] = how many workers gave task i the label k."""
        c = np.zeros((self.n_tasks, self.n_classes))
        np.add.at(c, (self.task, self.label), 1)
        return c

    def labels_per_worker(self):
        return np.bincount(self.worker, minlength=self.n_workers)
