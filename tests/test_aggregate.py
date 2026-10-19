import numpy as np

from crowdcheck import Labels, dawid_skene, gold_weighted_vote, majority_vote, spammer_scores
from crowdcheck.simulate import make_job


def test_majority_vote_basic():
    labels = Labels.from_triples([(0, 0, 1), (0, 1, 1), (0, 2, 0), (1, 0, 2), (1, 1, 2), (1, 2, 2)], n_classes=3)
    assert list(majority_vote(labels)) == [1, 2]


def test_majority_ties_are_not_always_class_zero():
    triples = [(i, 0, 0) for i in range(200)] + [(i, 1, 1) for i in range(200)]
    picks = majority_vote(Labels.from_triples(triples, n_classes=2))
    assert 50 < picks.sum() < 150


def test_dawid_skene_beats_majority_with_spammers():
    labels, truth, _ = make_job(spam_share=0.3, seed=1)
    ds = dawid_skene(labels)[0]
    assert (ds == truth).mean() > (majority_vote(labels) == truth).mean() + 0.05


def test_dawid_skene_learns_a_swapper_instead_of_ignoring_them():
    # one reliable worker who always swaps classes 0 and 1, plus two honest ones
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, 600)
    rows = []
    for i, y in enumerate(truth):
        rows.append((i, 0, {0: 1, 1: 0, 2: 2}[y]))
        for w in (1, 2):
            rows.append((i, w, y if rng.random() < 0.75 else rng.integers(0, 3)))
    _, _, conf, _ = dawid_skene(Labels.from_triples(rows, n_classes=3))
    assert conf[0, 0, 1] > 0.9 and conf[0, 1, 0] > 0.9


def test_spammer_scores_separate_spam_from_honest():
    perfect = np.eye(3)[None]
    spammer = np.full((1, 3, 3), 1 / 3)
    lazy = np.zeros((1, 3, 3)); lazy[0, :, 2] = 1
    scores = spammer_scores(np.concatenate([perfect, spammer, lazy]))
    assert np.allclose(scores, [1.0, 0.0, 0.0])


def test_gold_weighted_vote_ignores_workers_at_chance():
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 2, 400)
    rows = []
    for i, y in enumerate(truth):
        rows.append((i, 0, y if rng.random() < 0.9 else 1 - y))   # good
        rows.append((i, 1, rng.integers(0, 2)))                    # random
        rows.append((i, 2, rng.integers(0, 2)))                    # random
    labels = Labels.from_triples(rows, n_classes=2)
    gold = {i: int(truth[i]) for i in range(60)}
    acc_gold = (gold_weighted_vote(labels, gold)[60:] == truth[60:]).mean()
    acc_mv = (majority_vote(labels)[60:] == truth[60:]).mean()
    assert acc_gold > acc_mv + 0.1
