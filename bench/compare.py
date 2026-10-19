"""Majority vote vs gold-weighted vote vs Dawid-Skene on simulated jobs.

    python bench/compare.py
"""

import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowdcheck import dawid_skene, gold_weighted_vote, majority_vote, spammer_scores  # noqa: E402
from crowdcheck.simulate import make_job  # noqa: E402

SEEDS = range(5)
GOLD_SHARE = 0.05


def run(spam_share, redundancy, seed):
    labels, truth, kinds = make_job(spam_share=spam_share, redundancy=redundancy, seed=seed)
    rng = np.random.default_rng(seed + 100)
    gold_ids = rng.choice(labels.n_tasks, size=int(GOLD_SHARE * labels.n_tasks), replace=False)
    gold = {int(t): int(truth[t]) for t in gold_ids}
    scored = np.setdiff1d(np.arange(labels.n_tasks), gold_ids)  # grade everyone on the same tasks

    mv = majority_vote(labels)
    gw = gold_weighted_vote(labels, gold)
    ds, _, conf, _ = dawid_skene(labels)

    spam = np.isin(kinds, ["clicker", "lazy"])
    flagged = spammer_scores(conf) < 0.25
    hit = (flagged & spam).sum()
    return {
        "majority": (mv[scored] == truth[scored]).mean(),
        "gold-weighted": (gw[scored] == truth[scored]).mean(),
        "dawid-skene": (ds[scored] == truth[scored]).mean(),
        "spam precision": hit / max(flagged.sum(), 1),
        "spam recall": hit / max(spam.sum(), 1),
    }


def main():
    print(f"2,000 tasks, 4 classes, 60 workers, 10% swappers, spammers 3x faster, {len(SEEDS)} seeds, {int(GOLD_SHARE*100)}% gold\n")
    print("| Spammers | Labels per task | Majority vote | Gold-weighted | Dawid-Skene | Spammers caught | Flags that were spammers |")
    print("|---|---|---|---|---|---|---|")
    for r in (3, 5):
        for s in (0.0, 0.1, 0.2, 0.3, 0.4):
            res = [run(s, r, seed) for seed in SEEDS]
            avg = {key: statistics.mean(x[key] for x in res) for key in res[0]}
            caught = "-" if s == 0 else f"{avg['spam recall']:.0%}"
            precise = "-" if s == 0 else f"{avg['spam precision']:.0%}"
            print(f"| {int(s*100)}% | {r} | {avg['majority']:.1%} | {avg['gold-weighted']:.1%} | {avg['dawid-skene']:.1%} "
                  f"| {caught} | {precise} |")


if __name__ == "__main__":
    main()
