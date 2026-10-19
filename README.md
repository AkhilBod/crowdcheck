# crowdcheck

When a bunch of people label the same data, which answers do you trust? And
which people? This compares three ways of deciding, on simulated labeling
jobs where some workers are honest, some are consistently confused, and some
are just clicking.

- **Majority vote:** go with what most people said.
- **Gold-weighted vote:** sneak in a few tasks you already know the answer to
  (5% here), grade each worker on those, and weight their votes by that grade.
- **Dawid-Skene:** no answer key at all. It estimates each worker's habits
  (a full "when the answer is X, they say Y" table) and the true labels
  together, using EM. From Dawid & Skene, 1979.

It also flags spammers: workers whose answers don't depend on the question,
measured with the spammer score from Raykar & Yu (2012).

About 200 lines of Python on top of numpy.

```
pip install -e ".[test]"
pytest
python bench/compare.py
```

## The fake job

2,000 tasks with 4 classes (one class is more common than the others), 60
workers, 3 or 5 labels per task. Workers come in four kinds:

| Kind | Behavior |
|---|---|
| honest | right 65-92% of the time, mistakes spread out |
| swapper | honest, but always mixes up two specific classes |
| clicker | random answers |
| lazy | same answer every time |

Spammers (clickers and lazy workers) are 3x faster, so they do more than
their share. At 20% of workers, they end up writing about 42% of all labels.

## Results

Accuracy is measured on the 95% of tasks that weren't gold. Averaged over 5
seeds. "Spammers caught" is recall; "flags that were spammers" is precision,
with a score threshold of 0.25.

| Spammers | Labels per task | Majority vote | Gold-weighted | Dawid-Skene | Spammers caught | Flags that were spammers |
|---|---|---|---|---|---|---|
| 0% | 3 | 85.8% | 86.5% | 89.2% | - | - |
| 10% | 3 | 73.4% | 79.7% | 82.8% | 100% | 77% |
| 20% | 3 | 62.6% | 72.4% | 75.7% | 100% | 87% |
| 30% | 3 | 53.1% | 65.7% | 68.9% | 100% | 94% |
| 40% | 3 | 46.2% | 56.5% | 58.9% | 99% | 94% |
| 0% | 5 | 93.6% | 94.7% | 96.2% | - | - |
| 10% | 5 | 83.2% | 89.3% | 92.3% | 100% | 89% |
| 20% | 5 | 71.9% | 83.8% | 86.8% | 100% | 93% |
| 30% | 5 | 62.8% | 77.8% | 82.8% | 100% | 94% |
| 40% | 5 | 54.1% | 71.0% | 75.7% | 100% | 94% |

What stood out:

- Majority vote falls apart fast. With 20% spammers and 3 labels per task it
  gets 62.6%, because fast spammers often make up two of the three votes.
- Dawid-Skene, with no answer key, beats grading workers on gold tasks. It gets
  to use every label as evidence about every worker, not just the 5% of tasks
  with known answers.
- Swappers are worth keeping. Dawid-Skene learns "this person says B when the
  answer is A" and flips it back, so a consistently wrong worker becomes useful.
  A test checks it recovers a swap at over 90% confidence.
- The spam flag catches almost every spammer. The false flags are honest
  workers at the low end (around 65% accurate) whose score lands near the
  threshold. Dropping flagged workers and re-running Dawid-Skene only adds
  about 1.6 points, since it already gives spammers almost no weight.
- More labels per task helps every method, but it's the most expensive fix.

## Limits

- The workers are simulated. Real workers drift, learn, collude, and copy each
  other; none of that is modeled.
- Every task is equally hard. Real tasks aren't, and some models (GLAD) handle that.
- Workers are assigned at random. Real platforms route tasks, which changes
  what each worker sees.
