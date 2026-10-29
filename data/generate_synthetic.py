"""Generate two SYNTHETIC transaction datasets for the skarm capstone benchmarks.

These files are NOT downloads: they are generated locally by this script, so that
the benchmark suite includes controlled sparse and dense cases with known
ground truth. Both files are fully reproducible from the fixed seed below.

  python3 generate_synthetic.py

Outputs (in this same directory):
  synthetic_sparse_2000.dat  — N=2000, M=500 items, mean length ~8 (sparse)
  synthetic_dense_5000.dat   — N=5000, M=50 items,  mean length ~30 (dense)

Method
------
* Seed: 42 (Python `random.Random(42)` — deterministic across runs).
* Sparse set: transaction lengths drawn from Poisson(8), clipped to [1, 20];
  items drawn with Zipf-like weights w(i) = 1/(i+1), so a few items are
  frequent and most are rare (retail-like skew). One planted 3-itemset
  {1, 2, 3} is injected into 20% of transactions as known ground truth.
* Dense set: transaction lengths drawn from Normal(30, 5), clipped to [15, 45];
  items drawn uniformly from 50 items. Two planted itemsets, {1, 2} and
  {3, 4, 5}, are injected into 30% of transactions each.
* Items within a transaction are deduplicated and sorted ascending.
* Format: one transaction per line, space-separated integer item ids,
  exactly like the FIMI .dat files used elsewhere in this collection.
"""
import math
import random

SEED = 42
OUT_DIR = __file__.rsplit('/', 1)[0]


def poisson(rng, lam):
    # Knuth's algorithm; fine for lam=8.
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def gen_sparse(path, rng):
    n, m = 2000, 500
    weights = [1.0 / (i + 1) for i in range(m)]
    pop = list(range(1, m + 1))
    with open(path, 'w') as f:
        for _ in range(n):
            ln = min(20, max(1, poisson(rng, 8)))
            toks = set(rng.choices(pop, weights=weights, k=ln))
            if rng.random() < 0.20:
                toks.update({1, 2, 3})
            f.write(' '.join(str(t) for t in sorted(toks)) + '\n')


def gen_dense(path, rng):
    n, m = 5000, 50
    with open(path, 'w') as f:
        for _ in range(n):
            ln = min(45, max(15, round(rng.gauss(30, 5))))
            toks = set(rng.sample(range(1, m + 1), ln))
            if rng.random() < 0.30:
                toks.update({1, 2})
            if rng.random() < 0.30:
                toks.update({3, 4, 5})
            f.write(' '.join(str(t) for t in sorted(toks)) + '\n')


if __name__ == '__main__':
    rng = random.Random(SEED)
    gen_sparse(f'{OUT_DIR}/synthetic_sparse_2000.dat', rng)
    gen_dense(f'{OUT_DIR}/synthetic_dense_5000.dat', rng)
    print('wrote synthetic_sparse_2000.dat and synthetic_dense_5000.dat (seed 42)')
