import json
import os
import sys

sys.path.insert(0, ".")
from skarm.datasets import load_transactions

with open('benchmarks/thresholds.json') as f:
    cfg = json.load(f)['datasets']

print(f'Checking all {len(cfg)} datasets...')
mismatches = []
for name, dcfg in cfg.items():
    fname = dcfg['file']
    p = os.path.join('data', fname)
    if not os.path.exists(p):
        print(f'ERROR: File {p} missing!')
        mismatches.append(name)
        continue
    tx = load_transactions(p)
    exp = dcfg['expected_n']
    matched = (len(tx) == exp)
    flag = 'OK' if matched else f'MISMATCH ({len(tx)} != {exp})'
    print(f'{name:22s} ({fname:30s}): {len(tx):>8,} tx [{flag}]')
    if not matched:
        mismatches.append(name)

print('-' * 60)
if not mismatches:
    print('ALL 25 DATASETS MATCH INVENTORY EXACTLY!')
else:
    print(f'MISMATCHES FOUND: {mismatches}')
