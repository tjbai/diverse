# %%
import json
from core.utils import parse_math
from collections import defaultdict

with open('dumps/sample_math_mid.json') as f:
    data = json.load(f)

freq = defaultdict(int)
for d in data:
    for pred in d['preds']:
        freq[len(parse_math(pred['generation']['content']))] += 1

# >> defaultdict(<class 'int'>, {2: 1585, 0: 13, 1: 2})

# %%
import json
from core.utils import parse_math, subsample, maj_correct
from collections import defaultdict

with open('dumps/sample_math_mid.json') as f:
    data = json.load(f)

solutions = subsample(data[0]['preds'], k=4)
gold = data[0]['problem']['solution']

maj_correct(solutions, gold)

# >> {'correct': True, 'answers': [(68, '68'), (68, '68'), (68, '68'), (68, '68')], 'gold': [68, '68']}
