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

# %%
import json
import numpy as np
import matplotlib.pyplot as plt

with open('dumps/sample_math_val_t-1.0.jsonl') as f:
    data = [json.loads(line) for line in f]

ll = [np.mean(s['logprobs']) for d in data for s in d['preds']]
plt.hist(ll)
plt.show()

from core.utils import maj_correct, subsample
from tqdm import tqdm

bsz = 4
inputs = []
batches = []
for i, d in enumerate(data):
    for j in range(20):
        batch = subsample(d['preds'], k=bsz, seed=i+j)
        batches.append(batch)
        inputs.append({
            'solutions': batch['content'],
            'gold': d['problem']['solution'],
        })

maj_res = [maj_correct(**s) for s in tqdm(inputs)]
