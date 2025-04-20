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

# %%
import pandas as pd
from tabulate import tabulate

df = pd.read_parquet('dumps/bootstrap_small_acc.parquet')

metrics   = ['dpp', 'qual_dpp', 'cosine_sim', 'dist_n', 'avg_log_prob']
accuracy  = ['maj_correct', 'best_correct', 'sem_mbr_correct', 'lex_mbr_correct']
keep_cols = metrics + accuracy

acc_by_n = df.groupby('sample_size')[accuracy].mean()
acc_by_temp = df.groupby('temp')[accuracy].mean()

def corr_block(sub):
    return sub[keep_cols].corr().loc[metrics, accuracy]

print(tabulate(acc_by_n, headers='keys', tablefmt='github'))
print(tabulate(acc_by_temp, headers='keys', tablefmt='github'))

corr_by_pair = {
    (temp, n): corr_block(group)
    for (temp, n), group in df.groupby(['temp', 'sample_size'])
}

for k, v in corr_by_pair.items():
    print(k)
    print(v)

import statsmodels.formula.api as smf

m = smf.logit('maj_correct ~ dpp*avg_log_prob', data=df).fit()

print(m.summary())

from pingouin import partial_corr
partial_corr(df, x='dpp', y='maj_correct', covar='avg_log_prob', method='spearman')

df['qbin'] = pd.qcut(df.avg_log_prob,4,labels=False)
print(df.groupby('qbin')['dpp'].corr(df.maj_correct))

plt.scatter(df[df['temp']==0.7]['avg_log_prob'], df[df['temp']==0.7]['dpp'], color='blue', s=df[df['temp']==0.7]['sample_size']*5, alpha=0.7, label='temp=0.7')
plt.scatter(df[df['temp']==1.0]['avg_log_prob'], df[df['temp']==1.0]['dpp'], color='red', s=df[df['temp']==1.0]['sample_size']*5, alpha=0.7, label='temp=1.0')
plt.title('MATH preliminary')
plt.xlabel('avg log prob')
plt.ylabel('dpp score')
plt.legend()
plt.show()
