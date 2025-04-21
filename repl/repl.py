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

from pingouin import partial_corr
def par_corr_block(sub):
    return partial_corr(sub, x='dpp', y='maj_correct', covar='avg_log_prob', method='spearman')
    return sub[keep_cols].corr().loc[metrics, accuracy]

print(tabulate(acc_by_n, headers='keys', tablefmt='github'))
print(tabulate(acc_by_temp, headers='keys', tablefmt='github'))

corr_by_pair = {
    (temp, n): corr_block(group)
    for (temp, n), group in df.groupby(['temp', 'sample_size'])
}

par_corr_by_pair = {
    (temp, n): par_corr_block(group)
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

plt.figure(figsize=(12, 8), dpi=300)
plt.scatter(np.exp(-df[df['temp']==0.7]['avg_log_prob']), df[df['temp']==0.7]['dpp'], color='blue', s=df[df['temp']==0.7]['sample_size']*5, alpha=0.7, label='temp=0.7')
plt.scatter(np.exp(-df[df['temp']==1.0]['avg_log_prob']), df[df['temp']==1.0]['dpp'], color='red', s=df[df['temp']==1.0]['sample_size']*5, alpha=0.7, label='temp=1.0')
plt.xscale('log')
plt.title('MATH prelim batch likelihood vs. diversity')
plt.xlabel('ppl')
plt.ylabel('dpp score w/ sbert embeddings')
plt.legend()
plt.savefig('figures/batch_ll_vs_diversity.png')

# %%
import json
from statistics import correlation
from core.utils import is_correct, subsample, build_dpp_kernel, dpp_score
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('all-MiniLM-L6-v2')

for temp in [0.7]:
    with open(f'dumps/sample_math_val_t-{temp}.jsonl') as f:
        data = [json.loads(line) for line in f]
    kernels = {}
    for i, d in tqdm(enumerate(data), total=len(data), desc='caching kernels'):
        preds = [s['generation']['content'] for s in d['preds']]
        logps = [s['logprobs'] for s in d['preds']]
        kernels[i] = build_dpp_kernel(preds, sbert=sbert)
    for bsz in [4, 8, 16, 32]:
        xs = []
        ys = []
        for prob_idx, d in enumerate(tqdm(data)):
            batch = subsample(d['preds'], k=bsz, seed=prob_idx)
            num_correct = sum(is_correct(b, d['problem']['solution']) for b in batch['content'])
            dpp = dpp_score(kernels[prob_idx], batch['indices'])
            xs.append(dpp)
            ys.append(num_correct)
        print(f'bsz={bsz}', correlation(xs, ys))
        print(f'bsz={bsz}', correlation(xs, ys, method='ranked'))
