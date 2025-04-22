# %%
import pandas as pd
from tabulate import tabulate
import statsmodels.formula.api as smf
import statsmodels as sm
import json

def tab(*args, **kwargs):
    print(tabulate(*args, **kwargs, headers='keys', tablefmt='github'))

df = pd.read_parquet('dumps/bootstrap_acc.parquet')
metrics   = ['dpp', 'qual_dpp', 'cosine_sim', 'dist_n', 'avg_log_prob']
accuracy  = ['maj_correct', 'best_correct', 'sem_mbr_correct', 'lex_mbr_correct']
keep_cols = metrics + accuracy
acc = df.groupby(['sample_size', 'temp'])[accuracy].mean()
corr = df.groupby(['sample_size'])
df['maj_correct'] = df['maj_correct'].astype(int)

from pingouin import partial_corr
def par_corr_block(sub):
    return {
        'maj_correct': partial_corr(sub, x='dpp', y='maj_correct', covar='avg_log_prob', method='spearman')['r'].iloc[0],
        'best_correct': partial_corr(sub, x='dpp', y='best_correct', covar='avg_log_prob', method='spearman')['r'].iloc[0],
        'sem_mbr_correct': partial_corr(sub, x='dpp', y='sem_mbr_correct', covar='avg_log_prob', method='spearman')['r'].iloc[0],
        'lex_mbr_correct': partial_corr(sub, x='dpp', y='lex_mbr_correct', covar='avg_log_prob', method='spearman')['r'].iloc[0],
    }

import statsmodels.api as sm
def logistic_par_corr_block(sub):
    results = {}
    for outcome in ['maj_correct', 'best_correct', 'sem_mbr_correct', 'lex_mbr_correct']:
        model = sm.Logit(sub[outcome], sm.add_constant(sub[['dpp', 'avg_log_prob']])).fit(disp=0)
        results[outcome] = model.params['dpp']
    return results

par_corr_by_pair = {
    (temp, n): logistic_par_corr_block(group)
    for (temp, n), group in df.groupby(['temp', 'sample_size'])
}

for k, v in par_corr_by_pair.items():
    temp, n = k
    df_result = pd.DataFrame([v])
    df_result.insert(0, 'n', n)
    df_result.insert(0, 'temp', temp)
    tab(df_result)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bucket(df, num_buckets=4, sample_size=4, metric='dpp'):
    df = df[~np.isinf(df[metric])]
    df = df[df['sample_size'] == sample_size]
    bins = np.linspace(df[metric].min(), df[metric].max(), num_buckets + 1)
    df[f'{metric}_bucket'] = pd.cut(df[metric], bins=bins, labels=False, include_lowest=True)
    results = (
        df.groupby(f'{metric}_bucket')
          .agg(
            maj_correct=('maj_correct', 'mean'),
            best_correct=('best_correct', 'mean'),
            sem_mbr_correct=('sem_mbr_correct', 'mean'),
            lex_mbr_correct=('lex_mbr_correct', 'mean'),
            min=(metric, 'min'),
            max=(metric, 'max'),
            mean=(metric, 'mean'),
            count=(metric, 'size'),
          )
    )
    return results

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
temperatures = [0.3, 0.5, 0.7, 1.0]
for sample_size in [4, 8, 16, 32]:
    df_filtered_by_size = df[df['sample_size'] == sample_size]
    for y_col in ['dpp', 'avg_log_prob']:
        fig, ax = plt.subplots(figsize=(8, 6))
        for temp in [1.0, 0.7, 0.5, 0.3]:
            df_temp = df_filtered_by_size[df_filtered_by_size['temp'] == temp]
            df_deduped = df_temp.drop_duplicates(subset=['problem_id'], keep='first')
            ax.scatter(df_deduped[y_col], df_deduped['qual_dpp'], alpha=0.4, label=f'temp={temp}')
        ax.set_xlabel(f'{y_col}')
        ax.set_ylabel('qual_dpp')
        ax.set_title(f'batch_size={sample_size}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        fig.savefig(f'figures/qual_dpp_vs_{y_col}_{sample_size}.png', dpi=300)
        plt.close(fig)
