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

# # %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet('dumps/bootstrap_acc.parquet')

def bucket(df, num_buckets=4, sample_size=4, metric='dpp'):
    df = df[~np.isinf(df[metric])]
    df = df[df['sample_size'] == sample_size]
    lower_bound = np.percentile(df[metric], 2.5)
    upper_bound = np.percentile(df[metric], 97.5)
    df = df[(df[metric] >= lower_bound) & (df[metric] <= upper_bound)]
    bins = np.linspace(df[metric].min(), df[metric].max(), num_buckets + 1)
    df[f'{metric}_bucket'] = pd.cut(df[metric], bins=bins, labels=False, include_lowest=True)
    results = (
        df.groupby(f'{metric}_bucket')
          .agg(
            maj_correct=('maj_correct', 'mean'),
            best_correct=('best_correct', 'mean'),
            sem_mbr_correct=('sem_mbr_correct', 'mean'),
            lex_mbr_correct=('lex_mbr_correct', 'mean'),
            avg_log_prob=('avg_log_prob', 'mean'),
            avg_dpp=('dpp', 'mean'),
            min=(metric, 'min'),
            max=(metric, 'max'),
            mean=(metric, 'mean'),
            count=(metric, 'size'),
          )
    )
    return results

for sample_size in [4, 8, 16, 32]:
    b = bucket(df, num_buckets=12, sample_size=sample_size, metric='qual_dpp')
    x = np.arange(len(b))
    width = 0.2

    fig, ax_acc = plt.subplots(figsize=(12, 8))

    acc_metrics = ['maj_correct', 'best_correct', 'sem_mbr_correct', 'lex_mbr_correct']
    acc_labels = ['Majority', 'Best', 'SBERT MBR', 'BLEU MBR']
    colors = ['#F8B195', '#F67280', '#C06C84', '#6C5B7B']
    for i, (m, label, c) in enumerate(zip(acc_metrics, acc_labels, colors)):
        ax_acc.bar(x + (i - 1.5) * width, b[m], width, label=label, color=c)

    ax_acc.set_ylim(0, 1)
    ax_acc.set_xticks(x)
    bucket_labels = [f"n={b['count'][i]}, ({low:.1f}, {high:.1f})" for i, (low, high) in enumerate(zip(b['min'], b['max']))]
    ax_acc.set_xticklabels(bucket_labels, rotation=45, ha='right')
    ax_acc.set_xlabel('Quality-scaled SBERT DPP')
    ax_acc.set_ylabel('Accuracy')

    ax_dpp = ax_acc.twinx()
    ax_dpp.plot(x, b['avg_dpp'], 'o-', label='SBERT DPP', color='#355C7D')
    ax_dpp.set_ylabel('SBERT DPP')

    ax_lp = ax_acc.twinx()
    ax_lp.spines['right'].set_position(('outward', 60))
    ax_lp.plot(x, b['avg_log_prob'], 's--', label='Log-Likelihood', color='#2A363B')
    ax_lp.set_ylabel('Log-Likelihood')

    lines, labels = [], []
    for ax in (ax_acc, ax_dpp, ax_lp):
        for line in ax.get_lines() + ax.containers:
            label = getattr(line, 'get_label', lambda: None)()
            if label and not label.startswith('_'):
                lines.append(line)
                labels.append(label)
    ax_acc.legend(lines, labels, loc='upper left', ncol=2, framealpha=0.3, facecolor='lightgrey')

    plt.tight_layout()
    plt.savefig(f'figures/multidim_best_n-{sample_size}.png', dpi=300)

# %%
for sample_size in [4, 8, 16, 32]:
    for temp in [0.3, 0.5, 0.7, 1.0]:
        b = bucket(df[df['temp'] == temp], num_buckets=12, sample_size=4, metric='qual_dpp')
        x = np.arange(len(b))
        width = 0.2

        fig, ax_acc = plt.subplots(figsize=(12, 8))

        acc_metrics = ['maj_correct', 'best_correct', 'sem_mbr_correct', 'lex_mbr_correct']
        acc_labels = ['Majority', 'Best', 'SBERT MBR', 'BLEU MBR']
        colors = ['#F8B195', '#F67280', '#C06C84', '#6C5B7B']
        for i, (m, label, c) in enumerate(zip(acc_metrics, acc_labels, colors)):
            ax_acc.bar(x + (i - 1.5) * width, b[m], width, label=label, color=c)

        ax_acc.set_ylim(0, 1)
        ax_acc.set_xticks(x)
        bucket_labels = [f"n={b['count'][i]}, ({low:.1f}, {high:.1f})" for i, (low, high) in enumerate(zip(b['min'], b['max']))]
        ax_acc.set_xticklabels(bucket_labels, rotation=45, ha='right')
        ax_acc.set_xlabel('Quality-scaled SBERT DPP')
        ax_acc.set_ylabel('Accuracy')

        ax_dpp = ax_acc.twinx()
        ax_dpp.plot(x, b['avg_dpp'], 'o-', label='SBERT DPP', color='#355C7D')
        ax_dpp.set_ylabel('SBERT DPP')

        ax_lp = ax_acc.twinx()
        ax_lp.spines['right'].set_position(('outward', 60))
        ax_lp.plot(x, b['avg_log_prob'], 's--', label='Log-Likelihood', color='#2A363B')
        ax_lp.set_ylabel('Log-Likelihood')

        lines, labels = [], []
        for ax in (ax_acc, ax_dpp, ax_lp):
            for line in ax.get_lines() + ax.containers:
                label = getattr(line, 'get_label', lambda: None)()
                if label and not label.startswith('_'):
                    lines.append(line)
                    labels.append(label)
        ax_acc.legend(lines, labels, loc='upper left', ncol=2, framealpha=0.3, facecolor='lightgrey')

        plt.tight_layout()
        plt.savefig(f'figures/multidim_best_n-{sample_size}_t-{temp}.png', dpi=300)
