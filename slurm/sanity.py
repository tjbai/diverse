import json
from pathlib import Path

import torch, numpy as np, pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from core.utils import (
    subsample, build_dpp_kernel, dpp_score, maj_correct,
    best_correct, dist_n, avg_cosine_sim, mbr_correct
)

sbert = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('/scratch4/jeisner1/tjbai/llama_orm')
rm = AutoModelForCausalLM.from_pretrained('/scratch4/jeisner1/tjbai/llama_orm', torch_dtype=torch.bfloat16).to('cuda')

rows = []
for temp in [0.7, 1.0]:
    with open(f'/home/tbai4/diverse/dumps/sample_math_val_t-{temp}.jsonl') as f:
        data = [json.loads(line) for line in f]

    kernels, qual_kernels = {}, {}
    for i, d in tqdm(enumerate(data), total=len(data), desc='caching kernels'):
        preds = [s['generation']['content'] for s in d['preds']]
        logps = [s['logprobs'] for s in d['preds']]
        kernels[i] = build_dpp_kernel(preds, sbert=sbert)
        qual_kernels[i] = build_dpp_kernel(preds, sbert=sbert, seq_logprobs=logps)

    rm_cache = {}
    for bsz in [4, 8, 16, 32]:
        for prob_idx, d in enumerate(tqdm(data)):
            for boot in range(50):
                batch = subsample(d['preds'], k=bsz, seed=prob_idx+boot)

                seqs = batch['content']
                idxs = batch['indices']
                logp = batch['logprobs']

                row = dict(
                    problem_id=prob_idx,
                    bootstrap_id=f"{prob_idx}_{boot}_{bsz}",
                    sample_size=bsz,
                    temp=temp,
                    dpp=dpp_score(kernels[prob_idx], idxs),
                    qual_dpp=dpp_score(qual_kernels[prob_idx], idxs),
                    cosine_sim=avg_cosine_sim(seqs, sbert),
                    dist_n=dist_n(seqs),
                    avg_log_prob=np.mean([np.mean(lp) for lp in logp]),
                )

                inputs = dict(
                    solutions = seqs,
                    gold=d['problem']['solution'],
                    problem=d['problem']['problem'],
                    rm=rm,
                    tokenizer=tokenizer,
                )
                row['maj_correct'] = maj_correct(**inputs)['correct']
                row['best_correct'] = best_correct(**inputs, cache=rm_cache)['correct']
                row['sem_mbr_correct'] = mbr_correct(**inputs, sbert=sbert)['correct']
                row['lex_mbr_correct'] = mbr_correct(**inputs)['correct']

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet('/home/tbai4/diverse/dumps/bootstrap_acc.parquet', index=False)
