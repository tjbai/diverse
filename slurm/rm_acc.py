import json

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.utils import is_correct

tokenizer = AutoTokenizer.from_pretrained('/scratch4/jeisner1/tjbai/llama_orm')
rm = AutoModelForCausalLM.from_pretrained('/scratch4/jeisner1/tjbai/llama_orm', torch_dtype=torch.bfloat16).to('cuda')
plus_id = 10

rows = []
for temp in [0.3, 0.5, 0.7, 1.0]:
    with open(f'/home/tbai4/diverse/dumps/sample_math_val_t-{temp}.jsonl') as f:
        data = [json.loads(line) for line in f]

    acc = 0
    agg_acc = 0
    for i, d in enumerate(tqdm(data)):
        input_texts = [
            tokenizer.apply_chat_template(
                [{'role': 'user', 'content': f'{d['problem']['problem']} {sol['generation']['content']}'}],
                add_generation_prompt=True, tokenize=False
            )
            for sol in d['preds']
        ]
        inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits = rm(**inputs).logits
            last = inputs.attention_mask.sum(1) - 1
            batch = torch.arange(logits.size(0), device=logits.device)
            p_plus = torch.softmax(logits[batch, last], 1)[:, plus_id].tolist()

        preds = np.array(p_plus) > 0.5
        is_correct = np.array([is_correct(
            sol['generation']['content'],
            d['problem']['problem'],
        ) for sol in d['preds']])
        agg_acc += int(sum(preds == is_correct))

        rows.append({
            'unit': d,
            'p_plus': p_plus,
            'acc': int(sum(preds == is_correct)),
        })

    print(f't={temp}, acc={agg_acc / (len(data) * 32)}')

    df = pd.DataFrame(rows)
    df.to_parquet('/home/tbai4/diverse/dumps/rm_acc.parquet', index=False)
