import os
import json

from tqdm import tqdm

from llama import Llama
from core.utils import free_port
from core.data import load_math

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = free_port()

llama = Llama.build(
   ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
   tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
   max_seq_len=1024,
   max_batch_size=32,
   model_parallel_size=1,
)

problems = load_math(root_dir='/home/tbai4/diverse/data/MATH', split='val')[:500]

BSZ = 4
TOP_P = 1.0
TEMPS = [0.3, 0.5, 0.7, 1.0]
MAX_GEN = 1024

for temp in TEMPS:
    with open(f'/home/tbai4/diverse/dumps/sample_math_train_t-{temp}.jsonl', 'a+') as f:
        f.seek(0)
        existing = set(json.loads(line)['problem']['problem'] for line in f)

        for problem in tqdm(problems, desc=f'temp={temp}'):
            if problem['problem'] in existing:
                continue
            preds = llama.chat_completion(
                dialogs=[[{'role': 'user', 'content': f'Solve the following math problem step-by-step: {problem['problem']}\nPresent the answer in LaTex format: \\boxed{{Your answer}}'}] for _ in range(BSZ)],
                temperature=temp,
                top_p=TOP_P,
                max_gen_len=MAX_GEN,
                logprobs=True,
            )

            f.write(json.dumps({
                'problem': problem,
                'preds': preds,
                'temperature': temp,
            }))
            f.write('\n')
            existing.add(problem['problem'])
