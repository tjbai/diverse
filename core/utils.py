import socket
import random
from typing import List, Dict
from collections import Counter

import torch

from math_verify import LatexExtractionConfig, parse, verify
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama.generation import ChatPrediction

def free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        return str(port)

def parse_math(s: str):
    return parse(s, extraction_config=[LatexExtractionConfig()])

def subsample(preds: List[ChatPrediction], k: int, seed: int = 42):
    return random.Random(seed).sample([p['generation']['content'] for p in preds], k=k)

def best_correct(
    solutions: List[str],
    gold: str,
    problem: str,
    rm: AutoModelForCausalLM, # assumed RLHFlow/Llama3.1-8B-ORM-Mistral-Data
    tokenizer: AutoTokenizer
) -> Dict:
    plus_id = 10

    input_texts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': f'{problem} {sol}'}],
        add_generation_prompt=True,
        tokenize=False
    ) for sol in solutions]

    inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = rm(**inputs)
        logits = outputs.logits
        last_positions = inputs.attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        probs = torch.softmax(logits[batch_indices, last_positions], dim=1)
        best = torch.argmax(probs[:, plus_id]).item()

    answers = [parse_math(sol) for sol in solutions]
    gold_ans = parse_math(gold)

    return {
        'correct': verify(answers[best], gold_ans),
        'answers': answers,
        'gold': gold_ans,
    }

def maj_correct(solutions: List[str], gold: str) -> Dict:
    gold_ans = parse_math(gold)
    answers = [tuple(parse_math(sol)) for sol in solutions]
    freq = Counter(answers)
    mode, cnt = freq.most_common(1)[0]
    return {
        'correct': False if list(freq.values()).count(cnt) > 1 else verify(list(mode), gold_ans),
        'answers': answers,
        'gold': gold_ans,
    }

def mbr_correct(solutions: List[str]):
    pass
