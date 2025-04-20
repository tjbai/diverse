import socket
import random
from collections import Counter
from typing import List, Dict, Optional

import torch
import numpy as np

from math_verify import LatexExtractionConfig, parse, verify
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sacrebleu import sentence_bleu

from llama.generation import ChatPrediction

def free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        return str(port)

def parse_math(s: str):
    return parse(s, extraction_config=[LatexExtractionConfig()])

def sym_bleu(a: str, b: str) -> float:
    return 1/2 * (
        sentence_bleu(a, [b]).score +
        sentence_bleu(b, [a]).score
    ) / 100.0

def subsample(preds: List[ChatPrediction], k: int, seed: int = 42) -> Dict:
    indices = random.Random(seed).sample(list(range(len(preds))), k=k)
    return {
        'content': [preds[i]['generation']['content'] for i in indices],
        'logprobs': [preds[i]['logprobs'] for i in indices],
        'indices': indices,
    }

def best_correct(
    solutions: List[str],
    gold: str,
    problem: str,
    rm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cache: Optional[Dict] = None,
    **_
) -> Dict:
    cache = {} if cache is None else cache
    plus_id = 10

    missing, probs = [], []
    key = lambda sol: (problem, sol)
    for sol in solutions:
        if key(sol) in cache:
            probs.append(cache[key(sol)])
        else:
            probs.append(None)
            missing.append(sol)

    if missing:
        input_texts = [
            tokenizer.apply_chat_template(
                [{'role': 'user', 'content': f'{problem} {sol}'}],
                add_generation_prompt=True, tokenize=False
            )
            for sol in missing
        ]
        inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits = rm(**inputs).logits
            last = inputs.attention_mask.sum(1) - 1
            batch = torch.arange(logits.size(0), device=logits.device)
            p_plus = torch.softmax(logits[batch, last], 1)[:, plus_id].tolist()

        for sol, p in zip(missing, p_plus):
            cache[key(sol)] = p
        it = iter(p_plus)
        probs = [p if p is not None else next(it) for p in probs]

    best = int(np.argmax(probs))
    answers = [parse_math(sol) for sol in solutions]
    gold_ans = parse_math(gold)
    return {
        'correct': verify(answers[best], gold_ans),
        'answers': answers,
        'gold': gold_ans,
    }

def maj_correct(solutions: List[str], gold: str, **_) -> Dict:
    gold_ans = parse_math(gold)

    # matrix edge-case
    def to_hashable(matrix_list):
        return tuple(tuple(tuple(row) for row in matrix.tolist()) if hasattr(matrix, 'tolist') else matrix for matrix in matrix_list)

    answers = [to_hashable(parse_math(sol)) for sol in solutions]

    try:
        freq = Counter(answers)
        mode, cnt = freq.most_common(1)[0]
    except IndexError:
        print('encountered', answers)
        raise

    return {
        'correct': False if list(freq.values()).count(cnt) > 1 else verify(list(mode), gold_ans),
        'answers': answers,
        'gold': gold_ans,
    }

def mbr_correct(
    solutions: List[str],
    gold: str,
    sbert: Optional[SentenceTransformer] = None,
    **_
) -> Dict:
    if sbert is not None:
        emb = sbert.encode(solutions, normalize_embeddings=True)
        U = emb @ emb.T
    else:
        n = len(solutions)
        U = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                s = sym_bleu(solutions[i], solutions[j])
                U[i, j] = U[j, i] = s
        np.fill_diagonal(U, 1.0)

    scores = U.mean(axis=1)
    best = int(np.argmax(scores))
    answers = [parse_math(sol) for sol in solutions]
    gold_ans = parse_math(gold)
    out = {
        'correct': verify(answers[best], gold_ans),
        'gold': gold_ans,
        'best': best,
    }

    return out

def dist_n(strs: List[str], n: int = 3) -> float:
    all_ngrams = []
    for s in strs:
        all_ngrams.extend(list(ngrams(word_tokenize(s), n)))
    return len(set(all_ngrams)) / len(all_ngrams)

def avg_cosine_sim(seqs: List[str], sbert: SentenceTransformer) -> float:
    embds = sbert.encode(seqs)
    return torch.mean(torch.tril(sbert.similarity(embds, embds), diagonal=-1)).item()

def build_dpp_kernel(
    seqs: List[str],
    sbert: SentenceTransformer,
    seq_logprobs: Optional[List[List[float]]] = None,
    normalize: bool = False,
) -> Dict:
    emb = sbert.encode(seqs, convert_to_numpy=True, normalize_embeddings=True)
    if seq_logprobs is None: return emb.T
    q = np.exp(0.5 * np.array([np.mean(lp) for lp in seq_logprobs]))
    emb = emb * q[:, None]
    return emb.T

def dpp_score(kernel: np.ndarray, indices: List[int]) -> float:
    B = kernel[:, indices]
    _, score = np.linalg.slogdet(B.T @ B)
    return score
