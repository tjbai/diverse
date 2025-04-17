import json
import random
from pathlib import Path
from typing import Optional, List, Dict

random.seed(42)

def load_math(
    root_dir: str,
    split: str,
    problem_types: Optional[List[str]] = None,
    val_ratio: float = 0.1
) -> List[Dict]:
    root = Path(root_dir) / ('test' if split in ['val', 'test'] else split)

    if problem_types is None:
        problem_types = [d.name for d in root.iterdir() if d.is_dir()]

    problems = []
    for problem_type in problem_types:
        type_dir = root / problem_type
        if not type_dir.exists():
            print(f'Could not find {type_dir}')
            continue

        for prob_file in type_dir.glob("*.json"):
            with open(prob_file) as f:
                problem = json.load(f)
                problems.append(problem)

    random.shuffle(problems)
    split_idx = int(len(problems) * val_ratio)

    if split == 'val':
        problems = problems[:split_idx]
    elif split == 'test':
        problems = problems[split_idx:]

    return problems
