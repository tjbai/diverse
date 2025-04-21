import os
import json
import math
from typing import Optional, Tuple, Dict
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm

from llama import Llama
from llama.tokenizer import Tokenizer, ChatFormat
from core.utils import free_port, is_correct

class RmDataset(Dataset):
    def __init__(self, data_path: str | Path, tokenizer: Tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.format = ChatFormat(tokenizer)
        self.max_seq_len = max_seq_len
        with open(data_path) as f:
            _data = json.load(f)

        self.data = []
        for d in _data:
            for s in d['preds']:
                try:
                    tokens = self.format.encode_dialog_prompt([{
                        'role': 'user',
                        'content': f"{d['problem']['problem']} {s['generation']['content']}"
                    }])
                    if len(tokens) > self.max_seq_len:
                        tokens = tokens[:self.max_seq_len]

                    attn_mask = [1] * len(tokens)
                    padding = self.max_seq_len - len(tokens)
                    if padding > 0:
                        tokens.extend([0] * padding)
                        attn_mask.extend([0] * padding)

                    self.data.append({
                        'input_tokens': tokens,
                        'attn_mask': attn_mask,
                        'label': 10 if is_correct(s['generation']['content'], d['problem']['solution']) else 12
                    })
                except:
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class LoraTrainer:
    def __init__(self, llama: Llama, output_dir: str, learning_rate: float):
        self.llama = llama
        self.model = llama.model
        self.tokenizer = self.llama.tokenizer
        self.optimizer = AdamW(self.model.get_trainable_parameters(), lr=learning_rate)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in self.model.parameters())
        print(f"Training {num_trainable/1e6:.1f}M / {num_total/1e9:.1f}B parameters")

    def save_checkpoint(self, global_step: int):
        torch.save({
            "trainable_params": [p for p in self.model.get_trainable_parameters()],
            "optimizer": self.optimizer.state_dict()
        }, self.output_dir / f'lora_step-{global_step}.pt')

    def step(self, data: Dict) -> Tuple[torch.Tensor, Dict]:
        logits = self.model(data['input_tokens'], mask=data['attn_mask'])
        last_token_pos = data['attn_mask'].sum(dim=1) - 1
        last_token_logits = logits[torch.arange(logits.size(0)), last_token_pos]
        loss = F.cross_entropy(last_token_logits, data['label'])
        return loss, {'train/loss': loss.item()}

    def evaluate(self, val_loader: DataLoader, max_steps: Optional[int] = int(1e9)) -> Dict:
        return {}

def get_lr_factor(step, warmup_steps=10, total_steps=100):
    # steps = number of updates != number of samples
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

def set_model_env():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = free_port()

def training_schedule(
    epochs: Optional[int],
    steps: Optional[int],
    dataset_size: int,
    gradient_accumulation_steps: int
) -> Tuple[int, int, int]:
    if epochs is not None:
        total_steps = epochs * dataset_size // gradient_accumulation_steps
        warmup_steps = total_steps // 10
        return epochs, total_steps, warmup_steps

    assert steps is not None
    epochs = math.ceil(steps * gradient_accumulation_steps / dataset_size)
    warmup_steps = steps // 10
    return epochs, steps, warmup_steps

def finetune(
    data_path: str,
    ckpt_dir: str,
    tokenizer_path: str,
    task: str,
    output_dir: str = "checkpoints",
    max_batch_size: int = 32,
    max_seq_len: int = 2048,
    epochs: Optional[int] = 2,
    steps: Optional[int] = None,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 1,
    checkpoint_freq: int = 25,
    validation_freq: int = 25,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    set_model_env()
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=32,
        model_parallel_size=1,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    tokenizer = Tokenizer(tokenizer_path)
    dataset = RmDataset(data_path, tokenizer, max_seq_len=max_seq_len)
    trainer = LoraTrainer(llama, output_dir, learning_rate)

    generator = torch.Generator(device="cuda").manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=max_batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=max_batch_size, shuffle=False)
    assert len(train_dataset) > 0 and len(val_dataset) > 0
    print(f'Train Dataset: {len(train_dataset)} samples')
    print(f'Val Dataset: {len(val_dataset)} samples')

    print("Sanity check:")
    trainer.evaluate(val_loader, max_steps=1)
    print("Passed!")

    epochs, steps, warmup_steps = training_schedule(
        epochs=epochs,
        steps=steps,
        dataset_size=len(train_dataset),
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    print(f'Epochs: {epochs}, Steps: {steps}, Warmup: {warmup_steps}')
    validation_freq = min(validation_freq, steps // 2)
    print(f'Validation freq: {validation_freq}')

    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            lr_factor = get_lr_factor(global_step, warmup_steps, steps)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_factor
            try:
                step_result = trainer.step(batch)
            except RuntimeError as e:
                if 'cuda out of memory' in str(e).lower():
                    step_result = None
                else:
                    raise
            if step_result is None:
                continue # will create jagged batches
            loss, metrics = step_result
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                global_step += 1
                if (global_step + 1) % int(validation_freq) == 0:
                    wandb.log(trainer.evaluate(val_dataset))
                metrics.update({'lr': lr_factor})
                wandb.log(metrics)
                if (global_step + 1) % int(checkpoint_freq) == 0:
                    trainer.save_checkpoint(global_step)
                if steps is not None and global_step == steps:
                    break

    trainer.save_checkpoint(global_step)
    wandb.finish()
