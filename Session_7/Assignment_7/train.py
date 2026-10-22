"""Reproducible small transformer experiment: predict the last character."""
import argparse
import json
import random
import time
import torch
from torch import nn
from dynamic_kronecker import run_proofs

class TinyTransformer(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.chars = nn.Embedding(2, width)
        self.position = nn.Embedding(129, width)
        layer = nn.TransformerEncoderLayer(width, 4, 64, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(width, 2)

    def forward(self, x, lengths):
        pos = torch.arange(x.shape[1], device=x.device)[None, :]
        mask = pos >= lengths[:, None]
        h = self.encoder(self.chars(x) + self.position(pos), src_key_padding_mask=mask)
        return self.head(h[torch.arange(x.shape[0], device=x.device), lengths-1])

def samples(batch, low, high, rng):
    lengths = [rng.randint(low, high) for _ in range(batch)]
    x = torch.zeros(batch, max(lengths), dtype=torch.long)
    y = torch.zeros(batch, dtype=torch.long)
    for i, n in enumerate(lengths):
        chars = [rng.randrange(2) for _ in range(n)]
        x[i, :n] = torch.tensor(chars)
        y[i] = chars[-1]
    return x, torch.tensor(lengths), y

@torch.no_grad()
def evaluate(model, rng):
    model.eval()
    correct = 0
    for _ in range(10):
        x, lengths, y = samples(48, 33, 96, rng)
        correct += (model(x, lengths).argmax(-1) == y).sum().item()
    return correct / 480

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results.json")
    a = p.parse_args()
    torch.set_num_threads(2)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    run_proofs()
    rng = random.Random(a.seed)
    model = TinyTransformer()
    optim = torch.optim.AdamW(model.parameters(), lr=.003)
    start = time.perf_counter()
    for step in range(a.steps):
        model.train()
        x, lengths, y = samples(a.batch, 8, 96, rng)
        optim.zero_grad()
        loss = nn.functional.cross_entropy(model(x, lengths), y)
        loss.backward()
        optim.step()
        if (step+1) % 50 == 0:
            print(f"step={step+1}, loss={loss.item():.4f}")
    data = dict(problem=3, steps=a.steps, seed=a.seed, heldout_length_range=[33,96],
                dynamic_accuracy=evaluate(model, random.Random(a.seed+1)),
                fixed32_theoretical_accuracy=.5,
                elapsed_seconds=round(time.perf_counter()-start, 2),
                caveat="Toy last-character task; not a language model benchmark")
    with open(a.out, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
