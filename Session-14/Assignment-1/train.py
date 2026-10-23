"""Session 14: linear regression -> dense soft-gated Mixture of Experts.
Run with: python train.py
Requires: pip install torch
"""
import csv
from pathlib import Path
import torch

torch.set_num_threads(2)
torch.manual_seed(14)

N, D = 1024, 8
x = torch.randn(N, D)
w = torch.tensor([1.2, -0.7, 0.5, 0.3, -0.2, 0.8, -0.4, 0.6])
y = x @ w + 0.9 * torch.relu(x[:, 0]) + 0.5 * (x[:, 1] > 0).float() * x[:, 2]
train_x, val_x = x[:800], x[800:]
train_y, val_y = y[:800], y[800:]

# Reset for reproducible baseline initialization, separately from data generation.
torch.manual_seed(14)
linear = torch.nn.Linear(D, 1)
optim = torch.optim.Adam(linear.parameters(), lr=0.025)
records = []

def measure(model, phase, step):
    model.eval()
    with torch.no_grad():
        tr = ((model(train_x).squeeze(-1) - train_y) ** 2).mean().item()
        va = ((model(val_x).squeeze(-1) - val_y) ** 2).mean().item()
    model.train()
    row = dict(phase=phase, step=step, train_mse=round(tr, 6), val_mse=round(va, 6))
    records.append(row)
    print(f"{phase:6s} step={step:3d} train_mse={tr:.6f} val_mse={va:.6f}")

for step in range(121):
    if step:
        optim.zero_grad()
        loss = ((linear(train_x).squeeze(-1) - train_y) ** 2).mean()
        loss.backward()
        optim.step()
    if step % 20 == 0:
        measure(linear, "linear", step)

class MoE(torch.nn.Module):
    """Dense soft routing: every expert participates in every prediction."""
    def __init__(self, source, expert_count=3):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            torch.nn.Linear(D, 1) for _ in range(expert_count)
        )
        with torch.no_grad():
            for expert in self.experts:
                expert.weight.copy_(source.weight)
                expert.bias.copy_(source.bias)
        self.gate = torch.nn.Linear(D, expert_count)
        # A nonuniform router breaks expert-gradient symmetry.
        # Experts remain identical at conversion, so predictions are unchanged.
        torch.nn.init.normal_(self.gate.weight, mean=0.0, std=0.4)
        torch.nn.init.zeros_(self.gate.bias)

    def forward(self, z):
        probs = torch.softmax(self.gate(z), dim=-1)
        outputs = torch.cat([expert(z) for expert in self.experts], dim=-1)
        return (probs * outputs).sum(dim=-1).unsqueeze(-1)

moe = MoE(linear)
with torch.no_grad():
    max_error = (moe(val_x).squeeze(-1) - linear(val_x).squeeze(-1)).abs().max().item()
print(f"Conversion max absolute output difference: {max_error:.9g}")
assert max_error < 2e-6

optim = torch.optim.Adam(moe.parameters(), lr=0.025)
for step in range(241):
    if step:
        optim.zero_grad()
        loss = ((moe(train_x).squeeze(-1) - train_y) ** 2).mean()
        loss.backward()
        optim.step()
    if step % 40 == 0:
        measure(moe, "moe", step)

out = Path(__file__).with_name("training_logs.csv")
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["phase", "step", "train_mse", "val_mse"])
    writer.writeheader()
    writer.writerows(records)
print("Saved:", out)
