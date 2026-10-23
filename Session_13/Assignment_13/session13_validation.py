import copy
import torch
import torch.nn as nn
from session13_core import ReversibleBlock, ReversibleSequence

def reversibility_test(device="cuda" if torch.cuda.is_available() else "cpu"):
    torch.manual_seed(123)
    d, h, depth, B, T = 64, 4, 3, 2, 32

    blocks = nn.ModuleList([ReversibleBlock(d, h) for _ in range(depth)]).to(device)
    seq = ReversibleSequence(blocks).to(device)
    ref_blocks = copy.deepcopy(blocks)

    x1 = torch.randn(B, T, d, device=device, requires_grad=True)
    x2 = torch.randn(B, T, d, device=device, requires_grad=True)
    x1_ref = x1.detach().clone().requires_grad_(True)
    x2_ref = x2.detach().clone().requires_grad_(True)

    y1, y2 = seq(x1, x2)

    with torch.no_grad():
        r1, r2 = y1, y2
        for block in reversed(blocks):
            prev2 = r2 - block.g(r1)
            prev1 = r1 - block.f(prev2)
            r1, r2 = prev1, prev2
        reconstruction_error = max(
            (r1 - x1).abs().max().item(),
            (r2 - x2).abs().max().item(),
        )

    loss = (y1 + y2).pow(2).mean()
    loss.backward()

    a, b = x1_ref, x2_ref
    for block in ref_blocks:
        na = a + block.f(b)
        nb = b + block.g(na)
        a, b = na, nb
    ref_loss = (a + b).pow(2).mean()
    ref_loss.backward()

    input_grad_error = max(
        (x1.grad - x1_ref.grad).abs().max().item(),
        (x2.grad - x2_ref.grad).abs().max().item(),
    )

    parameter_grad_error = max(
        (p.grad - q.grad).abs().max().item()
        for p, q in zip(blocks.parameters(), ref_blocks.parameters())
    )

    return {
        "reconstruction_error": reconstruction_error,
        "input_grad_error": input_grad_error,
        "parameter_grad_error": parameter_grad_error,
    }
