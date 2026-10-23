# Session 14 — From Linear Model to Mixture of Experts

## Objective
I trained a linear regression model and then **converted its learned weights** into a 3-expert Mixture of Experts (MoE). I continued training, verified that loss continued to decrease, and committed the **actual training logs**.

## My setup
- Framework: PyTorch; CPU experiment, fixed seed 14, 2 CPU threads
- Data: 1,024 reproducibly generated synthetic regression samples, 8 features
- Split: first 800 samples for training; last 224 for validation
- Target: linear component with ReLU and feature-interaction nonlinear terms, allowing MoE to outperform a single linear model
- Baseline: one `nn.Linear(8, 1)`, Adam (learning rate 0.025), 120 updates
- Converted MoE: 3 `nn.Linear(8, 1)` experts, input-dependent softmax router, all experts active (dense MoE), Adam (learning rate 0.025), 240 additional updates
- Metric: mean squared error (MSE)

## Conversion method
I copied the trained linear layer's weight and bias into **each** expert. The gate receives the same input and computes three softmax probabilities. The MoE prediction is the probability-weighted sum of the three expert outputs. Because all experts have identical outputs immediately after copying, the new MoE returns the same prediction as the linear baseline, independent of the initial router probabilities.

I used a **random nonuniform router initialization** to break gradient symmetry. A zero-initialized, uniform gate with identical experts can trap the model in a symmetric solution, yielding virtually no specialization. After conversion, expert and router parameters are jointly updated.

Conversion check (maximum absolute difference on held-out inputs): **9.5367431640625e-07**.

## Measured results

| Model stage | Train MSE | Validation MSE |
|---|---:|---:|
| Linear, step 0 | 4.847176 | 4.460445 |
| Linear, step 120 | 0.133020 | 0.132206 |
| MoE, conversion step 0 | 0.133020 | 0.132206 |
| MoE, continuation step 40 | 0.029934 | 0.027532 |
| MoE, continuation step 120 | 0.026967 | 0.025696 |
| MoE, continuation step 240 | **0.025791** | **0.024812** |

These numbers were recorded in a local execution using PyTorch 2.10.0+cpu. The exact per-step checkpoints are included in [training_logs.csv](training_logs.csv). The MoE validation MSE decreased by approximately **81.2%** from the converted baseline.

## What I learned
1. Model **conversion** should be verified independently from subsequent MoE optimization. Duplicating trained weights gives function preservation at step zero.
2. **Identical experts + a uniform router** can prevent the experts from specializing. An input-dependent, nonuniform router lets their gradients diverge.
3. The higher-capacity MoE learned nonlinear structure the single linear model could not fit as well.
4. This is a **dense soft-gated MoE**, not sparse top-k token routing: all three experts execute. This assignment demonstrates conversion and continued loss reduction, **not** sparse inference speedups.
5. Training and held-out validation curves both improved, so this run provides stronger evidence than training loss alone. This does not establish generalization beyond the synthetic distribution.

## Files
- [train.py](train.py): self-contained training and weight conversion code
- [training_logs.csv](training_logs.csv): real captured train/validation MSE by stage
- [session14_linear_to_moe.ipynb](session14_linear_to_moe.ipynb): notebook with runnable end-to-end experiment and recorded output

## Reproduce
```bash
pip install torch
python train.py
```

No external dataset is needed. Different devices or library versions may change the final decimals.

## Assignment checklist
- [x] Train a linear model
- [x] Convert trained linear model weights into a multi-expert architecture
- [x] Verify prediction equivalence at conversion
- [x] Continue training and demonstrate falling loss
- [x] Save training logs in the repository
- [x] Include executable code and notebook
