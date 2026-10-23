# Session 11 — Adam and Learning-Rate Experiments

**Author:** Hari | ERA V5

This is a reproducible, small CPU-based **synthetic regression** assignment, not an LLM performance benchmark. Tested with PyTorch 2.10.0+cpu, seed 2026. Run `pip install torch numpy matplotlib`, then `python experiment.py`. The script generates plots, CSV evidence and summary JSON.

## Adam by hand
With initial scalar weight 1.0, gradients `[0.2,-0.1,0.3,0.05,-0.2]`, LR 0.01, β1 0.9, β2 0.999 and eps 1e-8:

```
m_t = 0.9*m_(t-1) + 0.1*g_t
v_t = 0.999*v_(t-1) + 0.001*g_t*g_t
mhat_t = m_t/(1-0.9**t)
vhat_t = v_t/(1-0.999**t)
w_t = w_(t-1) - 0.01*mhat_t/(sqrt(vhat_t)+1e-8)
```

The five manual updates agree with PyTorch Adam in float64 to a **maximum absolute error of 1.11e-16**. Output: `results/adam_hand_vs_torch.csv`.

## Bias correction
With a constant gradient 0.1, the script plots the first 20 corrected and uncorrected step magnitudes. With 'stops mattering' explicitly defined as relative *per-step update* error staying below a threshold thereafter, the difference reaches **<10% at step 1,751**, **<5% at step 2,375**, and **<1% at step 3,925**. These counts describe this particular constant-gradient example, not a universal Adam rule. Previously accumulated weight differences need not vanish.

## Update-to-weight ratios and warmup
For every trainable tensor, I log the actual AdamW ratio `||parameter_after - parameter_before|| / (||parameter_before||+1e-12)`, with weight decay included. The linear 30-step warmup multiplier stops increasing **at the end of step 30**; from step 31 onward the LR follows each scheduler. The **ratio can keep changing** after warmup due to changing gradients, moments and weights. Logs: `results/schedules_layer_ratios.csv`.

## Cosine vs WSD — both planned for 300, interrupted at 200
Same initial weights, data, warmup (30 steps), AdamW and weight decay (0.01), with a **separate LR sweep for each schedule** on the same candidate grid `{0.001,0.003,0.01}`. The planned cosine schedule decays after warmup to 0.1× by step 300; WSD holds a stable LR until step 240 and decays after that. Stop both at update 200.

| Schedule | Best base LR on tested grid | Training MSE at step 200 |
|---|---:|---:|
| Cosine | 0.01 | 0.0000925708 |
| WSD | 0.01 | **0.00000812094** |

**I would retain WSD at this interruption point** for this experiment (~11.4× lower training MSE). This is not evidence that WSD is universally better, nor proof of validation or generalization gains. Both LR choices were selected by training rather than held-out loss. The WSD run has not begun its decay by step 200. Logs: `results/schedule_lr_tuning.csv`.

## LR width sweep
An independent constant-LR AdamW sweep was performed for hidden widths 256, 512 and 1024; 60 full-batch steps each and LR grid `{.0001,.0003,.001,.003,.01,.03,.1}`.

| Width | Grid-minimizing LR | Training MSE |
|---|---:|---:|
| 256 | 0.03 | 0.00490342 |
| 512 | 0.03 | 0.00743844 |
| 1024 | 0.01 | 0.02189633 |

At width **4096 I would start by testing 0.01**, confidence **low**. This width was not measured. Next sweep `{0.003,0.01,0.03}` and use held-out loss and repeated seeds before accepting an optimum. Logs: `results/lr_sweep.csv`.

## Limitations
One fixed seed, full-batch synthetic MSE regression, no validation set, CPU, coarse LR grid, no 4096 run. Do not generalize to transformer pretraining without experiments. The script generates numerical evidence and three plots under `results/`; rerun to inspect and regenerate everything.
