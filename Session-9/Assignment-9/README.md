# Session 9 — Assignment 9: Loss Harness and Two Heads

**Hari | ERA V5**

[Open the Colab-ready notebook](Session_9_Loss_Harness.ipynb) · [View training log](training_log.txt)

I implemented a small PyTorch character-level causal loss harness, checked token shifts using actual strings, masked padding and a packed-document boundary, checked untrained perplexity, compared tied and untied head parameters, and benchmarked ordinary versus manually chunked cross-entropy. The notebook is self-contained and runs top-to-bottom with PyTorch.

## Part 1 — seven observed checkpoints

| Requirement | Measured result |
|---|---|
| 1. Tensor shapes | Tokens **[4,13]**, hidden **[4,13,32]**, logits **[4,13,12]**, shifted logits **[4,12,12]**, flattened logits **[48,12]**, targets **[48]**. Dimensions are batch, time, hidden width, vocabulary. |
| 2. Literal shift check | `'a' → ' '`, `' ' → 'c'`, `'c' → 'a'`, `'a' → 't'`. Full printed alignment in notebook. |
| 3. Padding | **48 → 36** contributing targets; CE **2.463489 → 2.438704** after mask. |
| 4. Packed-document boundary | **19 → 18** terms after excluding `'.' → 'a'`; CE **2.460906 → 2.455686**. Removing an invalid supervision term changes the mean; masking does *not* always decrease loss. |
| 5. Initial perplexity | **11.458** vs vocabulary **12**, PPL/V **0.955**; expected near vocabulary size for random logits. |
| 6. Tied versus untied | **1,824** tied vs **2,208** untied parameters in full model; saves **384 = 12 × 32** parameters by tying the t+1 head to embedding. |
| 7. Peak memory | CPU process peak RSS **321.01 MiB ordinary** vs **290.82 MiB chunked**; **1.104×** ratio; computed losses differ by <0.000002. |

The CPU memory numbers are a **forward-only** subprocess benchmark using identical fixed input shapes and random seeds (1024 target rows, hidden width 64, vocab 8192, chunk 64). RSS includes the runtime and model allocations; it is *not* CUDA memory or backward-pass peak. Results may differ in Colab.

## Part 2 — second head predicts t+2

| Stage | t+1 CE | t+2 CE | Sum |
|---|---:|---:|---:|
| Initialization | 2.494630 | 2.487981 | 4.982611 |
| After 150 steps | **0.231568** | **0.260515** | **0.492083** |

Both losses drop sharply. The second head remains slightly higher, because predicting two tokens ahead without observing the intervening token is less directly constrained in this setup. This tiny repetitive corpus can be memorized, so the observation is **not** a generalization claim.

## Correctness and reproducibility

- First head: `logits[:, :-1]` against `tokens[:, 1:]`; second head: `head2[:, :-2]` against `tokens[:, 2:]`. Both use `ignore_index=PAD`.
- Padding and packed-document transition are different masks: one removes PAD targets, the other removes one artificial between-document target. A **real attention model additionally needs block-diagonal causal attention masking** for packed training. This toy backbone is causal and only sees current/previous token.
- Chunked CE calculates logits for small row slices, accumulates summed cross-entropy and divides by the global target count; it is numerically checked against full CE.
- Evidence: notebook source plus `training_log.txt` from a successful executed CPU run. The log's memory values are environment dependent.

**To reproduce:** Open `Session_9_Loss_Harness.ipynb` in Google Colab and choose **Runtime → Run all**.
