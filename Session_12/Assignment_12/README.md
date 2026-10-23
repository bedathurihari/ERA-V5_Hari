# Session 12 — Assignment 12: ZeRO with 32 Virtual GPUs

**Author: Hari | ERA V5**

## Objective
I created 32 **logical** GPU ranks using CPU-backed PyTorch tensors and a small multilayer perceptron. I compared ordinary replicated data parallelism (ZeRO-0) against ZeRO-1, ZeRO-2 and ZeRO-3. These are virtual workers, **not 32 physical GPUs**.

## Run the notebook
Open [Session_12_Zero_32_Virtual_GPUs.ipynb](Session_12_Zero_32_Virtual_GPUs.ipynb) in Jupyter or Colab. Install `torch numpy matplotlib`; run all cells. It needs no real GPU and no distributed-process launcher. An executed version with figures is also included in my assignment materials.

## What I implemented and verified
1. A 16→64→4 PyTorch MLP with 1,348 learnable parameters.
2. 32 separate micro-batch losses and gradients on CPU, combined into a global mean gradient.
3. Nonoverlapping parameter/gradient slices that cover all 1,348 elements, with no omissions or duplicates.
4. Sharded Adam calculations and comparison against a reference full-parameter Adam update. All 4 modes pass.
5. Analytic per-rank mixed-precision state-memory budgets and idealized collective payload estimates, with plots.

## My understanding of the four modes

| Mode | Parameters | Gradients | Optimizer state + FP32 master | Why it matters |
|---|---|---|---|---|
| ZeRO-0 | Replicated | Replicated | Replicated | Data-parallel baseline |
| ZeRO-1 | Replicated | Replicated | Sharded | Removes repeated Adam moments and master weights |
| ZeRO-2 | Replicated | Sharded | Sharded | Also avoids persistent full gradient copies |
| ZeRO-3 | Sharded | Sharded | Sharded | Avoids full persistent parameter copies but needs gathers around compute |

The important idea I learned is that **ZeRO is primarily a model-state partitioning technique**, rather than a different model or automatically a 32× speedup. Each logical rank still needs forward/backward work for its micro-batch. Stages 1–3 partition local optimizer updates; Stage 3 adds parameter materialization and communication.

## Actual notebook validation (32 ranks, 1,348 parameters)

| Stage | Ideal persistent model-state KiB / rank | Fraction of baseline |
|---|---:|---:|
| ZeRO-0 | 21.06 | 1.000 |
| ZeRO-1 | 5.76 | 0.273 |
| ZeRO-2 | 3.21 | 0.152 |
| ZeRO-3 | 0.66 | 0.031 |

All 4 simulated Adam parameter updates matched the reference (maximum deviation reported as 0 in the test run). The shard sizes are 42 or 43 elements, and concatenation reconstructs the full weights and averaged gradients.

### At 1 billion parameters (analytical extrapolation)
With 2-byte model weights + 2-byte gradients + 4-byte FP32 master weights + 8-byte Adam moments, full replicated training states cost 16 bytes/parameter. Per-rank estimates with 32 ranks are: **14.90 GiB (baseline), 4.08 GiB (Stage 1), 2.27 GiB (Stage 2), 0.47 GiB (Stage 3)**.

## Computation and communication
The memory improvement is not free. A ring all-reduce's ideal traffic is `2*(R-1)/R` times tensor size per rank; reduce-scatter and all-gather each move approximately `(R-1)/R` tensor size per rank. Stage 1 shards the optimizer arithmetic, Stage 2 uses reduce-scatter for gradient shards, and Stage 3 also gathers parameter shards for computation.

My notebook charts estimated communication bytes and the share of Adam tensor elements locally updated per rank. It does **not** measure collective latency, GPU utilization or wall-clock speedup. The communication model is illustrative; real ZeRO implementations have different schedules and overlap strategies.

## Important limitations
- Logical CPU ranks are simulated sequentially, not separate CUDA devices or processes.
- The numeric Adam correctness test uses FP32 CPU tensors; the memory budget uses *hypothetical* FP16 training weights/gradients with FP32 optimizer master weights and moments.
- Estimates exclude activations, temporary buffers, fragmented allocations, communication workspaces, mixed-precision scaler buffers, and prefetch.
- A complete DeepSpeed ZeRO-3 system uses layer-aware parameter gathers and distributed autograd, whereas this demonstration checks partition and optimizer-update math.

**Conclusion:** I can explain the tradeoff: ZeRO-1 first saves optimizer storage, ZeRO-2 reduces gradient replication, and ZeRO-3 minimizes all persistent model states, but typically requires substantially more parameter communication and careful scheduling.
