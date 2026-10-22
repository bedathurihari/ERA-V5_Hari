# Attention Atlas — ERA V5 Session 8 Assignment

Interactive, source-linked chronological explorer covering 22 attention and positional mechanisms from 2017 to 2026.

## Running it
Open `index.html` in any browser. No dependencies, no build step. For Netlify/Vercel, publish this folder as a static site with no build command. GitHub Pages can publish the folder or use a workflow.

## Chronology / dating method
Use the **first public arXiv v1 date** where available, rather than the conference publication year. For organization releases use the official release notice. All entries include a primary source link in the interactive detail panel.

Important edge cases:
- Scaled dot-product attention, sinusoidal encoding and learned absolute positions share **12 June 2017** because the original Transformer paper discusses each. Learned absolute positions also have earlier antecedents; this entry specifically refers to the Transformer formulation, not the invention of learned position vectors.
- Sparse Transformer: OpenAI's **23 April 2019** publication.
- MQA: arXiv **6 November 2019**; GQA: arXiv **22 May 2023**.
- NTK-aware scaling: **June 2023** community release, not a refereed original paper. Day-level dates for informal techniques are less robust than arXiv timestamps; verify the linked thread before citing a precise timestamp.
- DeepSeek Sparse Attention: official V3.2-Exp announcement **29 September 2025**. Do not conflate with MLA in DeepSeek-V2 (**7 May 2024**) or KV compression.
- DroPE: original preprint **13 December 2025**, despite **12 January 2026** Sakana AI announcement.
- LazyAttention: 2026 ICML proceeding; the displayed July 2026 date is a **conference-period placeholder**, not verified first arXiv availability, and should not be interpreted as day-accurate. Further verify original preprint before claiming first appearance.

## Primary source index
- 2017-06-12 — **Scaled dot-product attention** — https://arxiv.org/abs/1706.03762
- 2017-06-12 — **Sinusoidal positions** — https://arxiv.org/abs/1706.03762
- 2017-06-12 — **Learned absolute positions** — https://arxiv.org/abs/1706.03762
- 2019-04-23 — **Sparse Transformer** — https://openai.com/index/sparse-transformer/
- 2019-06-26 — **Transformer-XL recurrence** — https://arxiv.org/abs/1901.02860
- 2019-11-06 — **Multi-query attention (MQA)** — https://arxiv.org/abs/1911.02150
- 2020-04-10 — **Longformer sliding window** — https://arxiv.org/abs/2004.05150
- 2020-06-29 — **Linear attention** — https://arxiv.org/abs/2006.16236
- 2020-09-30 — **Performer (FAVOR+)** — https://arxiv.org/abs/2009.14794
- 2021-04-20 — **RoPE** — https://arxiv.org/abs/2104.09864
- 2021-08-27 — **ALiBi** — https://arxiv.org/abs/2108.12409
- 2022-05-27 — **FlashAttention** — https://arxiv.org/abs/2205.14135
- 2023-05-22 — **Grouped-query attention (GQA)** — https://arxiv.org/abs/2305.13245
- 2023-06-29 — **NTK-aware RoPE scaling** — https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
- 2023-08-31 — **YaRN** — https://arxiv.org/abs/2309.00071
- 2023-09-29 — **Attention sinks / StreamingLLM** — https://arxiv.org/abs/2309.17453
- 2024-05-07 — **Multi-head latent attention (MLA)** — https://arxiv.org/abs/2405.04434
- 2024-06-10 — **Delta rule / DeltaNet** — https://arxiv.org/abs/2406.06484
- 2024-12-09 — **Gated DeltaNet** — https://arxiv.org/abs/2412.06464
- 2025-09-29 — **DeepSeek Sparse Attention (DSA)** — https://api-docs.deepseek.com/news/news250929/
- 2025-12-13 — **DroPE** — https://arxiv.org/abs/2512.12167
- 2026-07-01 — **LazyAttention** — https://proceedings.mlr.press/v306/xia26f.html

## What the date order reveals
The field alternates between reducing pairwise compute, reducing decode KV traffic, preserving memory, and extending usable length. These are not interchangeable optimizations. FlashAttention improves the exact softmax kernel's IO without changing quadratic arithmetic; MQA/GQA/MLA optimize KV capacity; window/sparse strategies avoid some score pairs; recurrent delta memory changes representation and retrieval guarantees; RoPE/ALiBi/YaRN/DroPE modify positional generalization. Selection depends on length, model quality goals, deployment hardware, and retrieval workload.

## Limitations
Visual matrix is an illustrative connectivity mask, not empirical learned attention. Claims are qualitative, not benchmark results. Some technique names denote kernels or position methods rather than new softmax rules. Dates must be sourced with the original public artifact, not copied from later conference dates.

Prepared as an ERA V5 Session 8 learning artifact.
