# Assignment 5 — V5 Mixture and Curriculum Plan

## Executive decision

This plan treats every mixture number as a **testable hypothesis**, not as a fixed truth.

The total V5 training budget used here is **2.00T tokens**, split cleanly into:

- **1.96T tokens (98%)** for the bulk curriculum
- **40B tokens (2%)** protected for the final anneal / cooldown

The bulk curriculum is allocated across seven capability lanes:

| Capability lane | Bulk share | Tokens @ 1.96T | Primary benchmark / signal |
|---|---:|---:|---|
| General web / knowledge | **37%** | **725.2B** | MMLU / held-out perplexity |
| Code | **24%** | **470.4B** | LiveCodeBench / code pass@1 |
| Indic | **14%** | **274.4B** | MILU / Indic evaluation |
| STEM / mathematics | **12%** | **235.2B** | GPQA / math exact-match |
| Reasoning traces | **6%** | **117.6B** | math/reasoning exact-match |
| Long-context | **5%** | **98.0B** | RULER / long-context retrieval |
| Agentic / tool-use | **2%** | **39.2B** | BFCL / tau-bench / SWE-style execution |
| **Bulk total** | **100%** | **1.96T** | |

Then:

**1.96T bulk + 40B protected anneal = 2.00T total training budget.**

The mixture deliberately gives abundant web and code data most of the budget while protecting scarce Indic, reasoning and agentic capabilities from being optimized away.

---

## 1. Inventory supply check

The percentages above are constrained by the actual Session 5 inventory, not chosen independently of supply.

| Lane | Inventory supply | Bulk demand | Supply conclusion |
|---|---:|---:|---|
| General web + STEM pool | **~4.84T** | web 725.2B + STEM 235.2B | abundant |
| Code | **~1.103T** | 470.4B | abundant |
| Indic | **~275.9B** | 274.4B | essentially at supply limit |
| Reasoning + math traces | **~85.1B** | 117.6B | **~32.5B deficit** |
| Long-context | **~100B** | 98.0B | fits, but leaves almost no cleaning-loss margin |
| Agentic / tool-use | **~0.6–0.63B** | 39.2B | **severely starved** |

### Inventory evidence used

The Session 5 inventory numbers are represented in public cohort artifacts as:

- General web + STEM: **4,837B**
- Code: **1,103B**
- Indic: **275.9B**
- Long-context: **100B**
- Reasoning + math: **85.1B**
- Agentic: **~0.6B**

Cross-check:
- [Session 5 mined inventory numbers](https://github.com/millermuttu/ERA-V5/blob/main/Session5/reference/mined-numbers.md)
- [Session 5 inventory configuration cross-check](https://github.com/iabh1shekbasu/mixture-curriculum/blob/main/config/inventory.yaml)

These links are used as a reproducible cross-check of the course inventory. The final implementation must use the cohort's authoritative Session 5 inventory/manifest values if they differ.

### What this means

The **agentic lane is the binding constraint**.

A 2% agentic lane sounds small, but 39.2B requested tokens against only about 0.6B of real inventory means the target cannot honestly be reached from unique real trajectories. Most of this lane must therefore come from replayed or newly generated **executable and verified** trajectories.

Reasoning is the second constrained lane. Its 117.6B target exceeds the ~85.1B inventory, so it needs modest repetition and/or verified generation.

Long-context is almost exactly supply-limited: aggressive cleaning could push the usable pool below the 98B target.

---

## 2. Indic provenance split

A headline number such as "14% Indic" is not sufficient because verified native, unverified native, translated and synthetic tokens are not equivalent.

The **274.4B-token Indic bulk slot** is therefore split explicitly:

| Indic tier | Share of Indic lane | Tokens | Policy |
|---|---:|---:|---|
| **Verified native** | **35%** | **96.0B** | highest priority; repeat only within repetition cap |
| **Unverified native / crawl** | **15%** | **41.2B** | quality-selected + provenance retained |
| **Translated** | **15%** | **41.2B** | translation lineage retained; source-side dedup required |
| **Synthetic** | **35% maximum** | **96.0B max** | verifier-gated; ancestry tracked |
| **Total** | **100%** | **274.4B** | |

The **35% synthetic figure is a ceiling, not a target**. If additional verified native material survives cleaning, it replaces synthetic material first.

### Indic inventory sources

The inventory is anchored by:

- **Sangraha verified** — ~64B
- **Sangraha unverified** — ~24B
- **Sangraha synthetic** — ~162–163B
- **IndicCorpV2** — ~20.9B
- **BPCC** — ~3B
- **Samanantar** — ~2B

For translated expansion, BPCC/Samanantar and IndicTrans-style generation can supply aligned examples, but translated and synthetic descendants remain separately tagged.

Every generated/translated record must preserve:

- original source or seed ID
- source language
- target language/script
- generation/translation method
- teacher/model identity where applicable
- verifier status
- transformation lineage
- deduplication ancestry

A thousand descendants of one source document do **not** count as a thousand independent sources.

---

## 3. Capability lane → exact inventory mapping

### General web / knowledge — 725.2B

Inventory sources:

- **DCLM**
- **FineWeb-Edu**
- compatible high-quality web/reference slices from the Session 5 inventory

Purpose: broad knowledge, language modelling, discourse and factual breadth.

Target signals: MMLU-family performance and general held-out perplexity.

### Code — 470.4B

Inventory sources:

- **The Stack v2**
- **D3/V4 code**
- **CommitPack**
- repository code, tests, commits and documentation represented in the inventory

Purpose: code generation, repair, repository understanding and executable reasoning.

Target signals: LiveCodeBench, code pass@1 and later SWE-style evaluation.

### STEM / mathematics — 235.2B

Inventory sources include:

- **Proof-Pile**
- **D4-STEM**
- **peS2o**
- technical/scientific subsets from the common inventory

Purpose: technical knowledge and mathematical/scientific substrate.

Target signals: GPQA and math exact-match.

### Reasoning traces — 117.6B

Exact inventory sources:

- **AON/V4** — the dominant reasoning source
- **OpenThoughts**
- **OpenMath**
- **OpenR1**
- **Numina**

The inventory provides ~85.1B against a 117.6B target.

Required effective reuse/generation factor:

`117.6 / 85.1 ≈ 1.38×`

That is a modest shortfall compared with the agentic lane. Verified traces can therefore be selectively replayed, with generated traces admitted only when the final answer is checkable.

For math: use symbolic/numeric answer verification.

For code: compile/run tests.

For factual reasoning: use answer/reference verification where available.

### Long-context — 98.0B

Exact inventory buckets:

- **repo-packed long-context** — ~60B
- **book-length long-context** — ~40B

These are repacked/structured long sequences, not necessarily 100B of new unique semantic content.

Because supply is ~100B and the target is 98B, this lane has effectively **no cleaning-loss margin**. If usable supply falls below target, I reduce the lane rather than fabricate long sequences merely to hit a percentage.

Target signals: RULER / retrieval-under-long-context / needle-style and multi-document tasks.

### Agentic / tool-use — 39.2B

Inventory sources include:

- **SWE-Gym** — ~150M
- **SWE-smith** — ~120M
- **OpenHands trajectories** — ~90M
- **ToolBench** — ~80M
- **ToolACE** — ~60M
- **Glaive** — ~50M
- **Nexus** — ~30M
- **xLAM** — ~25M
- **Hermes** — ~22M

Total real inventory is only about **0.6B tokens**.

Therefore the 39.2B target cannot be reached by repetition alone without destroying diversity.

The expansion mechanism is **environment-grounded synthetic trajectory generation**:

1. sample a real tool/environment/task;
2. generate a multi-step trajectory;
3. validate schema and permissions;
4. replay the actions where possible;
5. verify the final environment/test state;
6. reject traces that only imitate tool syntax without succeeding.

Target signals: BFCL, tau-bench, SWE-style execution success and recovery behavior.

---

## 4. Protected Always-On floor

The selector is **not allowed** to reduce the following lanes below these batch-level floors:

| Protected lane | Hard floor |
|---|---:|
| Indic | **12%** |
| Reasoning | **4%** |
| Agentic / tools | **2%** |
| **Protected minimum** | **18%** |

These are floors, not desired averages.

- Indic target = 14%, but **never <12%**
- Reasoning target = 6%, but **never <4%**
- Agentic target = 2%, therefore **the entire lane is protected**

Why protect them?

Generic quality selection can systematically remove exactly the unusual structures we care about: low-resource scripts, long derivations, JSON/tool traces, failure/recovery steps and sparse multilingual examples.

The remaining 82% of any batch may be adaptively redistributed subject to the overall curriculum.

---

## 5. Protected anneal reserve

The anneal reserve is **40B tokens = 2% of the total 2T budget**.

It is excluded from the 1.96T bulk accounting and cannot be consumed by the normal selector.

Proposed anneal composition:

| Lane | Anneal share | Anneal tokens |
|---|---:|---:|
| High-quality general | 10% | 4B |
| Code | 20% | 8B |
| Verified Indic | 20% | 8B |
| STEM | 10% | 4B |
| Reasoning | 20% | 8B |
| Long-context | 10% | 4B |
| Agentic | 10% | 4B |
| **Total** | **100%** | **40B** |

The cooldown intentionally shifts away from bulk web and toward high-confidence, high-value examples.

No synthetic item is admitted merely because it is difficult. It must pass the lane-specific verifier.

---

## 6. Curriculum schedule

The schedule changes both **what capability is sampled** and **how difficult/long the example is**.

### Stage A — foundation

Favor high-quality web, native Indic and straightforward code.

Use mostly easy/medium examples while all protected floors remain active.

### Stage B — code + STEM

Increase executable code, math and technical documents.

Introduce medium-depth reasoning.

### Stage C — reasoning shaping

Increase verified reasoning traces while keeping the Indic and agentic floors active.

Prefer difficult examples only where verifier confidence remains high.

### Stage D — long-context + agentic

Increase repo-packed/book-length sequences and longer executable trajectories.

Teach recovery, state tracking, error handling and multi-step completion rather than only single function calls.

### Stage E — 40B anneal

Consume only the protected premium reserve at reduced learning rate.

All stage transitions should be blended rather than instantaneous to avoid a sharp distribution discontinuity.

---

## 7. Difficulty bands

Difficulty is metadata used by the curriculum sampler.

| Band | Definition | Concrete example |
|---|---|---|
| **D0 — foundational** | direct language/recognition | Translate a simple Telugu sentence into English |
| **D1 — elementary** | one operation or one fact | 40 seats × 3 buses |
| **D2 — intermediate** | several dependent operations | Solve a quadratic and verify both roots |
| **D3 — advanced** | undergraduate multi-concept reasoning | Derive the gradient of softmax cross-entropy |
| **D4 — expert** | interacting systems / substantial analysis | Compare two attention implementations at 128K context for memory and compute |
| **D5 — research/agentic** | open-ended diagnosis + verification | Inspect a failing repo, identify the bug, patch it, execute tests and verify the final state |

Early stages emphasize D0–D2. D3–D5 increase later without eliminating the easier distribution.

---

## 8. Reasoning-length bands

Difficulty and reasoning length are separate axes.

The curriculum uses four explicit reasoning-depth bands:

| Band | Working trace budget | Concrete example |
|---|---:|---|
| **LOW** | **≤128 tokens** | Apply inclusion-exclusion directly to count integers divisible by 3 or 5 |
| **MEDIUM** | **129–512** | Compute multiples of 3, 5 and 15 explicitly, combine them and perform a boundary check |
| **HIGH** | **513–2,048** | For a harder algebra/algorithm problem, derive the solution, test alternatives and independently verify the result |
| **ULTRA** | **2,049–4,096+** | Repository-level debugging: inspect multiple files, form hypotheses, patch code, run tests, recover from failures and verify final state |

The key rule is:

> **Longer reasoning is not automatically better.**

The trivial inclusion-exclusion problem should not be inflated into an ULTRA trace. ULTRA is reserved for tasks whose complexity genuinely requires long-horizon reasoning.

This prevents the curriculum from rewarding verbosity instead of correctness.

---

## 9. Proxy experiments — numbers remain hypotheses until tested

No full-scale mixture decision is trusted before cheap matched-control experiments.

### Hypothesis

> Protecting Indic, reasoning and agentic lanes should improve those capabilities enough to justify their opportunity cost without materially degrading general language or coding.

### Arm A — supply-proportional baseline

Sample approximately according to usable post-cleaning supply.

No protected floor.

### Arm B — proposed mixture

Use the proposed:

`37 / 24 / 14 / 12 / 6 / 5 / 2`

bulk mixture plus the Always-On floors.

For a fair comparison, hold constant:

- tokenizer
- architecture
- optimizer
- learning-rate schedule
- sequence length policy
- training-token count
- evaluation harness
- checkpoint cadence

### 1B proxy

Train matched 1B-parameter models.

Measure:

- general held-out perplexity
- MILU / Indic held-out metrics
- LiveCodeBench-style code pass@1
- GPQA/math/reasoning exact-match
- BFCL-mini/tool execution success
- RULER/long-context retrieval

**Initial confirmation thresholds:**

- Indic held-out PPL improves by **≥5% relative**
- reasoning accuracy improves by **≥5% relative**
- agentic/tool success improves by **≥8% relative**
- code pass@1 regression is **<2 percentage points**
- general held-out PPL degradation is **<3%**
- long-context retrieval does not regress

Failure of these gates triggers a mixture adjustment before the next proxy.

### 3B proxy

Repeat the experiment at 3B after changing only the decisions justified by the 1B result.

The exact effect size need not be identical, but the important capability gains must reproduce in the **same direction**.

Promotion to full scale requires:

1. Indic, reasoning and agentic gains at both scales;
2. no material general/code regression;
3. no instability at curriculum transitions;
4. measurable return per added/repeated/synthetic token;
5. no evidence that synthetic expansion improves formatting without improving task success.

A gain that appears at 1B and disappears or reverses at 3B is treated as scale-specific noise, not validation.

---

## 10. Falsification rules

I will change the proposed numbers if any of the following occurs:

- Indic gains saturate while repetition rises;
- translated/synthetic Indic underperforms native repetition;
- agentic synthesis improves function-call formatting but not executable success;
- reasoning data increases response length without increasing correctness;
- long-context data hurts short-context performance without improving long-context retrieval;
- a protected lane's gain is smaller than the regression it creates elsewhere.

This is what makes the mixture a falsifiable training hypothesis rather than a presentation target.

---

## 11. Data-gating and cleaning evidence

The mixture is valid only if cleaned, documented tokens actually exist behind it.

The data gate is therefore separate from the mixture proposal.

For every cleaned shard, the manifest must retain at least:

- source
- licence
- cleaning-script hash
- content hash
- token count
- language(s)

### Current rule

**Do not claim the data gate is complete unless the cumulative cleaned-token threshold and manifests can be linked from this repository.**

When that evidence is available, it should be linked here under:

`Session-5/Assignment-5/evidence/`

or to the authoritative earlier-session data-cleaning artifact.

### Cleaning priority implied by this mixture

Cleaning effort should follow **scarcity**, not raw corpus size:

1. **Agentic** — replayable/executable trajectories, schema validation, final-state checks
2. **Reasoning** — answer verification, difficulty tags, reasoning-length tags
3. **Verified/native Indic** — provenance, script-aware normalization, deduplication
4. **Long-context** — document completeness, packing quality, cross-document deduplication
5. **STEM** — equations, citations and answer validation
6. **Code/web** — continue cleaning, but do not let their abundance consume all cleaning capacity

The next cleaned tokens should preferentially target lanes where usable supply falls furthest below the planned mixture.

---

## 12. Final commitments

| Decision | Commitment |
|---|---|
| Total training budget | **2.00T** |
| Bulk curriculum | **1.96T / 98%** |
| Protected anneal | **40B / 2%** |
| Bulk capability mix | **37 / 24 / 14 / 12 / 6 / 5 / 2** |
| Indic provenance split | **35 / 15 / 15 / ≤35** verified / unverified / translated / synthetic |
| Always-On floor | **Indic 12% + reasoning 4% + agentic 2%** |
| Most supply-starved lane | **Agentic** |
| Second supply constraint | **Reasoning** |
| Long-context policy | **cap at usable real/repacked supply; do not fabricate quota** |
| Proxy requirement | **matched 1B then 3B experiments** |
| Promotion rule | gains reproduce at both scales without material core regression |
| Data-gating rule | no "complete" claim without cleaned-token manifests/evidence |

## Bottom line

The numbers in this document are not trusted because they add to 100%.

They are trusted only if:

1. the cleaned inventory can fund them,
2. repetition/synthetic generation is declared wherever supply is insufficient,
3. protected floors prevent scarce capabilities from disappearing,
4. the anneal reserve remains genuinely held back, and
5. the 1B and 3B proxy experiments show that the proposed token allocation buys the intended capability.
