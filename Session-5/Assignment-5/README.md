# Assignment 5 — V5 Mixture and Curriculum Plan

## 1. Design hypothesis

The V5 mixture should not follow raw corpus size. It should allocate enough tokens to each capability to move its target benchmark while respecting the real inventory behind that lane.

I assume a **2T-token main pretraining budget**.

| Capability lane | Share | Tokens @ 2T | Primary evaluation |
|---|---:|---:|---|
| General web / knowledge | **37%** | 740B | MMLU / broad held-out perplexity |
| Code | **24%** | 480B | LiveCodeBench / coding pass@1 |
| Indic | **14%** | 280B | MILU / IndicGenBench |
| STEM / mathematics | **12%** | 240B | GPQA / math evaluation |
| Reasoning traces | **6%** | 120B | reasoning exact-match / GPQA / math |
| Long-context | **5%** | 100B | long-eval / retrieval accuracy |
| Agentic / tool-use | **2%** | 40B | BFCL / SWE-style tool execution |
| **Total** | **100%** | **2T** | |

This mixture deliberately gives the abundant web and code lanes most of the budget while protecting scarce capabilities from disappearing.

The central constraint is whether the inventory can actually fund each lane.

---

## 2. Supply check

| Lane | Approx. inventory supply | Demand | Decision |
|---|---:|---:|---|
| Web | ≥4.5T | 740B | Fully funded with unique data |
| Code | ~1.1T | 480B | Fully funded with unique data |
| Indic | ~276B total | 280B | Raw volume is close, but provenance constraints require selective repetition |
| STEM / math | ~250B | 240B | Nearly saturates high-quality supply |
| Reasoning | ~85B | 120B | **Starved:** ~35B gap; repetition or verified generation required |
| Long-context | ~100B | 100B | At supply limit; aggressive filtering could create a deficit |
| Agentic | ~0.63B real trajectories | 40B | **Severely starved:** cannot be achieved without generated/replayed trajectories |

The **agentic lane is the binding constraint**. A 2% allocation already requires far more sampled tokens than currently available as real tool trajectories.

Therefore I do not allocate 5–10% agentic pretraining. That would mostly describe synthetic-data production rather than the current inventory.

---

## 3. Indic provenance budget

A single "14% Indic" number hides the quality question, so the 280B Indic allocation is split into four explicit tiers.

| Indic tier | Share of Indic | Share of total run | Tokens |
|---|---:|---:|---:|
| **Verified native** | **35%** | 4.9% | 98B |
| **Unverified native/crawl** | **15%** | 2.1% | 42B |
| **Translated** | **15%** | 2.1% | 42B |
| **Synthetic** | **35% max** | 4.9% | 98B |
| **Total Indic** | **100%** | **14%** | **280B** |

The synthetic tier is a **ceiling, not a target**. Unused synthetic budget is first replaced with additional verified native data, then high-quality unverified native data.

All generated Indic records retain:

- source/teacher lineage,
- target language and script,
- generation method,
- verifier status,
- deduplication ancestry.

Synthetic descendants of the same seed are not counted as independent diversity.

---

## 4. Inventory → capability mapping

### General web
Primary inventory:
- FineWeb / FineWeb-Edu-class data
- DCLM-style web data

Purpose: world knowledge, language modelling, discourse, factual breadth.

### Code
Primary inventory:
- The Stack v2
- D3 Code
- CommitPack
- repository documentation, commits and tests from the inventory

Purpose: code generation, repair and repository understanding.

### Indic
Primary inventory:
- Sangraha
- verified native Indic sources
- cleaned native crawl
- translated tier
- lineage-tracked synthetic tier

Purpose: MILU/IndicGenBench capability, Indian-language generation and code-mixed competence.

### STEM / mathematics
Primary inventory:
- STEM subsets of the inventory
- mathematics problem/solution corpora
- scientific and technical text

Purpose: factual technical knowledge and the substrate required by reasoning training.

### Reasoning
Primary inventory:
- verified math traces
- code reasoning traces
- general reasoning traces
- generated traces retained only after answer verification

Reasoning supply is smaller than its 120B allocation, so high-quality traces may be replayed approximately 1.4× or the deficit must be produced using verifier-backed generation.

### Long-context
Primary inventory:
- long books/documents
- whole manuals
- multi-document packs
- long repositories

The 100B target approximately exhausts currently identified supply. Cleaning losses therefore directly reduce this lane and must be monitored.

### Agentic / tool-use
Primary inventory:
- ToolBench
- Glaive function-calling data
- ToolACE
- xLAM/APIGen-class data
- NexusRaven
- Hermes function-calling data
- SWE-style/OpenHands trajectories

Real supply is far below the 40B target.

The gap is filled only with **executable synthetic trajectories**. A generated trajectory is accepted when the tool schema is valid, actions can be replayed, and final environment state satisfies the task.

Text that merely resembles a tool call does not qualify.

---

## 5. Protected always-on floor

Adaptive selection must not optimize away capabilities that look statistically unusual.

| Protected lane | Minimum batch share |
|---|---:|
| Indic | **12%** |
| Reasoning | **4%** |
| Agentic | **2%** |
| **Total protected minimum** | **18%** |

These are floors, not target weights.

- Main Indic target = 14%, but the selector may never reduce it below 12%.
- Agentic remains at its full 2% because tool traces can resemble templated/log data and may otherwise be rejected by generic quality selectors.
- Reasoning receives a 4% floor so easy web/code loss does not eliminate difficult traces.

The remaining sampling mass may be dynamically reweighted.

---

## 6. Anneal reserve

I hold back **2% of the total budget = 40B tokens** for the final low-learning-rate anneal.

These tokens are protected from early consumption.

| Lane | Anneal share |
|---|---:|
| General high-quality web | 10% |
| Code | 20% |
| Verified Indic | 20% |
| STEM | 10% |
| Reasoning | 20% |
| Long-context | 10% |
| Agentic | 10% |

The anneal deliberately differs from the main mixture.

Abundant general web falls sharply, while scarce high-quality reasoning, verified Indic and agentic examples become more concentrated.

No synthetic item enters the anneal simply because it is difficult; it must also pass its domain verifier.

---

## 7. Curriculum

The curriculum changes both capability mixture and difficulty over time.

### Stage 1 — language and representation foundation
Emphasize general web, verified Indic and straightforward code.

Long-context and difficult reasoning remain low but non-zero.

### Stage 2 — code + STEM
Increase executable code, mathematics and technical documents.

Introduce medium-complexity reasoning.

### Stage 3 — reasoning
Raise reasoning-trace exposure while preserving web/code/Indic floors.

Hard examples increasingly replace easy duplicated examples.

### Stage 4 — long-context + agentic
Increase long sequences after the model has learned useful local representations.

Introduce longer tool trajectories, recovery behavior and final-state verification.

### Stage 5 — anneal
Use the reserved 40B premium tokens at reduced learning rate.

Transitions are blended rather than instantaneous so a curriculum boundary does not become an optimization discontinuity.

---

## 8. Difficulty bands

Difficulty is attached as metadata and used by the sampler.

| Band | Interpretation | Concrete example |
|---|---|---|
| **D0 — foundational** | direct recognition/language | Translate a simple Telugu sentence into English |
| **D1 — elementary** | one operation or fact | A bus has 40 seats. How many seats are in 3 buses? |
| **D2 — intermediate** | several connected steps | Solve a quadratic equation and verify both roots |
| **D3 — advanced** | undergraduate reasoning | Derive the gradient of softmax cross-entropy |
| **D4 — expert** | several interacting concepts | Compare two attention implementations for memory and compute at 128K context |
| **D5 — research / agentic** | open-ended diagnosis and verification | Diagnose a failing repository, modify the correct files, run tests and justify the fix |

Early training is dominated by D0–D2.

D3–D5 rise gradually instead of appearing at one hard boundary.

---

## 9. Reasoning-length bands

Difficulty and reasoning length are independent.

| Band | Trace budget | Example behaviour |
|---|---:|---|
| **R0 Direct** | ≤128 tokens | answer or one obvious calculation |
| **R1 Short** | 129–384 | several explicit operations |
| **R2 Medium** | 385–1,024 | multi-stage derivation plus verification |
| **R3 Long** | 1,025–4,096 | complex proof/debugging/planning |
| **R4 Extended** | >4,096 | long-horizon agent/research task with intermediate validation |

Example problem:

> Find how many integers from 1 to 1000 are divisible by 3 or 5.

An R0 response can apply inclusion-exclusion directly.

An R1 version can show the three counts.

An R2 version can derive the formula and verify boundaries.

There is no justification for generating an R4 trace for this problem.

Reasoning length is therefore sampled conditional on difficulty rather than rewarded monotonically.

---

## 10. Proxy experiments

Every percentage above is a hypothesis.

I will not promote this mixture to the full V5 run until it survives both a **1B-parameter** and a **3B-parameter** proxy.

### Hypothesis

> Protecting Indic, reasoning and agentic lanes improves those capabilities enough to justify their opportunity cost without materially degrading general language or coding.

### Arm A — supply-proportional baseline
Sample mainly according to available corpus volume.

No protected floor.

### Arm B — proposed V5 mixture
Use:

`37 / 24 / 14 / 12 / 6 / 5 / 2`

with the 12% Indic, 4% reasoning and 2% agentic floors.

Architecture, tokenizer, optimizer, sequence length and training-token budget stay identical between arms.

### 1B proxy metrics
Measure:
- Indic held-out perplexity
- MILU / Indic evaluation
- coding pass@1
- reasoning exact-match
- BFCL-mini / tool-call success
- long-context retrieval accuracy
- general held-out perplexity

The proposed mixture passes the first gate if:
- Indic held-out PPL improves by **≥5% relative**
- agentic/tool success improves by **≥8% relative**
- reasoning accuracy improves by **≥5% relative**
- code pass@1 falls by **<2 percentage points**
- general validation PPL degrades by **<3%**

Otherwise the corresponding lane share is changed before the 3B run.

### 3B proxy
Repeat the comparison after incorporating only changes justified by the 1B result.

The exact numerical improvement need not match the 1B proxy, but the **direction must reproduce**.

A mixture is accepted only when:
1. protected capabilities improve at both scales,
2. neither code nor general language shows a material regression,
3. curriculum transitions remain optimization-stable,
4. observed gain per sampled token justifies repeating or synthesizing scarce data.

If the 1B gain disappears at 3B, the 1B result is treated as scale-specific noise rather than evidence for full-scale allocation.

---

## 11. Falsification criteria

I will reduce or redesign a lane if:

- Indic improvement saturates despite repeated exposure;
- synthetic Indic performs worse than native-data repetition;
- agentic synthetic trajectories improve formatting but not executable success;
- reasoning traces increase answer length without increasing correctness;
- long-context training reduces short-context quality without improving retrieval;
- a lane's benchmark improvement is smaller than the regression created elsewhere.

This makes the mixture falsifiable rather than aspirational.

---

## 12. Cleaning priority

Cleaning effort now follows scarcity, not raw corpus size.

1. **Agentic:** executable/replayable trajectories, schema validation, final-state verification.
2. **Reasoning:** answer verification, difficulty tags, reasoning-length tags, contamination checks.
3. **Indic verified/native:** script-aware normalization, provenance, deduplication and benchmark decontamination.
4. **Long-context:** document completeness, cross-document deduplication and sequence-quality checks.
5. **STEM:** equation/code/citation validation.
6. **Code and web:** continue normal cleaning, but do not consume most cleaning capacity simply because these corpora are largest.

The next cleaned billion tokens should preferentially come from the lanes whose real supply is furthest below target.

---

## 13. Final commitments

| Decision | Commitment |
|---|---|
| Main capability mix | **37 / 24 / 14 / 12 / 6 / 5 / 2** |
| Indic provenance | **35 / 15 / 15 / 35 max** verified / unverified / translated / synthetic |
| Protected floor | **12% Indic + 4% reasoning + 2% agentic** |
| Anneal reserve | **40B / 2%** |
| Main supply risk | **Agentic** |
| Secondary supply risk | **Reasoning** |
| Proxy requirement | **1B and 3B matched-control experiments** |
| Promotion rule | Same beneficial direction at both scales with no major core regression |

The mixture is not trusted because the percentages look reasonable.

It is trusted only when the inventory can fund them and proxy runs show that the sampled tokens actually purchase the intended capability.
