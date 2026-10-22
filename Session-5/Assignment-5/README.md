# Assignment 5 — V5 Mixture and Curriculum Plan

## My approach

For this assignment, I am treating the mixture as a hypothesis, not as a fixed answer.

My main goal is to make sure every capability gets enough training budget to matter, but at the same time I do not want to assign a large percentage to a lane if the inventory does not have enough real data to support it.

I am using a total training budget of **2.00T tokens**.

I am keeping:

- **1.96T tokens (98%)** for the main curriculum
- **40B tokens (2%)** separately for the final anneal / cooldown

So the 40B anneal is not counted again inside the 1.96T mixture.

---

## 1. My proposed main mixture

| Capability lane | Share | Tokens from 1.96T | Main evaluation |
|---|---:|---:|---|
| General web / knowledge | **37%** | **725.2B** | MMLU / held-out perplexity |
| Code | **24%** | **470.4B** | LiveCodeBench / code pass@1 |
| Indic | **14%** | **274.4B** | MILU / Indic evaluation |
| STEM / mathematics | **12%** | **235.2B** | GPQA / math accuracy |
| Reasoning traces | **6%** | **117.6B** | reasoning / math exact-match |
| Long-context | **5%** | **98.0B** | RULER / long-context retrieval |
| Agentic / tool-use | **2%** | **39.2B** | BFCL / tau-bench / SWE-style execution |
| **Total** | **100%** | **1.96T** | |

Then I keep another **40B tokens for annealing**.

So:

**1.96T main curriculum + 40B anneal = 2.00T total**

I am giving the largest shares to general data and code because those are both important and well supplied in the inventory.

I am giving smaller but protected shares to Indic, reasoning, long-context and agentic data because these are important capabilities, but the available data is much more limited.

---

## 2. Checking my mixture against the available inventory

Before fixing these percentages, I compared them against the Session 5 inventory.

| Lane | Approx. available supply | My requirement | My conclusion |
|---|---:|---:|---|
| General web + STEM | **~4.84T** | 960.4B combined | Enough supply |
| Code | **~1.103T** | 470.4B | Enough supply |
| Indic | **~275.9B** | 274.4B | Almost fully using the available pool |
| Reasoning + math | **~85.1B** | 117.6B | Short by about 32.5B |
| Long-context | **~100B** | 98B | Very close to the available limit |
| Agentic / tool-use | **~0.6–0.63B** | 39.2B | Extremely short |

This is important because the percentage alone can be misleading.

For example, **2% agentic sounds small**, but 2% of 1.96T is 39.2B tokens. The real agentic inventory is only around 0.6B tokens.

So I cannot claim that I have 39.2B of unique real agentic data. Most of this lane would have to come from verified synthetic trajectories or carefully controlled replay.

The same issue exists for reasoning, but it is much smaller. I need 117.6B and the inventory is around 85.1B, so the effective reuse / generation requirement is:

**117.6 / 85.1 ≈ 1.38×**

I consider this manageable compared with the agentic lane.

For long-context, I am almost exactly at the available supply. This means cleaning losses matter a lot. If the usable long-context data drops below my target after cleaning, I would reduce the lane rather than fill it with low-quality data just to maintain 5%.

---

## 3. Indic split

I do not want to write only "14% Indic" because all Indic tokens are not equal.

I am splitting my **274.4B Indic tokens** into four tiers.

| Indic tier | Share inside Indic | Tokens |
|---|---:|---:|
| **Verified native** | **35%** | **96.0B** |
| **Unverified native / crawl** | **15%** | **41.2B** |
| **Translated** | **15%** | **41.2B** |
| **Synthetic** | **35% maximum** | **96.0B max** |
| **Total** | **100%** | **274.4B** |

My preference order is:

**verified native > good unverified native > translated > synthetic**

The 35% synthetic number is a maximum, not something I want to force.

If I get more good verified native data after cleaning, I would reduce the synthetic share.

From the inventory, the main Indic sources I am considering are:

- Sangraha verified
- Sangraha unverified
- Sangraha synthetic
- IndicCorpV2
- BPCC
- Samanantar
- translated data generated from approved source data when required

For translated or synthetic data, I would keep the lineage information.

At minimum I want to know:

- original source
- source language
- target language
- script
- translation or generation method
- verifier status
- parent / seed record
- deduplication ancestry

I do not want multiple synthetic versions of one source to be treated as completely independent data.

---

## 4. Which datasets I am using for each capability

### General web / knowledge

For this lane I would mainly use:

- DCLM
- FineWeb-Edu
- other high-quality general web/reference data from the inventory

I want this lane to provide broad language understanding, factual knowledge and normal discourse.

My main checks here are general held-out perplexity and broad knowledge benchmarks.

### Code

For code I would use:

- The Stack v2
- D3/V4 code
- CommitPack
- repository code
- tests
- commits
- documentation

I want this lane to support generation, debugging and repository-level understanding.

My main benchmark is LiveCodeBench-style execution accuracy, and later I would also look at SWE-style tasks.

### STEM / mathematics

For STEM and math I would use:

- Proof-Pile
- D4-STEM
- peS2o
- other technical/scientific subsets from the inventory

This lane supports GPQA-style science and the base knowledge needed for reasoning.

### Reasoning

For reasoning I would use:

- AON/V4
- OpenThoughts
- OpenMath
- OpenR1
- Numina

The inventory gives around **85.1B** reasoning/math tokens, while my target is **117.6B**.

So I need around **1.38× effective use**.

I would prefer selective repetition of high-quality verified traces before generating large amounts of new reasoning data.

For generated reasoning:

- math answers should be numerically or symbolically checked
- code reasoning should be compiled or tested
- factual tasks should use answer/reference verification where possible

I do not want to keep a long reasoning trace just because it looks intelligent. The final result should be verifiable.

### Long-context

For long-context I am using the inventory buckets:

- **repo-packed long-context — ~60B**
- **book-length long-context — ~40B**

That gives around 100B total.

My target is 98B, so I have almost no buffer.

I would rather reduce the percentage slightly if cleaning removes too much data than create weak long sequences only to satisfy the target.

### Agentic / tool-use

For agentic data I am using:

- SWE-Gym
- SWE-smith
- OpenHands trajectories
- ToolBench
- ToolACE
- Glaive
- Nexus
- xLAM
- Hermes

Together these provide only around **0.6B real tokens**, while my target is **39.2B**.

This is clearly the most data-starved capability in my plan.

So I would generate additional agent trajectories, but I would not accept them just because the JSON or tool-call syntax is correct.

For me, a synthetic agent trajectory is useful only if:

1. the tool call is valid,
2. the action sequence can be replayed where possible,
3. permissions/rules are followed,
4. the final state is correct,
5. failures and recovery are realistic.

This is the biggest area where synthetic generation is unavoidable.

---

## 5. My protected Always-On floor

I do not want the selector to remove scarce but important data just because it looks unusual compared with normal web text.

So I am setting these hard floors:

| Lane | Minimum allowed |
|---|---:|
| Indic | **12%** |
| Reasoning | **4%** |
| Agentic | **2%** |
| **Total protected floor** | **18%** |

These are minimums, not target averages.

For example:

- Indic target is 14%, but it should never go below 12%
- Reasoning target is 6%, but it should never go below 4%
- Agentic target itself is 2%, so I am protecting the whole lane

The selector can adjust the remaining mixture, but it cannot cross these floors.

My reason is simple: multilingual text, reasoning traces and tool trajectories can look statistically unusual and may get rejected by generic quality filters even when they are exactly the data I need.

---

## 6. Anneal reserve

I am holding back **40B tokens**, which is **2% of the full 2T training budget**, for the final anneal.

This is separate from the 1.96T main curriculum.

My proposed anneal mix is:

| Lane | Share of anneal | Tokens |
|---|---:|---:|
| High-quality general | 10% | 4B |
| Code | 20% | 8B |
| Verified Indic | 20% | 8B |
| STEM | 10% | 4B |
| Reasoning | 20% | 8B |
| Long-context | 10% | 4B |
| Agentic | 10% | 4B |
| **Total** | **100%** | **40B** |

During the anneal I want more high-confidence, high-value examples and less broad web data.

I would not put synthetic examples into the anneal just because they are difficult. They should first pass the appropriate verifier.

---

## 7. My curriculum

I would not keep exactly the same data distribution from the first token to the last token.

I would change the emphasis gradually.

### Stage A — foundation

I would start with more:

- high-quality general web
- native Indic
- straightforward code

Most examples would be easy to medium difficulty.

### Stage B — code and STEM

Then I would increase:

- executable code
- mathematics
- technical/scientific data

At this point I would introduce more medium-depth reasoning.

### Stage C — reasoning

Next I would increase verified reasoning traces.

I would still keep the Indic and agentic minimum floors active.

### Stage D — long-context and agentic

Later I would increase:

- repo-packed long-context data
- book-length sequences
- multi-step tool trajectories
- failure and recovery trajectories

I think long-context and agentic data make more sense after the model already has a reasonable base representation.

### Stage E — anneal

Finally I would use the protected 40B anneal set at a lower learning rate.

I would blend the transitions rather than making a sudden change between stages.

---

## 8. Difficulty bands

I would tag training examples by difficulty so the sampler can control how quickly hard examples are introduced.

| Band | What I mean | Example |
|---|---|---|
| **D0 — foundational** | direct recognition / language | Translate a simple Telugu sentence into English |
| **D1 — elementary** | one operation or one fact | 40 seats × 3 buses |
| **D2 — intermediate** | several connected steps | Solve a quadratic equation and verify both roots |
| **D3 — advanced** | undergraduate-level multi-step reasoning | Derive the gradient of softmax cross-entropy |
| **D4 — expert** | multiple interacting concepts | Compare two attention implementations at 128K context for memory and compute |
| **D5 — research / agentic** | open-ended diagnosis and verification | Debug a failing repository, patch it, run the tests and verify the final state |

Early in training I would use mostly D0-D2.

Later I would gradually increase D3-D5 without completely removing the easier examples.

---

## 9. Reasoning-length bands

I am treating **difficulty** and **reasoning length** as separate things.

A hard question can sometimes have a short solution, and an easy question should not be expanded into thousands of tokens just because longer reasoning is available.

My reasoning-length bands are:

| Band | Approx. trace length | Example |
|---|---:|---|
| **LOW** | **≤128 tokens** | Directly apply inclusion-exclusion to a simple counting problem |
| **MEDIUM** | **129–512** | Show the individual counts, combine them and check the boundary |
| **HIGH** | **513–2,048** | Solve a harder algebra/algorithm problem, compare alternatives and verify the answer |
| **ULTRA** | **2,049–4,096+** | Repository-level debugging across files with tool use, retries, tests and final-state verification |

My rule is:

**longer reasoning is not automatically better reasoning.**

For example, for:

> How many integers from 1 to 1000 are divisible by 3 or 5?

LOW or MEDIUM is enough.

I would not create an ULTRA trace for that question.

I would reserve ULTRA reasoning for tasks that actually need long-horizon work.

---

## 10. How I would test this before full-scale training

I do not want to trust these percentages just because they look reasonable.

I would first test them using smaller proxy models.

### My hypothesis

My hypothesis is:

> Protecting Indic, reasoning and agentic data should improve those capabilities enough to justify the tokens taken away from abundant general/code data, without causing a large regression in the core model.

### Baseline

For the baseline I would use a supply-proportional mixture with no protected floor.

### My proposed mixture

For the experimental model I would use:

**37 / 24 / 14 / 12 / 6 / 5 / 2**

for:

**General / Code / Indic / STEM / Reasoning / Long-context / Agentic**

with the Always-On floors active.

Everything else should remain the same:

- architecture
- tokenizer
- optimizer
- learning-rate schedule
- training-token count
- sequence-length policy
- evaluation harness

This is important because otherwise I cannot say the difference came from the mixture.

### 1B proxy run

First I would run the comparison at approximately **1B parameters**.

I would measure:

- general held-out perplexity
- MILU / Indic performance
- code pass@1
- GPQA/math/reasoning accuracy
- BFCL-mini/tool success
- long-context retrieval performance

My initial pass criteria would be:

- Indic held-out PPL improves by at least **5% relative**
- reasoning accuracy improves by at least **5% relative**
- agentic/tool success improves by at least **8% relative**
- code pass@1 drops by less than **2 percentage points**
- general held-out PPL gets worse by less than **3%**
- long-context retrieval does not regress

If the result does not meet these conditions, I would change the mixture before the next experiment.

### 3B proxy run

After that I would repeat the test at around **3B parameters**.

I do not expect exactly the same improvement percentage as the 1B run.

What I want to see is the same direction of improvement.

I would only trust the mixture for full-scale training if:

1. Indic improves at both 1B and 3B
2. reasoning improves at both scales
3. agentic/tool success improves at both scales
4. general and code performance do not materially regress
5. training remains stable when curriculum stages change

If an improvement appears at 1B but disappears or reverses at 3B, I would not treat the 1B result as enough evidence.

---

## 11. What would make me change this plan

I would change the mixture if I see any of these:

- Indic performance stops improving while repetition keeps increasing
- translated or synthetic Indic performs worse than repeating good native data
- agentic synthetic data improves tool-call formatting but not actual task completion
- longer reasoning traces make answers longer without making them more correct
- long-context training hurts normal short-context performance without improving long-context tasks
- the gain from one protected lane is smaller than the regression it causes in the rest of the model

So these percentages are my starting hypothesis, not numbers I want to defend even if experiments prove them wrong.

---

## 12. Data-gating and cleaning

For me, this mixture is useful only if the cleaned and documented tokens actually exist behind it.

For every cleaned shard I want the manifest to include:

- source
- licence
- cleaning-script hash
- content hash
- token count
- language(s)

I would not mark the data-gating requirement as complete until the cumulative cleaned-token target and manifests are actually available.

Based on my mixture, my next cleaning priority would be:

1. **Agentic** — because it is by far the most starved lane
2. **Reasoning** — because there is a real supply gap
3. **Verified/native Indic** — because I want to reduce dependence on synthetic Indic
4. **Long-context** — because my target is almost equal to the full available supply
5. **STEM**
6. **Code and general web**

I would not spend most of the cleaning effort on web just because web is the largest corpus.

I want the cleaning effort to go where the mixture is most supply-constrained.

---

## 13. Final summary

My final proposal is:

| Decision | My choice |
|---|---|
| Total budget | **2.00T** |
| Main curriculum | **1.96T** |
| Anneal reserve | **40B** |
| Main mixture | **37 / 24 / 14 / 12 / 6 / 5 / 2** |
| Indic split | **35 / 15 / 15 / ≤35** verified / unverified / translated / synthetic |
| Protected floors | **Indic 12%, Reasoning 4%, Agentic 2%** |
| Biggest supply problem | **Agentic** |
| Next biggest supply problem | **Reasoning** |
| Long-context policy | use real/repacked supply and reduce share if cleaning removes too much |
| Validation | **1B proxy first, then 3B proxy** |

The main point of my plan is that I do not want to choose percentages independently of the data that actually exists.

If a lane is short, I want to say clearly whether I am repeating data, generating data, or reducing the target.

And before using this mixture at full scale, I want the 1B and 3B proxy runs to show that these token allocations are actually buying the capabilities I expect.
