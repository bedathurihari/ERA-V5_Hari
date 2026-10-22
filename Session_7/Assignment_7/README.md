# Session 7 — Assignment 7: Dynamic Kronecker Embeddings

**Author:** Hari · **Course:** ERA V5 · **Selected problem:** **3**

## What I am solving

The proposed Kronecker embedding reserves 32 positions for each word. That creates two problems: short words waste positional space, and words longer than 32 characters lose information. I investigate whether we can preserve the defining character-position structure without a fixed 32-character ceiling.

## My proposal: sparse, variable-length Kronecker coordinates

For an alphabet of size **A**, let **e_i** be the basis vector for position **i** and **e_c** the basis vector for character **c**. The Kronecker representation of that character occurrence is:

`K(i,c) = e_i ⊗ e_c`

Instead of storing the full dense tensor, I store its nonzero coordinate `(i,c)`. A word of length **L** becomes an ordered list of **L** coordinates, and its shape is `L × A`, not the same 32-position shape for every word.

Example:

```text
apple -> [(0,a),(1,p),(2,p),(3,l),(4,e)]
a     -> [(0,a)]
[33-character input] -> 33 coordinates; no truncation
```

Decoding requires only the coordinates, the alphabet mapping, and the length. This implementation defines a restricted lowercase/digit/space alphabet explicitly, and rejects unsupported characters instead of silently changing them.

**Proof sketch.** Two distinct strings must differ either in length or at some character position. Distinct lengths yield distinct coordinate counts. At the first differing position, their character indices differ, so their coordinate sets differ. Therefore the representation is injective over the specified alphabet and finite strings. Reconstruction is exact.

**Storage trade-off.** Sparse coordinates use O(L) entries compared with O(32A) dense entries in the fixed scheme. They avoid wasting reserved positions for short words; they do **not** guarantee lower byte memory in every real implementation, and a dense expansion still costs O(LA). The sequence-processing cost remains length-dependent.

## What I tested

1. **Lossless round trip:** lengths 0, 1, 31, 32, 33, 64, 128, and 1,024 decoded exactly.
2. **Collision test:** `"a"*32 + "a"` and `"a"*32 + "b"` become identical after fixed-32 truncation, but stay distinct under dynamic coordinates.
3. **Dense-equivalence test:** active entries of dense Kronecker vectors equal the sparse coordinates.
4. **Small transformer experiment:** a one-layer TransformerEncoder (32-dimensional hidden states, four heads) predicts the **last** character of random binary strings. Training uses lengths 8–96; testing uses held-out examples of lengths 33–96. Since the label is independent of the first 32 random characters, a first-32-only model has **50% theoretical expected accuracy**.

### Results (seed 7, 150 steps)

| Measure | Outcome |
| --- | ---: |
| Unit tests | 4 passed |
| Roundtrip length | Tested through 1,024 |
| Dynamic Transformer, 480 held-out examples | **100% accuracy** |
| Fixed-32 theoretical expected accuracy | **50%** |
| Training duration in local CPU environment | 1.81 s |

The baseline figure is a *theoretical expectation*, **not** a separately trained baseline. The model uses ordinary learned character embeddings and positional embeddings over the variable-length sequences; the experiment demonstrates downstream consumption of variable-length inputs, **not** an end-to-end learned Kronecker factorization. In addition, the task permits direct access to the last valid position, making it an easy sanity check rather than evidence for general language understanding.

## Reproduce

Python 3.10+ and PyTorch required.

```bash
pip install torch
python -m unittest -v
python dynamic_kronecker.py
python train.py --steps 150 --batch 24 --seed 7
```

The training script writes `results.json`. Results depend on platform and PyTorch version; seed, workload, and evaluation split are specified.

## What the result does and does not establish

**Established:** an unrestricted *mathematical length* coordinate scheme (subject to finite machine resources), exact decoding on the supported alphabet, noncollision where truncation collides, and a working toy Transformer that consumes lengths greater than 32.

**Not established:** improved perplexity, reduced GPU memory for real LLM training, faster inference, superior performance against byte/character tokenization, or generalization to lengths unseen during training.

## Next experiment

I would benchmark a genuinely sparse projection `sum_i U_i ⊗ V_{c_i}` against fixed-32 Kronecker, byte models, and a variable-length character Transformer on matched parameter counts. The main metrics should be reconstruction rate, loss/perplexity, peak memory, tokens/second, and generalization to **unseen lengths**. A promising paper contribution would require performance and efficiency advantages beyond simply eliminating truncation.

## Files

- `dynamic_kronecker.py` — sparse/dense representations, exact decoder, basic proofs
- `test_dynamic.py` — four reproducible correctness tests
- `train.py` — tiny Transformer experiment and evaluation
- `results.json` — results from the local run

This is an initial research prototype, not a claim that the full Kronecker Embedding V2 problem is solved.
