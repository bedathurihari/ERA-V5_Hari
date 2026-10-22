# ERA V5 — Session 6 — Assignment 6

This is my small end-to-end training data execution system. It focuses on deterministic data lineage, actual trainable-model updates, ledger-driven recovery and generated evidence.

## Run

```bash
python run_demo.py
python -m unittest -v test_system.py
```

Python 3.9+; standard library only. The first command recreates `submission_artifacts/` from scratch. I use a deliberately tiny whitespace tokenizer and trainable bigram classifier.

## Design

- Frozen training-only vocabulary and hashed per-document token shards and manifests, including source, license, languages, cleaning hash and split.
- Strict train versus evaluation/validation firewall before OPUS or scheduling.
- OPUS accepts, rejects, defers, and overrides Indic examples for protected floors; two stage curriculum uses weighted deficit selection.
- Per-batch attention/loss masks, lower triangular causal mask, position IDs, immutable identifiers, source spans and SHA-256 hashes.
- Trainable two-class sampled next-token SGD objective with token-wise losses attached to source shard and batch identifiers.
- Checkpoint containing model state and both ledger offsets; deliberate exception, state restoration, exact next-batch matching, no skipped or repeated batch IDs.
- Historical replay verifies IDs, spans and hashes. A separate fork trains three batches from an earlier checkpoint with a different learning rate.
- Measured wall time, packing utilization, and useful loss-bearing token throughput with machine-readable evidence.

## Scope

This is a small correctness demo, not production-scale training. It uses one document per batch (no cross-document attention leakage), and a sampled binary next-token loss rather than full-vocabulary LM cross-entropy. Production systems need atomic fsync/rename or a transactional ledger/checkpoint commit and more sophisticated packing and distributed failure recovery. Source shards may be reused across epochs; batch IDs are never repeated. Throughput varies per machine.

The generated output is `submission_artifacts/` with logs, evidence bundles, manifest records, shard data, consumption/learning ledgers, checkpoints, replay/fork trace and performance metrics.
