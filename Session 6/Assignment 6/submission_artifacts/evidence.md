# V5 Training Data Execution Evidence

Generated from the actual run, manifests and ledgers.

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | `manifests/; tokenizer.json` |
| Evaluation firewall | PASS | `ledgers/blocked_shards.json` |
| Packing correctness | PASS | `ledgers/batch_catalog.json` |
| Mixture compliance | PASS | `mixture.json` |
| OPUS audit trail | PASS | `ledgers/opus.json` |
| Crash recovery | PASS | `checkpoints/precrash.json; ledgers/consumption.json` |
| Replay | PASS | `ledgers/replay.json` |
| Learning trace | PASS | `ledgers/learning.json` |
| Throughput | PASS | `performance.json` |
| Fork | PASS | `ledgers/fork.json` |
| Checkpoint | PASS | `checkpoints/final.json` |

Resume next batch: `batch-0010` and SHA-256 `d19b685717937c3741b5a50f10d36c46677856b64d3e67c19634ee9d9233a2a4`.
Replay interval: batches 3–7, all matching SHA-256 hashes.
Packing utilization: 100.00%; useful loss tokens/sec: 15126.15.
