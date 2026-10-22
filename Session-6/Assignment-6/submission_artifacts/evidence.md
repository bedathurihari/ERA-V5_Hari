# ERA V5 Session 6 — Generated Evidence

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer Integrity | PASS | `manifests/tokenizer.json; manifests/*.manifest.json` |
| Evaluation Firewall | PASS | `ledgers/firewall.json` |
| Packing Correctness | PASS | `manifests/packed_batches.json` |
| Mixture Compliance | PASS | `ledgers/mixture_audit.json` |
| Opus Audit | PASS | `ledgers/opus.json` |
| Crash Recovery | PASS | `checkpoints/before_crash.json; ledgers/consumption.json` |
| Replay | PASS | `ledgers/replay.json` |
| Learning Trace | PASS | `ledgers/learning.json` |
| Throughput | PASS | `performance.json` |
| Fork | PASS | `checkpoints/fork.json` |
| Checkpoint Integrity | PASS | `checkpoints/final.json` |

**Loss-bearing tokens:** 198; **Packing utilization:** 1.000; **Useful tokens/s:** 294.1.

Note: original example corpus; byte tokenizer and bigram model deliberately small. Data integrity and recovery are the evaluated behaviors.