# P5 — the oracle-attributed A/B (per length)

Prompts are P2's own arms' prompt files; oracle = llama.cpp-SYCL `qwen4exp` on the same IQ3_S file, f16 KV,
-ub 2048 unless stated; both sides greedy, same prompt ids, 256 requested at every length.
`margin` is the oracle's own top1-top2 log-prob gap at that position; `decided?` asks whether it clears the
oracle's own -ub band (0.106 nats, I2); `ub-512 agrees?` asks whether the oracle's -ub 512 instantiation
emits OUR token at that position.


## ctx 4096  (prompt 3832 ids)

oracle: 148 generated, prefill 492.67512177059626 tok/s, decode 18.010256289622564 tok/s; its own top1-top2 margins: min 0.0482 / median 8.2204 / max 20.9223 nats; 3 of 148 positions below 0.15 nats

| arm | prefill tok/s | decode tok/s | gen | first div | @div oracle vs engine | margin (nats) | decided? | ub-512 agrees? | prefix agreed | aligned matched |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped | 213.6 | 21.30 | 150 | 78 | 449 vs 264 | 0.1731 | no | YES | 78/79 (98.7%) | 110/150 (73.3%) |
| batched | 281.3 | 21.70 | 137 | 78 | 449 vs 264 | 0.1731 | no | YES | 78/79 (98.7%) | 109/148 (73.6%) |
| shipped-fp16 | 178.0 | 21.50 | 148 | **none** | - vs - | n/a | - | n/a | 148/148 (100.0%) | 148/148 (100.0%) |
| batched-fp16 | 280.5 | 21.30 | 137 | 78 | 449 vs 264 | 0.1731 | no | YES | 78/79 (98.7%) | 109/148 (73.6%) |


## ctx 32768  (prompt 32256 ids)

oracle: 256 generated, prefill 380.80207854467875 tok/s, decode 15.028729626788905 tok/s; its own top1-top2 margins: min 0.0120 / median 9.4823 / max 24.0151 nats; 3 of 256 positions below 0.15 nats

| arm | prefill tok/s | decode tok/s | gen | first div | @div oracle vs engine | margin (nats) | decided? | ub-512 agrees? | prefix agreed | aligned matched |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped | 342.8 | 21.80 | 256 | 101 | 16 vs 31995 | 0.0436 | no | no | 101/102 (99.0%) | 162/256 (63.3%) |
| batched | 765.2 | 22.30 | 256 | 75 | 3983 vs 3992 | 0.0120 | no | YES | 75/76 (98.7%) | 162/256 (63.3%) |
| shipped-fp16 | 275.2 | 21.60 | 256 | 75 | 3983 vs 3992 | 0.0120 | no | YES | 75/76 (98.7%) | 162/256 (63.3%) |
| batched-fp16 | 765.3 | 21.60 | 256 | 75 | 3983 vs 3992 | 0.0120 | no | YES | 75/76 (98.7%) | 255/256 (99.6%) |

