# P5 — which attention path is right? The oracle-attributed prompt-vs-batched comparison

Card `t_bc967b65`, repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`. Base `a8c562c` (P4); the harness is
`bea1cc0`, the evidence is `p5/evidence/`. Nothing pushed (origin has no `sycl-xpu` branch).

## 0. Verdict — the oracle cannot separate the two paths: a tie, and the 2.21x stays refused

**Measured.** Over the two lengths where an independent reference exists (4K and 32K prompts; the 128K reference
cannot be produced on this box — §6), every divergence between our engine and the oracle sits at an oracle
top1-top2 margin between **0.0120 and 0.1731 nats** — and the oracle's own **-ub 512** instantiation, on the same
prompt, emits *our* token at exactly those positions (4K index 78, 32K index 75). One divergence is ours alone
(the 32K index-101 cluster at a 0.0436-nat margin, in 3 of our 4 arms). Both attention kernels are FP32-accurate
against FP64 (2.13-2.15e-06 vs 1.90-2.14e-06 on an output of scale 3.2-3.6, and 1.2e-06 of scale apart from each
other, §5).

**Verdict: a tie, and therefore no change to the default configuration.** On the raw per-position metric the
shipped prompt path is ahead where an oracle exists (4K with matched f16 KV: shipped 148/148 tokens identical to
the oracle, batched 78/79 then a different continuation; 32K at the config of record: shipped first diverges at
101, batched at 75). On the band-aware metric the batched path is ahead (the only divergence no oracle
instantiation reproduces — 32K/101 — is absent from the **batched**-fp16 arm and present in the other three). The
two readings point in opposite directions, both gaps are single near-tie events of 0.01-0.17 nats, and that is a
tie, not a winner.

**The ship condition in the card is not met, so the fast path is not landed.** The card says land it "if and only
if the fast path matches or beats the shipped path's oracle agreement at all three lengths". It does not match at
4K, and at 128K there is no oracle at all. The claim a shipped change would have to state — "the new stream is
*closer to the oracle*" — is not true at 4K (78/79 against 148/148) and not true at 32K at the config of record
(75 against 101). The 2.15-2.23x prefill remains the measured headroom, not a shipped change.

## 1. What was measured, and on what

* **The A/B.** `p5/p5_arm.sh` runs one engine arm on the config of record (`strata-sycl-iq3s.json`: both B70s,
  `--kv int8`, `--layer-split auto`, `--no-capture`, `--kv-resident`, MTP on, `ZE_AFFINITY_MASK` unset, warm
  program cache `sycl-cache/m6c`) through the engine's own serve protocol, 256 greedy tokens at every length.
  `shipped` = `qsa_prompt_attn_batch` (the portable v1 kernel); `batched` = `STRATA_PROMPT_ATTN_OLD=1` →
  `qsa_decode_attn_batch`, the engine's own batched decode attention. The runner sets and clears that switch
  itself and records which path ran (the shipped path prints the portable-v1 line at load, the batched path does
  not — `prefill.cpp:1621`). Prompts are P2's own arms' files: 3,832 / 32,256 / 129,024 ids.
* **The oracle.** `p5/p5_oracle_launch.sh` + `p5_oracle_probe.py` reuse I2's harness: llama.cpp-SYCL
  (`qwen4exp`, tree read-only) on the **same** GSQ-RCO IQ3_S file, same two cards, **f16 KV**, fed the same
  prompt-id file, greedy for 256 tokens, with `n_probs 5` so the oracle's own top-5 log-probs come back at every
  position. 4K: prefill 492.7 tok/s, decode 18.0. 32K: 380.8, 15.0.
* **The comparison.** `p5/p5_compare.py` reports per arm: the prefix view up to the first divergence (I2's), the
  aligned view (difflib — an insertion otherwise charges every later position), the nats margin at every
  difference, and the arm-vs-arm diff with no oracle in the loop.
* **The oracle's own resolution.** `p5/p5_oracle_ub512_chain.sh` runs the same prompts through the same oracle at
  **-ub 512** (I2's 5b method) — that is the measurement of the reference's own band, and it is what the margins
  above have to be compared against.
* **The kernels on their own axes.** `p5/p5_parity.sh` runs `build-sycl/sycl_prompt_attn_parity` at ctx 3,832 and
  ctx 32,768 on card 0: both kernels' max error against an FP64 host reference, and against each other, no model.

## 2. The A/B reproduces P2 exactly, and the three-way table

The int8 arms reproduce P2's recorded streams **byte for byte**: 4K `ad985c23…` (150 ids, P2's and P1b's record),
32K `124a3cd3…` (256), 128K `c60e72d3…` (256); the batched 32K arm is P2's old-attention stream `c52712f4…`.
Speeds: **4K 213.6 → 281.3 tok/s (1.32x); 32K 342.8 → 765.2 (2.23x); 128K 411.5 → 883.0 (2.15x)**; per-chunk
`qsa attn` 52,754 of 70,279 ms (75.1%) → 8,097 of 25,640 (31.6%) at 32K, 185,511 of 282,596 (65.6%) → 28,445 of
126,093 (22.6%) at 128K.

Oracle margins over its own stream: 4K min 0.0482 / median 8.22 nats (3 of 148 positions below 0.15); 32K min
0.0120 / median 9.48 (3 of 256 below 0.15).

| ctx | arm | prefill tok/s | decode tok/s | gen | first div vs oracle | margin there | decided by the oracle? | ub-512 oracle emits ours? | prefix agreed | aligned matched |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | shipped (int8) | 213.6 | 21.3 | 150 | 78 | 0.1731 | no | YES | 78/79 (98.7%) | 110/150 |
| 4096 | batched (int8) | 281.3 | 21.7 | 137 | 78 | 0.1731 | no | YES | 78/79 (98.7%) | 109/148 |
| 4096 | shipped (fp16) | 178.0 | 21.5 | 148 | **none** | — | — | n/a | **148/148 (100%)** | 148/148 |
| 4096 | batched (fp16) | 280.5 | 21.3 | 137 | 78 | 0.1731 | no | YES | 78/79 (98.7%) | 109/148 |
| 32768 | shipped (int8) | 342.8 | 21.8 | 256 | 101 | 0.0436 | no | no | 101/102 (99.0%) | 162/256 |
| 32768 | batched (int8) | 765.2 | 22.3 | 256 | 75 | 0.0120 | no | YES | 75/76 (98.7%) | 162/256 |
| 32768 | shipped (fp16) | 275.2 | 21.6 | 256 | 75 | 0.0120 | no | YES | 75/76 (98.7%) | 162/256 |
| 32768 | batched (fp16) | 765.3 | 21.6 | 256 | 75 | 0.0120 | no | YES | 75/76 (98.7%) | **255/256 (99.6%)** |
| 131072 | shipped (int8) | 411.5 | 19.8 | 256 | — | — | — | no oracle (§6) | — | — |
| 131072 | batched (int8) | 883.0 | 19.0 | 256 | — | — | — | no oracle (§6) | — | — |

"decided by the oracle" means the margin at the first divergence clears the oracle's own implementation band
(0.106 nats, I2's -ub 512 measurement) **and** the oracle's -ub 512 instantiation does not itself emit our token
there. No arm of ours has one: every divergence this card measured sits at a position the reference cannot
decide. The same table with the fp16 arms is regenerated by `p5/p5_table.py` into `p5/TABLE.md`.

**The aligned column is not an accuracy column and must not be read as one.** After the first divergence the two
streams are in different contexts, so every later position is a different continuation; the batched-fp16 arm's
255/256 exists because it re-syncs after its one inserted token, while the shipped-fp16 arm's 162/256 is what
losing the same 32K coin flip costs when the text then goes another way. Only the first divergence is evidence
about a kernel, and a reader who wants one number should use that column.

## 3. Every divergence is at a margin the oracle itself changes its mind on

`p5/p5_evidence/oracle-band.txt`. The oracle's own -ub 512 instantiation, same prompt, same file, same cards:

* **4K**: at index 78 the ub-2048 oracle chooses `449, 39328, 7359` (margin **0.1731 nats**); the ub-512 oracle
  chooses **`264, 9640, 24277, 4621`** — the exact tokens our three diverging arms emit there (shipped-int8,
  batched-int8, batched-fp16). The ub-512 oracle then runs 185 tokens against 148. So the oracle's own choice at
  4K/78 is not stable under its own prompt-batch size, and one of its two answers is ours.
* **32K**: the ub-512 oracle differs from the ub-2048 oracle by **one inserted token at index 75** (margin
  **0.0120 nats**) — `3992`, the exact token the batched arms and shipped-fp16 emit there — and by nothing else
  (255/256 positions matched).
* **32K index 101** (margin 0.0436 nats) is the one divergence **both** oracle instantiations disagree with: they
  both emit `16`, and three of our four arms emit the `31995` cluster. `batched-fp16` matches both.

The same 0.0120-0.1731 nat band is where I2's `-ub` perturbation moved the oracle's own choices (0.064-0.106
nats). Nothing our arms did at either length exceeds 0.18 nats.

## 4. The mechanism, if it is one — precision vs algorithm, separated by measurement

* **Precision (int8 vs f16 KV) is what moves the 32K stream the most.** With f16 KV the shipped arm's 32K first
  divergence moves 101 → 75 (it loses the int8-caused `31995` cluster and only keeps the shared 75 flip), and the
  batched arm's 32K stream stops losing that cluster entirely (255/256 against the oracle). At 4K the picture is
  the opposite and sharper: **`batched-fp16` is byte-identical to `batched-int8`** (md5 `9428de79…` in both), so
  the batched kernel's answer at 4K/78 does not depend on KV precision at all, while `shipped-fp16` is
  byte-identical to the oracle. That asymmetry is the one reproducible difference between the two kernels at 4K.
* **Algorithm (the two kernels' numerics).** `p5/evidence/parity-p5.txt`, both kernels on the same synthetic
  fixture against an FP64 reference: ctx 3,832 int8 old 2.15e-06 / new 1.90e-06, fp16 1.33e-06 / 1.62e-06; ctx
  32,768 int8 2.13e-06 / 2.14e-06, fp16 1.89e-06 / 1.90e-06 (output scale 2.7-3.6); the two kernels are
  1.2-1.3e-06 (int8) and 0.79-0.92e-06 (fp16) of scale apart. Both are FP32-accurate; neither is grossly wrong;
  the difference between them is a rounding path (chunk cells, thread count, warp reduction order, and the
  batched kernel's two-pass online softmax with `expf_fast` and a merge stage), not a difference in what they
  attend to — the pool, the page tables and the gather are shared and bit-exact in `qsa_parity`.
* **Structural difference in what the kernel attends to: not evidenced.** Nothing measured here shows the
  batched path reading a different cell set; the flips it produces are the same size as the oracle's own.
* **At the config of record the two arms differ by almost nothing**: at 4K their streams are identical for the
  first 100 tokens (first difference at 100, `shipped[100] = 20335` absent in the batched stream); at 32K 249 of
  256 positions match with a single insertion at 75; at 128K 80 of 256 match, first difference at 53.

## 5. Speeds, for the record

| ctx | arm | prefill tok/s | decode tok/s | wall s | qsa attn share of the chunk GPU timeline |
|---|---|---|---|---|---|
| 4096 | shipped int8 / batched int8 | 213.6 / 281.3 | 21.3 / 21.7 | 57 / 52 | (dequant-dominated at 4K) |
| 4096 | shipped fp16 / batched fp16 | 178.0 / 280.5 | 21.5 / 21.3 | 168 / 53 | — |
| 32768 | shipped int8 / batched int8 | 342.8 / 765.2 | 21.8 / 22.3 | 138 / 86 | 75.1% / 31.6% |
| 32768 | shipped fp16 / batched fp16 | 275.2 / 765.3 | 21.6 / 21.6 | 161 / 85 | — |
| 131072 | shipped int8 / batched int8 | 411.5 / 883.0 | 19.8 / 19.0 | 358 / 191 | 65.6% / 22.6% |
| 131072 | shipped fp16 / batched fp16 | 332.3 / 877.1 | 19.9 / 18.3 | 433 / 193 | — |

The fp16 4K shipped arm's 168 s wall is the one-time JIT of the fp16-KV kernel instantiation, and that is
measured, not inferred: `p5/p5_gaps.py` finds a **111.3 s** silence between two load lines in that arm's timeline
against **18.2 s** at the same point in the int8 arm (and 18.7 s in the *next* fp16 arm, which found the
instantiation compiled). Steady state is unaffected: prefill 178.0 against 213.6 tok/s.

## 6. The 128K oracle: the wall, measured twice

The oracle cannot produce a 128K token stream on this box. Two attempts at the primary configuration
(`-c 131072 -ub 2048 -ncmoe 0`, the same one that works at 4K and 32K):

* attempt 1 dies at **n_tokens 49,152 (progress 0.38, t = 286.11 s)** — `level_zero backend failed with error: 20
  (UR_RESULT_ERROR_DEVICE_LOST)` in `Error OP FLASH_ATTN_EXT`;
* attempt 2 dies at the same point (49,152, 286.11 s) with the same `UR_RESULT_ERROR_DEVICE_LOST`, reported from
  `ggml_sycl_mul_mat_id` (backtrace `ggml_sycl_error` ← `ggml_sycl_mul_mat_id` ← `ggml_backend_sycl_graph_compute`).

Both runs show the prompt rate collapsing with depth before the loss (484 tok/s at 2,048 tokens, 350 at 40,960,
174 at 49,152), and both end inside the expert/attention kernels rather than at load. This is consistent with
M6c's earlier finding that the oracle is unusable at 262,144; at 128K it is the depth, not the context size. A
bounded third attempt at `-ub 512` (four times smaller query batches) is recorded in
`p5/evidence/oracle-128k-wall.txt`; the 128K row of the table above therefore has **engine-vs-engine evidence
only**, and the ship decision does not rest on it.

## 7. What was NOT done, and what is not validated

* **Nothing is shipped and the default configuration is unchanged** — no source change, no rebuild beyond the
  no-op relink at HEAD, and the batched path remains behind `STRATA_PROMPT_ATTN_OLD=1`.
* **One sample per arm.** Every arm is a single run; the two 4K shipped-int8 runs (the smoke arm and the chain
  arm) are byte-identical with prefill 212.7 / 213.6 tok/s, which is the only repeat in this card. No variance
  estimate exists for the arms that differ, and the differences are single events — a sign test on one event per
  length is not a statistical statement.
* **The 128K row has no reference**, and the engine's two arms diverge at index 53 there (80/256 aligned match)
  with nothing to say which is closer.
* **The kernel-level mechanism is bounded, not located.** The parity fixture shows both kernels equally accurate
  in aggregate; it does not exercise the per-position case that flips a 0.17-nat tie, so "a rounding path that is
  locally larger than its aggregate error" is the honest reading, not a demonstrated defect. Locating it needs a
  per-position parity fixture (the engine's own q rows and selection at the divergent position), which this card
  did not build.
* **The oracle's own band is measured at two instantiations and two prompts** (-ub 2048 vs 512, I2's three-way
  check at 4K/2,700 tokens and P5's at 3,832/32,256). A wider band would need more perturbations (-ncmoe,
  different `-ts`), which were not run.
* Decode numbers move ±4% between arms (128K int8 19.8 → 19.0 tok/s) on window counts of 99 against a different
  number of MTP windows; no per-window comparison was made, and decode is not what this card changes.
* Nothing is pushed: origin has no `sycl-xpu` branch.

## 8. Files

| what | path |
|---|---|
| the harness | `p5/p5_arm.sh`, `p5/p5_oracle_launch.sh`, `p5/p5_oracle_probe.py`, `p5/p5_oracle_stop.sh`, `p5/p5_compare.py`, `p5/p5_pairdiff.py`, `p5/p5_streams.py`, `p5/p5_oracle_ub_compare.py`, `p5/p5_table.py`, `p5/p5_meta.py` — commit `bea1cc0` + this commit |
| the chains | `p5/p5_chain_engine.sh` (the 12 arms), `p5/p5_chain_oracle.sh`, `p5/p5_oracle_ub512_chain.sh`, `p5/p5_oracle_128k_ub512.sh`, `p5/p5_parity.sh` |
| the evidence, regenerated by `p5/p5_evidence.sh` | `p5/evidence/three-way-tables.txt`, `oracle-band.txt`, `arm-vs-arm.txt`, `parity-p5.txt`, `oracle-128k-wall.txt`, and the four chain logs |
| every arm's raw output | `~/strata-xpu/p5/runs/p5-<ctx>-<arm>[-fp16]/{out,err,tokens,log}.txt` (12 arms) |
| the oracle's streams | `~/strata-xpu/p5/oracle/oracle-{4k,32k}.json` (+ `-ub512`), `oracle-{4k,32k,128k}*-server.log` |
| the oracle's own context length, read from the GGUF | `p5/p5_meta.py`: `qwen4exp.context_length 262144`, 12 full-attention layers of 48, head_count_kv 2, key/value length 256 |
