# P5 — which attention path is right? The oracle-attributed prompt-vs-batched comparison

Card `t_bc967b65`, repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`. Base `a8c562c` (P4); the harness is
`bea1cc0`, the report and evidence are the two commits after it. Nothing pushed (origin has no `sycl-xpu` branch).

## 0. Verdict — the oracle sides with the shipped path: the 2.21x stays refused

**The evidence, in one paragraph.** Twelve engine arms (the shipped prompt attention vs `STRATA_PROMPT_ATTN_OLD=1`'s
batched decode attention, at int8 and fp16 KV, at 4K/32K/128K, 256 greedy tokens each) were compared token by token
against an independent reference (llama.cpp-SYCL `qwen4exp` on the same GSQ-RCO IQ3_S file, same prompt ids, f16 KV,
greedy), and every divergence carries the oracle's own top1-top2 margin. **At the two lengths where the KV
precision matches the reference's (f16), the shipped kernel reproduces the oracle exactly — 148/148 tokens at 4K and
256/256 at 128K — and the batched kernel diverges once at each (4K index 78 at a 0.1731-nat margin, 128K index 59 at
0.1436 nats, the only margin in this card above the oracle's own measured band of 0.106 nats). At the config of
record (int8 KV) the shipped arm also diverges later at every length: 4K equal at index 78, 32K 101 against 75, 128K
132 against 53.** The batched path's one favourable reading is 32K, where both fp16 arms sit out the same 0.0120-nat
insertion and the batched arm then re-syncs with the oracle while the shipped arm loses a later 0.0436-nat flip.

**Verdict: refused, and the default configuration is unchanged.** The card's ship condition — the fast path must
match or beat the shipped path's oracle agreement at all three lengths — is not met: the batched path is strictly
worse at 4K and at 128K, and at 32K it is level on the attributable event. The claim a shipped change would have to
state, "the new stream is *closer to the oracle*", is false at two of the three lengths.

**Weakness, stated plainly.** Four attributable events in eight matched-KV arm-runs is not a statistical
statement, and three of the four sit at margins (0.0120-0.1731 nats) inside or at the edge of the band in which the
oracle itself changes its own mind under a prompt-batch change — the oracle's `-ub 512` instantiation *emits our
token* at 4K/78 and 32K/75. So this is not "the batched kernel is broken"; it is "the batched kernel's difference
from the shipped kernel is a near-tie resolution difference, and at the resolution this reference offers, the
shipped kernel is the one that matches it". That is enough to refuse a 2.2x change whose whole justification was
correctness, and not enough to call the fast kernel a defect.

## 1. What was measured, and on what

* **The A/B.** `p5/p5_arm.sh` runs one engine arm on the config of record (`strata-sycl-iq3s.json`: both B70s,
  `--kv int8`, `--layer-split auto`, `--no-capture`, `--kv-resident`, MTP on, `ZE_AFFINITY_MASK` unset, warm
  program cache `sycl-cache/m6c`) through the engine's own serve protocol, 256 greedy tokens at every length.
  `shipped` = `qsa_prompt_attn_batch` (the portable v1 kernel, `prefill.cpp:1621`); `batched` =
  `STRATA_PROMPT_ATTN_OLD=1` → `qsa_decode_attn_batch`, the engine's own batched decode attention. The runner sets
  and clears that switch itself and records which path ran (the shipped path prints the portable-v1 line at load,
  the batched path does not). Prompts are P2's own arms' files: 3,832 / 32,256 / 129,024 ids.
* **The oracle.** `p5/p5_oracle_launch.sh` + `p5_oracle_probe.py` reuse I2's harness: the same tree read-only, the
  same model file, the same two cards, **f16 KV**, fed the same prompt-id file, greedy for 256 tokens, with
  `n_probs 5` so the oracle's own top-5 log-probs come back at every position.
* **The reference at 128K is a different instantiation.** The oracle at `-ub 2048` **cannot run the 128K prompt**
  (§6); the 128K row's reference is the oracle at **`-ub 512`**, which can. How much that matters is measured, not
  assumed: on the same two prompts the `-ub 512` oracle differs from the `-ub 2048` oracle by one inserted token at
  32K/75 (0.0120 nats) and by a divergence at 4K/78 (0.1731 nats) — so the 128K row is a reference of the same
  kind, weaker by exactly that band.
* **The comparison.** `p5/p5_compare.py` reports per arm: the prefix view up to the first divergence (I2's), the
  aligned view (difflib — an insertion otherwise charges every later position), the nats margin at every
  difference, and the arm-vs-arm diff with no oracle in the loop. `p5/p5_table.py` assembles the table in §2.
* **The oracle's own resolution.** `p5/p5_oracle_ub512_chain.sh` runs the same prompts through the same oracle at
  `-ub 512` (I2's 5b method) — the measurement of the reference's own band, which is what a margin has to exceed
  to be evidence.
* **The kernels on their own axes.** `p5/p5_parity.sh`: `build-sycl/sycl_prompt_attn_parity` at ctx 3,832 and ctx
  32,768 on card 0 — both kernels' max error against an FP64 host reference and against each other, no model.

## 2. The three-way table

The int8 arms reproduce P2's recorded streams **byte for byte**: 4K `ad985c23…` (150 ids), 32K `124a3cd3…` (256),
128K `c60e72d3…` (256); the batched 32K arm is P2's old-attention stream `c52712f4…`. Speeds in §5.

| ctx | arm | prefill tok/s | decode tok/s | gen | first div vs oracle | margin there | decided by the oracle? | -ub 512 oracle emits ours? | prefix agreed | aligned matched |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | shipped (int8) | 213.6 | 21.3 | 150 | 78 | 0.1731 | no | YES | 78/79 (98.7%) | 110/150 |
| 4096 | batched (int8) | 281.3 | 21.7 | 137 | 78 | 0.1731 | no | YES | 78/79 (98.7%) | 109/148 |
| 4096 | shipped (fp16) | 178.0 | 21.5 | 148 | **none** | — | — | n/a | **148/148 (100%)** | 148/148 |
| 4096 | batched (fp16) | 280.5 | 21.3 | 137 | 78 | 0.1731 | no | YES | 78/79 (98.7%) | 109/148 |
| 32768 | shipped (int8) | 342.8 | 21.8 | 256 | 101 | 0.0436 | no | no | 101/102 (99.0%) | 162/256 |
| 32768 | batched (int8) | 765.2 | 22.3 | 256 | 75 | 0.0120 | no | YES | 75/76 (98.7%) | 162/256 |
| 32768 | shipped (fp16) | 275.2 | 21.6 | 256 | 75 | 0.0120 | no | YES | 75/76 (98.7%) | 162/256 |
| 32768 | batched (fp16) | 765.3 | 21.6 | 256 | 75 | 0.0120 | no | YES | 75/76 (98.7%) | 255/256 (99.6%) |
| 131072 | shipped (int8) | 411.5 | 19.8 | 256 | 132 | 0.0688 | no | n/a | 132/133 (99.2%) | 141/256 |
| 131072 | batched (int8) | 883.0 | 19.0 | 256 | 53 | 0.0771 | no | n/a | 53/54 (98.1%) | 76/256 |
| 131072 | shipped (fp16) | 332.3 | 19.9 | 256 | **none** | — | — | n/a | **256/256 (100%)** | 256/256 |
| 131072 | batched (fp16) | 877.1 | 18.3 | 256 | 59 | **0.1436** | **YES** | n/a | 59/60 (98.3%) | 78/256 |

Oracle margins over its own stream: 4K min 0.0482 / median 8.22 nats (3 of 148 below 0.15); 32K min 0.0120 /
median 9.48 (3 of 256); 128K min 0.0195 / median 3.35 (12 of 256) — 128K is a far more tie-dense regime, so a
0.14-nat margin there is a much more ordinary event than the same number at 4K.

"decided by the oracle" = the margin at the first divergence clears the oracle's own measured band (0.106 nats,
I2's `-ub` measurement) **and** the oracle's `-ub 512` instantiation does not itself emit our token there. Exactly
one arm-run in the card has one: the batched-fp16 arm at 128K/59 (0.1436 nats, odds 1.15:1). (The raw
`p5/p5_compare.py` line `ATTRIBUTABLE: ... OUTSIDE the oracle's own band` is the margin test alone; the table's
column is the margin test plus the `-ub 512` cross-check, which is why the 4K rows read `no` there.)

**The aligned column is not an accuracy column and must not be read as one.** After the first divergence the two
streams are in different contexts, so every later position is a different continuation; the batched-fp16 arm's
255/256 at 32K exists because it re-syncs after its one inserted token, and the batched arm's 30% at 128K is what
going off at index 53 costs in tokens — not 70% of positions being wrong. Only the first divergence is evidence
about a kernel.

## 3. Every divergence's margin, and what the oracle itself does at those positions

`p5/evidence/oracle-band.txt`. The oracle's own `-ub 512` instantiation, same prompts, same file, same cards:

* **4K index 78** (margin 0.1731): the `-ub 2048` oracle chooses `449, 39328, 7359`; the `-ub 512` oracle chooses
  **`264, 9640, 24277, 4621`** — the exact tokens the three diverging arms of ours emit there (shipped-int8,
  batched-int8, batched-fp16). The `-ub 512` oracle then runs 185 tokens against 148. The oracle's own choice at
  4K/78 is not stable under its own prompt-batch size, and one of its two answers is ours.
* **32K index 75** (margin 0.0120): the `-ub 512` oracle differs from the `-ub 2048` oracle by **one inserted
  token**, `3992` — the exact token the batched arms and shipped-fp16 emit there — and by nothing else (255/256).
* **32K index 101** (margin 0.0436): the one divergence both oracle instantiations disagree with (they both emit
  `16`, three of our four arms emit the `31995` cluster). `batched-fp16` matches both.
* **128K index 59** (margin 0.1436): the `-ub 512` oracle's own top-2 there are 332 over 9764 with 15% more mass
  on its pick; nothing in this card shows the oracle moving at that margin (the 128K `-ub 2048` instantiation
  cannot be run to check).

I2's `-ub` perturbation moved the oracle's own choices at margins of 0.064-0.106 nats; nothing our arms did at 4K or
32K exceeds 0.18 nats, and the single 0.1436 at 128K is the same order.

## 4. The mechanism — a near-tie resolution difference in the batched kernel's reduction path

* **Not a gross numerical error, and not the KV precision.** `p5/evidence/parity-p5.txt`, both kernels on the same
  synthetic fixture against an FP64 reference: ctx 3,832 — int8 old 2.15e-06 / new 1.90e-06, fp16 1.33e-06 /
  1.62e-06; ctx 32,768 — int8 2.13e-06 / 2.14e-06, fp16 1.89e-06 / 1.90e-06 (output scale 2.7-3.6); the two kernels
  are 1.2-1.3e-06 (int8) and 0.79-0.92e-06 (fp16) of **scale** apart. Both are FP32-accurate.
* **Not a structural difference in what the kernel attends to.** The cell selection, the KV pool, the page tables
  and the gather are shared code and bit-exact (`qsa_parity`: `kv_append`/`kv_gather` bit-exact at page sizes
  1/4/512, the spare slot's key bit-exact); the batched kernel is handed the same per-query selection
  (`m.sel_ids + t0 * m.cap`). Nothing here shows it reading a different set of cells.
* **What differs is the reduction and softmax path, and it is nameable.** The batched decode kernel
  (`build-sycl/sycl/qsa_decode_attn.cpp`) tiles the selection in `CHUNK = 64` cells with 256 threads, **one query
  per block** in `blockIdx.z`, scores by warp with 8 dimensions per lane, and accumulates **per-chunk partials**
  (`part_acc/part_m/part_l`) that a **second kernel** (`attn_merge_kernel`) re-normalises with another
  `expf_fast(m - M)` — three `expf_fast` sites, and `expf_fast` is `sycl::native::exp`
  (`include/strata/sycl_compat/intrinsics.hpp:300`), the hardware fast exponential. The v1 prompt kernel
  (`src/kernels/sycl/qsa_prompt_attn.cpp:121`) uses `CH` cells with 128 threads, 8 queries per warp row, an
  m16n8k16-emulated score path, and a single-kernel online softmax whose rescaling is `exp2f(m_old - m_new)` —
  the precise base-2 exponential (`:308`, `:316`). Same mathematics, a different exponential function, a different
  summation order, and one extra rounding stage (the merge kernel's re-normalisation) on the batched side.
* **The one reproducible kernel-only difference is at 4K index 78, and it is instructive**: `batched-fp16` is
  **byte-identical to `batched-int8`** (md5 `9428de79…` in both) — the batched kernel's choice there does not
  depend on KV precision — while the shipped kernel's does (int8 → the oracle's #2; fp16 → the oracle's #1, and the
  whole stream then matches the oracle 148/148). So at that position the batched kernel carries a difference from
  the shipped kernel that is *larger than the int8-KV perturbation* and in a stable direction, which is what a
  difference in the reduction/softmax path looks like at a 0.17-nat tie — not what random rounding looks like.
* **The lead for a correct-and-fast kernel** is therefore precise: keep the batched kernel's block shape (it is
  6.5x at the kernel: 78-80 ms against 497-510 ms for 2,048 queries at ctx 32,768) and give it the v1 kernel's
  arithmetic — `exp2f` rather than `sycl::native::exp`, one in-kernel online softmax rather than per-chunk
  partials re-normalised by a merge kernel, and the v1 cell/summation order. That is a re-tiling of the v1 body
  onto a 32-queries-per-block shape, which is what P2 §7 already named as the only path to closing the 4.3x.

## 5. Speeds, for the record

| ctx | arm | prefill tok/s | decode tok/s | wall s | qsa attn share of the chunk GPU timeline |
|---|---|---|---|---|---|
| 4096 | shipped int8 / batched int8 | 213.6 / 281.3 | 21.3 / 21.7 | 57 / 52 | (dequant-dominated at 4K) |
| 4096 | shipped fp16 / batched fp16 | 178.0 / 280.5 | 21.5 / 21.3 | 168 / 53 | — |
| 32768 | shipped int8 / batched int8 | 342.8 / 765.2 | 21.8 / 22.3 | 138 / 86 | 75.1% / 31.6% |
| 32768 | shipped fp16 / batched fp16 | 275.2 / 765.3 | 21.6 / 21.6 | 161 / 85 | — |
| 131072 | shipped int8 / batched int8 | 411.5 / 883.0 | 19.8 / 19.0 | 358 / 191 | 65.6% / 22.6% |
| 131072 | shipped fp16 / batched fp16 | 332.3 / 877.1 | 19.9 / 18.3 | 433 / 193 | — |

The batched path is worth **1.32x / 2.23x / 2.15x** prefill at 4K / 32K / 128K (it buys little at 4K, where the
prefill is expert-dequant-bound: 30.7% vs 5.8% of the 4K chunk timeline was `qsa attn`). Per-chunk `qsa attn` at
32K: 52,754 of 70,279 ms (75.1%) → 8,097 of 25,640 (31.6%); at 128K 185,511 of 282,596 (65.6%) → 28,445 of 126,093
(22.6%).

The fp16 4K shipped arm's 168 s wall is the one-time JIT of the fp16-KV kernel instantiation, measured not
inferred: `p5/p5_gaps.py` finds a **111.3 s** silence between two load lines there against **18.2 s** at the same
point in the int8 arm (18.7 s in the *next* fp16 arm, which found the instantiation already compiled). Steady
state is unaffected (178.0 against 213.6 tok/s).

## 6. The 128K oracle: a wall at -ub 2048, then the row itself

Two attempts at the reference configuration (`-c 131072 -ub 2048 -ncmoe 0`, the one that works at 4K and 32K) both
die at **n_tokens 49,152 (progress 0.38, t = 286.11 s)**: `level_zero backend failed with error: 20
(UR_RESULT_ERROR_DEVICE_LOST)`, once reported from `Error OP FLASH_ATTN_EXT`, once from `ggml_sycl_mul_mat_id`
(`ggml_sycl_error` ← `ggml_sycl_mul_mat_id` ← `ggml_backend_sycl_graph_compute`). Both runs show the prompt rate
collapsing with depth before the loss (484 tok/s at 2,048 tokens, 350 at 40,960, 174 at 49,152).

The wall is the **query-batch size times the depth**, not the context: at **`-ub 512`** the same prompt, the same
file, the same cards processed all 129,024 tokens and generated 256 (**prefill 126.9 tok/s, decode 8.9 tok/s, wall
1,045 s**) with no error. That is the 128K row above; it is a two-instantiation reference (§1) and its own
`-ub 2048` check is unavailable by construction. This is also the honest answer to "why is there no -ub 2048 128K
row": the reference implementation cannot produce one on this box at any `-ncmoe`.

## 7. What was NOT done, and what is not validated

* **Nothing is shipped and the default configuration is unchanged** — no source change; the batched path remains
  behind `STRATA_PROMPT_ATTN_OLD=1`. The rebuild at HEAD was a relink (P4's header change was not in the binary
  P2 measured; the arms reproduce P2's ids byte for byte, which is the check that the A/B is on the same code).
* **One sample per arm.** The only repeat is the 4K shipped-int8 arm (run twice: byte-identical ids, prefill
  212.7 / 213.6 tok/s). Everything else is a single run, and the differences are single near-tie events; a sign
  test on one event per length is not a statistical statement.
* **The 128K row's reference is `-ub 512`**, whose own band against `-ub 2048` is measured at 4K and 32K
  (0.0120-0.1731 nats) but is not measurable at 128K.
* **The kernel-level mechanism is bounded, not located.** The parity fixture shows both kernels equally accurate
  in aggregate; it does not exercise the per-position case that flips a 0.14-0.17-nat tie, so "a reduction path
  whose local error at a near-tie exceeds its aggregate error" is the reading the data supports, not a
  demonstrated defect. Locating it needs a per-position parity fixture built on the engine's own q rows and
  selection at the divergent position, which this card did not build.
* **The oracle's own band is measured at two instantiations and three prompts** (`-ub 2048` vs `512`). A wider band
  would need more perturbations (`-ncmoe`, different `-ts`), which were not run.
* **Decode is not what this card changes** and moves ±8% between arms (128K: 19.8 int8 vs 18.3 batched-fp16) on
  different window counts; no per-window comparison was made.
* Nothing is pushed: origin has no `sycl-xpu` branch.

## 8. Files

| what | path |
|---|---|
| the harness | `p5/p5_arm.sh`, `p5/p5_oracle_launch.sh`, `p5/p5_oracle_probe.py`, `p5/p5_oracle_stop.sh`, `p5/p5_compare.py`, `p5/p5_pairdiff.py`, `p5/p5_streams.py`, `p5/p5_oracle_ub_compare.py`, `p5/p5_table.py`, `p5/p5_gaps.py`, `p5/p5_meta.py` |
| the chains | `p5/p5_chain_engine.sh` (the 12 arms), `p5/p5_chain_oracle.sh`, `p5/p5_oracle_ub512_chain.sh`, `p5/p5_oracle_128k_ub512.sh`, `p5/p5_parity.sh`, `p5/p5_evidence.sh` |
| the evidence, regenerated by `p5/p5_evidence.sh` | `p5/evidence/{three-way-tables,oracle-band,arm-vs-arm,parity-p5,oracle-128k-wall}.txt` + the four chain logs |
| the assembled table | `p5/TABLE.md` (from `p5/p5_table.py`), `p5/compare-{4096,32768,131072}.json` |
| every arm's raw output | `~/strata-xpu/p5/runs/p5-<ctx>-<arm>[-fp16]/{out,err,tokens,log,timeline}.txt` (12 arms) |
| the oracle's streams | `~/strata-xpu/p5/oracle/oracle-{4k,32k}.json`, `oracle-{4k,32k}-ub512.json`, `oracle-128k-ub512.json`, and the server logs of the two failures |
| the model's own context metadata | `p5/p5_meta.py` (`qwen4exp.context_length 262144`; 12 of 48 layers are full attention; `head_count_kv 2`, key/value length 256) |
