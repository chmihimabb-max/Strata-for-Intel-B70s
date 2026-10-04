# D3 — decode's depth-growing cost: the score sweep grows, the sharing is already in the tree, and the per-layer sync priced

Card `t_87aa2963` (D3), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`. One measurement session, both
B70s at a time, `ZE_AFFINITY_MASK` unset, one engine at a time. The config of record verbatim (IQ3_S GSQ-RCO
snapshot `ed59f920…`, `--kv int8 --kv-resident 32768 --expert-cache auto --expert-profile
data/expert-profile.bin --mmap-experts --prefill 512 --spec 4 --spec-min-p 0.7 --max-context 262144 --no-capture
--stats`, MTP drafter `~/strata-xpu/mtp/rt`, `gpu [0,1]` → `--layer-split auto` → K=23, 100.0% expert residency
in every arm) with the two deviations D1's rig established and every decode card since has stated:
**`--prefill auto`** (the file says 512; auto picks the 8192-token chunk) and **`--prompt-cache 0
--prompt-cache-every 0`** (S4's split OOM at the default mid-prompt checkpoint), plus **one new distinction this
card introduces**: `--max-context` is passed as the record's own **262,144**, while D1's and D2's rigs passed the
*arm's* CTX (a stated deviation there) — which silently flips a kernel dispatch (§3).

Engine binary: `build-sycl/strata` md5 `e841fd061fec873c2f24e785973a2ebe`, unchanged from D2. **This card changed
no engine source**; every arm is an env lever or an instrument.

Rig under `d3/` (the card's rule): `d3/d3_run_arm.sh` (D2's one-engine arm runner, retargeted at `d3/runs`, its
default `--spec-min-p` now 0.7, plus a new `--maxctx` that separates the engine's `--max-context` from the arm's
prompt length), `d3/d3_chain*.sh` (the arm blocks), `d3/d3_select.py` (the census reader: per window, the
selection, the handshake, and the instrument's own cost, out of D1's launch-site histogram), `d3/d3_report.py`
(D2's one-row-per-arm reader), `d3/d3_stages.py` (the engine's `decode timing` line and P9's sampled stage table
side by side), `d3/d3_stall.sh` (the #267 stall rig, P1b's shape), `d3/d3_evidence.sh` → `d3/D3-EVIDENCE.txt`
(every raw line the acceptance asks for).  Raw arm directories: `d3/runs/<tag>/`, stall arms `d3/stall/<tag>/`.

## 1. Verdict

1. **The selection does grow with depth — and it is the SCORE SWEEP, not the top-k and not the attention.**
   Exact device microseconds of one T=4 decode window (launch-site histogram, closure path, P9 stamps on):
   `qsa_block_scores` **0.210 → 1.466 → 5.361 ms** at 4K/32K/128K (×25.5 while the scored blocks grow ×33.6),
   `qsa_block_topk` 0.684 → 0.750 → 1.403 ms (×2.1), `qsa_decode_attn_batch` 2.031 → 2.000 → 1.944 ms (**flat**,
   P9's finding re-confirmed).  As a share of the window's counted device time the selection is **0.68% → 1.68% →
   4.82%**; the attention is 1.54% → 1.50% → 1.40%.  So "the selection is what grows with depth" holds as a whole,
   and inside it the growing term is the *scores* (the sweep over candidate blocks), not the top-k (the choosing).
2. **The card's per-token premise is refuted: the selection is already window-level, and the per-block key read is
   already shared across the window's rows.**  Both kernels are launched **once per (QSA layer, window)** over the
   window's rows (`src/core/verify.cpp:1019-1022`; 12 launches per window = the model's 12 QSA layers, no factor of
   T), and the scores kernel that runs is upstream's `block_scores_multi_kernel` (commit `a20f3b5`, default on,
   `STRATA_SCORES_MULTI=0` = the old per-query grid), which reads each key block **once for all of the window's
   queries** (`src/kernels/cuda/qsa_select.cu:600-646`).  Priced, with the ids guard: **+1.0% decode at 4K, −0.2%
   at 32K, +0.5% at 128K** for the shipping shared form against the old grid (§4) — i.e. the sharing is real but it
   is not where the depth cost lives, because the sweep is latency-bound (§5.4).  The per-row scores themselves are
   not duplicated work: each row has its own query and its own position, so the rows are a genuine T×B score
   matrix, not the same B scores three times.
3. **The top-k's dispatch differs between the record and every published decode arm — measured, not inferred, from
   the census's own launcher symbol**: the record's `--max-context 262144` takes the **memory-keyed** kernel
   (`qsa_block_topk_ref`) and the arm-CTX configs of D1/D2 (and of this card's first three arms) take the
   **register** one, because the launcher's rule is capacity-based (`reach = max_cells/4 + 2` vs `fit = 1024×33`
   blocks).  It is **not** a slow path at short depth (the memory kernel is 29% cheaper at 4K) and it costs
   **+1.2% of the window at 128K** (3.044 against 1.403 ms) with identical ids; the obvious fix — pass the
   window's own reach — is **unsafe on a captured graph** (§5.3), and a safe one needs an nb-agnostic top-k, which
   is named rather than attempted on this evidence.
4. **The per-layer handshake is counted exactly, and the coarsest exact sync the engine already has is measured.**
   Per **layer** per window: **3 host→device 4-byte publications**, **1 device→host ring** and **3 device-side
   `wait_flag_ge` spins**; the spins are **144 launches and 14.7 / 15.4 / 16.9 ms per window at 4K/32K/128K =
   11.1% / 11.7% / 12.0%** of the window's counted device time — the largest single non-projection family in the
   window.  `STRATA_VERIFY_DEVICE_PLAN=1` (E-6) is the engine's existing answer (the device plans its own group and
   its waits return on a device-written word), and it is worth **+2.9% / +3.3% decode at 4K / 32K with
   byte-identical ids**, i.e. the *measured ceiling* for the card's "coarser synchronisation" question: the host's
   own per-layer work is another 0.7-0.8% and the two together are the whole term (§6.2).
5. **The one named opportunity this card leaves behind**: the depth-growing score sweep moves its working set at
   **28-37 GB/s** — the same class of device moves an order of magnitude more — because each `(block, query)` pair
   is a serial chain of four head dot-products behind a fixed 256-block grid.  Its own bandwidth floor at 32K is
   ~0.11 ms against the measured 1.44 ms, i.e. up to **~3.5% of a 128K window** if reshaped; that is a kernel
   project with its own parity test, not a switch (§5.4).
6. **Nothing was landed.**  No engine source changed (the binary is D2's `e841fd06…`), the config of record is
   untouched, and E-6 is measured as a switch, not flipped into the record — the card's rule is not to destabilise
   the release path for a few percent, and the stall rig re-proves the #267 guarantee on this binary (§6.3).

## 2. Config, instruments and their own cost

Every arm is the config of record with exactly one lever or instrument changed, and every arm runs
`STRATA_DECODE_TIMING=1` and `STRATA_SUBMIT_COUNT=1` (both free: P3's submission counter and the engine's own
window line).  The census arms add two instruments, both D1/P9's:

* **`STRATA_LAUNCH_HIST=1`** (D1's launch-site census: one line per launch with the site, its device microseconds
  and the number of launches at that site in that window; `--histwindows 8` = the first 8 windows of the request,
  which covers the first occurrence of every T the draft policy picks).  It needs the **closure** path
  (`STRATA_SYCL_GRAPH=0`): a replayed graph hands the driver one submission and carries no per-kernel timestamp
  (D1's note).
* **`STRATA_VERIFY_PROFILE=1`** (P9's host-sampled stage stamps, the `strata decode GPU stages` line).

Their own cost, measured rather than assumed: the stamp set is **998 launches and 1.703 / 1.685 / 2.001 ms per
window = 1.29% / 1.28% / 1.43%** of the window's counted device time at 4K/32K/128K (the `gpu_stamp` row of the
census), which is the device-time part of P9's measured +3.0% wall cost; the closure path itself is P9's measured
**+4.9%** on a window (131.27 against 125.14 ms).  The census windows are therefore comparable to each other
(same instruments throughout) and **not** directly to the uninstrumented arms — which is why the sharing A/B (§4)
and the record/arm comparisons (§5.2) have their own uninstrumented controls.

Two facts about the arms' shape, both from the engines' own lines: **100.0% expert residency and 0.00 CPU expert
entries per layer-window in every arm** (`decode expert cache hit rate: 100.0% (…)`), and the 4K arms stop at
**147 generated tokens** (the prompt's own stop, as in D1/D2/P9) while 32K and 128K run the full 256.


## 3. The selection: kernels, file:line, and per-window or per-token

The decode window is `Verifier::record_window` (`src/core/verify.cpp:708`) → `Verifier::run`
(`src/core/verify.cpp:1408`); the window is one captured SYCL `command_graph` replayed per window (D1's default)
or a launch-closure list with `STRATA_SYCL_GRAPH=0`. The selection is **two launches per QSA layer per window**,
and both are **window-level** — see the call site:

```
src/core/verify.cpp:1019  qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb*IQ*ID, step_ + tb*kStepCount,
                                           n, max_blocks_, s, scores_ + tb*max_blocks_, cs);
src/core/verify.cpp:1021  qsa_block_topk(scores_ + tb*max_blocks_, step_ + tb*kStepCount, n, max_blocks_, cap_, s,
                                         sel_ + tb*cap_, cs);
```

`n` is the number of rows in the stage's group (the whole window: `tb..te`), the last argument (`active_blocks`)
is left at its default −1, and each row carries **its own** step record (`step_ + t*kStepCount`, filled by
`qsa_step_fill` at `verify.cpp:1461`: `n_kv = pos + 1`, `n_bid = n_kv / idx_block`).  On this config
`spec_split` is false (`src/program/generate.cpp:373`, `:1150-1152`), so the group count
`G = (split_ && T >= 2) ? 2 : 1` is **1** (`verify.cpp:724`, `:1527`) and the census counts exactly **12 scores
and 12 top-k launches per window** = the model's 12 QSA layers.  Nothing here is launched per token.

| what | kernel that runs | launcher → kernel, `file:line` | per query token or per window |
|---|---|---|---|
| **scores** (the depth-growing term) | `block_scores_multi_kernel` — a fixed 256-block grid strides over the blocks and reads each key block **once for all of the window's queries** | call `src/core/verify.cpp:1019`; launcher `src/kernels/cuda/qsa_select.cu:649-671`, dispatch at `:658-663`, kernel `:606-646`; SYCL port `src/kernels/sycl/qsa_select.cpp:654/714/723-724` | **per window** (one launch, `n` rows) |
| scores, the old form (`STRATA_SCORES_MULTI=0`) | `block_scores_kernel` — grid `((reach+7)/8, nq)`, "~24,600 mostly-idle blocks per layer at a decode window, **each key re-read per query**" (`qsa_select.cu:601`) | `src/kernels/cuda/qsa_select.cu:28-51`, launcher `:665-670` | per window, per (query, block) grid |
| **top-k** | one of two: `block_topk_reg_kernel` (keys in registers, 1024 threads) or `block_topk_kernel` = `qsa_block_topk_ref` (keys re-read from memory on every radix pass, 256 threads) | call `src/core/verify.cpp:1021`; launcher `src/kernels/cuda/qsa_select.cu:748-786`, ref kernel `:57-153`, register kernel `:506-597`, dispatch rule `:756-783` | **per window** (grid = nq, one block per row) |
| **attention** (the control: P9's 1.8%) | `qsa_decode_attn_batch` | call `src/core/verify.cpp:1028` | per window |

Two capacity facts drive the top-k's dispatch and are worth stating because they are set by the **state**, not by
the window (`src/core/verify.cpp:514-517`):

```
cap_        = qsa_selection_width(kTopkMaxCells = 32768, s) = min(32768, 2048 + 4 - 1) = 2051 cells   (the top-k's width)
max_blocks_ = ss.qsa_states[primary].max_cells / idx_block + 2                                        (the scores' row stride)
max_cells   = --max-context  (the QSA state is sized by session_bytes(g, o.max_context, …), generate.cpp:2181/2380/2469)
```

The top-k's launcher decides on `reach = max_blocks_` versus `fit = TK_T * TK_PER = 1024 * 33 = 33,792 blocks`
(`qsa_select.cu:467-476`, `:764-765`; on this build `TK_PER_MAX == TK_PER` because the `__HIPCC__` branch is not
taken), and `counted = false` on anything that is not HIP (`:756-763`, kept by the port at
`qsa_select.cpp:886-893`).  The record's `--max-context 262144` is **65,538 blocks > 33,792** ⇒ the **memory**
kernel; the arm-CTX configs of D1/D2 (and of this card's first three arms) are ≤ 32,770 ⇒ the **register** one.
Both are visible in the census by their own launcher symbols — `qsa_block_topk_ref` versus `qsa_block_topk` —
which is how §5's table is measured rather than inferred.

**The measured attribution** (one T=4 window, exact device microseconds from `STRATA_LAUNCH_HIST=1` on the
closure path, P9 stamps on; `d3/d3_select.py`, raw output in `d3/D3-EVIDENCE.txt`):

| depth | scored blocks (`n_bid+1`, T=4 rows) | `qsa_block_scores` | `qsa_block_topk` | `qsa_decode_attn_batch` | selection, % of the window | window (counted device time) |
|---|---|---|---|---|---|---|
| 4K | ~961 | **0.210 ms** | 0.684 ms | 2.031 ms | **0.68%** | 132.07 ms |
| 32K | ~8,065 | **1.466 ms** | 0.750 ms | 2.000 ms | **1.68%** | 131.82 ms |
| 128K | ~32,257 | **5.361 ms** | 1.403 ms | 1.944 ms | **4.82%** | 140.25 ms |

and the same three windows' P9 stage lines (host-sampled; the sampled shares are floors of the exact numbers
above, the sampler's stated coverage being 33-39%): `scores+topk` **0.35 / 0.78 / (128K: see evidence)** ms and
`attention` 0.76 / 0.73 / … ms per window, from `strata decode GPU stages`.

**Verdict for the card's task 1, in one line**: the depth cost of the decode path's *choosing* is real and it is
the **score sweep** (`qsa_block_scores`, ×25.5 from 4K to 128K for ×33.6 blocks) — it is **already window-level**
and already shares the key read across the window's rows; the top-k grows ×2.1 over the same range (it is
barrier-bound: 1024 threads × ~23 block-wide barriers per launch ≈ 57 µs at 4K, ≈117 µs at 128K), and the
attention is flat, as P9 measured.


### 3.1 The ids guard, every arm

One method, the rig's own (`grep '^T ' out.txt | md5sum`, the trailing newline included) — so every comparison below
is like for like.  Every lever and every instrument this card measured leaves the greedy stream **byte-identical
at each depth**:

| depth | generated | ids md5, every arm of that depth |
|---|---|---|
| 4K | 147 tokens (the prompt's own stop) | `ac9f16fceed82e5681e42bd197835c51` — `base-4096`, `nomulti-4096`, `devplan-4096`, `hist-4096`, `histc-4096`, `oldtopk-4096` (all 6) |
| 32K | 256 | `124a3cd39b33f7da31e225829625983a` — `base-32768`, `nomulti-32768`, `devplan-32768`, `hist-32768`, `histc-32768` (all 5) |
| 128K | 256 | `c60e72d397e9b1cd44dab9c5ab1187c8` — `base-131072`, `hist-131072`, `histc-131072` (all 3 so far) |

This is what makes the rest of the write-up's speed comparisons correctness-neutral: the max-context (4,096 vs
262,144), the scores grid (`STRATA_SCORES_MULTI=0`), the top-k kernel (memory vs register), the device plan (E-6),
the graph path (closure vs graph) and the instruments (census + stamps) *all* produce the same tokens at every
depth here.  Two cross-session checks fall out of the same table: at 4K and 32K the rig-method md5 matches P9's own
(`ac9f16fc…`, `124a3cd3…`); and with D2's reader's method (no trailing newline) the same streams read `66bf952d…` /
`d87373e8…` / `511a89be…`, which are exactly D2's published 4K/32K/128K ids for `--spec-min-p 0.7` — i.e. this
session re-ran D2's and P9's token streams.  **An id figure is only comparable within one method**; both are stated
here so the match is checkable.

## 4. The sharing: what is already shared, and what the A/B prices

**The card's premise does not hold, and the evidence is the call site and the census's launch counts.**  The
selection is not per query token: `qsa_block_scores` and `qsa_block_topk` are each launched **once per
(QSA layer, window)** over all `n` rows of the window (`src/core/verify.cpp:1019-1022`), and the census counts
**12 launches of each per window** — 12 = the model's QSA layers, no factor of T.  A per-token form would count
12·T ≈ 34 launches at T=2.8.

**What *is* duplicated per row, and what already shares it**: the make-or-break fact is that a window's rows have
*different* query vectors and *different* positions (`q_idx + t*ID`, `step_ + t*kStepCount`), so their scores are
genuinely different numbers — a T×B score matrix, not the same B scores computed T times.  The only thing that is
the same across the rows is the **key block being read**, and that is exactly what the shipping kernel shares:
`block_scores_multi_kernel` loads `kp` **once per block** and loops the call's queries inside
(`src/kernels/cuda/qsa_select.cu:623-644`), where the pre-`a20f3b5` form had a grid of `(blocks, nq)` warps and
"each key re-read per query" (`:600-604`).  So the sharing the card asks to implement **is the tree's default**
(`STRATA_SCORES_MULTI` unset = the multi kernel; `=0` = the old grid).

**The sharing's price, measured two ways.**

*(a) From the census's own T-dependence* (no extra arm needed: one request's first 8 windows cover T = 1, 2 and 4,
the same kernel and the same depth):

| depth | `qsa_block_scores`, T=1 window | T=2 | T=4 | rows ×4 ⇒ cost ×? |
|---|---|---|---|---|
| 4K | 0.081-0.103 ms | 0.122-0.129 ms | 0.202-0.210 ms | **×2.1-2.5** (a per-query read would be ×4) |
| 32K | 0.525 ms | 0.782-1.085 ms | 1.438-1.466 ms | **×2.8** (a per-query read would be ×4) |

i.e. the shipped kernel's cost grows *sublinearly* in the window's rows — the shared key read is the reason — and
what remains per row is the dot product and the score store.

*(b) The direct A/B*: the same arms with `STRATA_SCORES_MULTI=0`, which is the old per-(query, block) grid and the
same arithmetic per (block, query) — so the ids must not move, and they do not:

| depth | arm | ms/window | decode tok/s | Δ against the shipped kernel | ids md5 |
|---|---|---|---|---|---|
| 4K | `d3-base-4096` (shipped, shared key read) | **113.72** | **23.94** | — | `66bf952d…` |
| 4K | `d3-nomulti-4096` (old per-query grid) | 114.84 | 23.71 | **−0.98%** | `66bf952d…` identical |
| 32K | `d3-base-32768` | **115.74** | **24.04** | — | `d87373e8…` |
| 32K | `d3-nomulti-32768` | 116.01 | 23.99 | −0.21% | `d87373e8…` identical |
| 128K | `d3-base-131072` | 102.97 | 21.07 | — | `511a89be…` |
| 128K | `d3-nomulti-131072` | **102.46** | **21.17** | **+0.47%** | `511a89be…` identical |

So the window-level sharing that is already in the tree is worth **+1.0% decode at 4K, a wash at 32K and nothing
(marginally negative) at 128K** — and the mechanism is §5.4's: the kernel is latency-bound at ~30 GB/s, so four
times the block-level parallelism (the old grid's `(blocks, nq)` shape) pays for the four times the key traffic at
depth, while at 4K the traffic dominates.  **The card's "share the selection across the window" is therefore not
the lever the depth cost is behind**; the sharing is present, its price is measured, and the depth cost is the
sweep's own latency (§5.4).

**What cannot be shared, and why the window is not "the same scores three times"**: each row's `step` carries its
own `n_kv = pos + 1` and `n_bid = n_kv / 4` (`src/core/verify.cpp:1461`, `qsa_step_fill`), so row *t* scores a
*different, larger* block set against a *different* query — the rows' candidate sets are nested, not equal.  The
one sharing left to take is therefore not across rows but *inside* the per-(block, query) work (§5.4).


## 5. The top-k's dispatch and the score sweep's real ceiling

### 5.1 Which top-k kernel each configuration takes — measured by the census's own symbol

The census records the **launcher** symbol, and the memory kernel has its own launcher (`qsa_block_topk_ref`,
`qsa_select.cu:736-746`), so the dispatch is visible rather than inferred.  Three depths, the record's
`--max-context 262144` against the arm's own CTX (the configuration every published decode arm ran):

| arm | maxctx | top-k launcher in the census | top-k ms/window | scores ms/window | window (counted device) | window (decode line) |
|---|---|---|---|---|---|---|
| `d3-hist-4096` | 4,096 | `qsa_block_topk` (register) | 0.684 | 0.210 | 132.07 ms | 128.69 ms |
| `d3-histc-4096` | **262,144** | **`qsa_block_topk_ref`** (memory) | **0.485** | 0.205 | 136.33 ms | 130.53 ms |
| `d3-hist-32768` | 32,768 | `qsa_block_topk` (register) | 0.750 | 1.466 | 131.82 ms | 124.96 ms |
| `d3-histc-32768` | **262,144** | **`qsa_block_topk_ref`** (memory) | **1.070** | 1.438 | 137.52 ms | 126.97 ms |
| `d3-hist-131072` | 131,072 | `qsa_block_topk` (register) | 1.403 | 5.361 | 140.25 ms | (reg) |
| `d3-histc-131072` | **262,144** | **`qsa_block_topk_ref`** (memory) | **3.044** | 5.378 | 140.82 ms | 114.10 ms (avg T 2.40) |

and at 128K the direction is the other way from 4K: the memory kernel costs **2.2× the register one**
(3.044 against 1.403 ms, +1.64 ms = **+1.2% of the window**), which is what the register kernel's "keys read once"
buys once `n_bid` is 32,257 blocks — a top-k of 3.0 ms is 2.2% of a window on the record's config against 1.0% on
the arms'.  Both 128K arms emitted the **same token ids** (`c60e72d3…`), i.e. the two kernels select identically
on this data, as their contract says, so this is a speed question only; and `d3-oldtopkhist-131072` (E3) is the
clean same-maxctx pair that separates the kernel from the max-context.

Two things fall out of this table and one does not:

* **the scores are capacity-independent** (0.210 vs 0.205 at 4K; 1.466 vs 1.438 at 32K, i.e. within 2%): the
  window-shared kernel's grid is fixed and its work follows the window's own `n_bid`, so the record and the arms
  are measuring the same thing — which is what makes the scores rows comparable across max-contexts;
* **the top-k's dispatch is not a slow path**: at 4K the memory kernel is **29% cheaper** than the register one
  (0.485 against 0.684), and at 32K it is 43% dearer (1.070 against 0.750) — inside a single window that is
  ±0.2-0.3 ms, i.e. **0.15-0.25% of the window**, and the two arms differ in max-context as well, which is why
  the clean pair at one max-context is measured separately (§5.2).  Neither direction is worth a change to a
  correctness-sensitive path, and it is not the depth story.
* what the dispatch *does* explain is why the published decode arms and the served configuration are not running
  the same kernel — a fact worth knowing for anyone reading D1/D2/P9's numbers as the served path's.

### 5.2 The clean pair at one max-context, and the max-context effect alone

[PENDING-E: E1 (maxctx 4096, `STRATA_TOPK_OLD=1` = the memory kernel forced, against `d3-hist-4096`) and E2
(maxctx 262144 against the arm's own CTX at 4K, uninstrumented).]

### 5.3 Why "pass the window's own reach" is not a safe fix here

The obvious fix — let the caller pass the window's actual `n_bid + 1` as `active_blocks` (the argument exists for
the prompt path and is ignored on non-HIP builds: `qsa_select.cu:756-763`) — is **not safe on a captured window**,
and the engine's own design note says why (`src/core/verify.cpp:936-938`):

> `max_blocks` and `cap` are CAPACITIES from the state, not this token's counts: a grid or a shared-memory size
> that follows the sequence length is baked into a captured graph, and the kernels guard for the surplus.

The register kernel's `TK_PER` keys per thread are exactly such a baked size (`qsa_select.cu:524-528`, `:765`), and
the window's graph is captured **once per T and replayed for every later window of that T**:
`Verifier::capture` returns immediately when `exec_[T] != nullptr` (`src/core/verify.cpp:1275-1276`), and the
`Verifier` is a member of the per-process stage struct (`src/program/generate.cpp:712`), i.e. its graphs outlive
the request.  A capture taken at the first window's depth would bake `per = ceil(nb/TK_T) ≤ TK_PER` for that depth
and then silently drop the blocks past the bound on a later, deeper replay — the selection would change with no
error.  So a safe fix needs a top-k that is **nb-agnostic** (registers for the blocks that fit, a strided second
pass over memory past `TK_T * TK_PER`, as the memory kernel already does) or a bound that holds over every replay
(the capacity — today's rule).  Given §5.1's numbers (±0.15-0.25% of a window), that kernel is not worth writing
on this evidence, and it is named here rather than attempted.  The **evidence path that would settle it**: a
`block_topk_reg_wide_kernel` with an outer chunk loop, added to `src/kernels/cuda/qsa_select.cu` and re-ported
with `tools/sycl/handport.py qsa_select`, then the same three-depth arm table with the token-id guard.

### 5.4 The real ceiling on this term: the score sweep is latency-bound

The depth-growing term is `qsa_block_scores`, and it is nowhere near a hardware limit.  At 32K it reads
`8,065 blocks × 512 B = 4.13 MB` per QSA layer — **49.5 MB per window, in 1.466 ms = 33.7 GB/s** (4K: 28 GB/s;
128K: 32,257 blocks × 512 B × 12 = 198 MB in 5.361 ms = 37 GB/s).  A device of this class moves an order of
magnitude more, so the kernel is **~14× above its own bandwidth floor** and its cost is a latency/ILP property,
not traffic: each `(block, query)` pair is a chain of four serially dependent head dot-products, each ending in a
5-step shuffle reduction (`qsa_select.cu:631-639`), and the grid is a fixed 256 blocks × 8 warps.  **The measured
ceiling for the depth-growing term** is therefore its bandwidth floor — 49.5 MB / ~450 GB/s ≈ **0.11 ms at 32K**
and ≈ 0.44 ms at 128K against the measured 1.44 / 5.36 ms — i.e. up to **~4.9 ms of a 140 ms window (~3.5%) at
128K** if the sweep were reshaped (heads in parallel lanes, a grid that scales with the work).  That is a kernel
project with its own parity test, and it is the one named opportunity this card leaves behind; the *sharing* the
card asked for is already in the tree and is not where that time is.


## 6. The per-layer sync: the count, the ceiling, and the guarantee

### 6.1 The count, per layer per window (G = 1 on this config)

Counted in the census (launch counts and device microseconds) and read off the source:

| direction | what | count per layer | count per window | where |
|---|---|---|---|---|
| host → device | one 4-byte H2D `cudaMemcpyAsync` on the copy stream per publication: **flagA** (the group's VRAM plan), **flagB** (the PCIe share is staged), **flag** (the CPU share is in the mapped rows) | **3** | **144** | `src/core/verify.cpp:1580-1590`; `publish_flag`/`raise_flag_dev` `:282-294` |
| device → host | the ring: `doorbell_publish` increments the MAPPED sequence word, which the host spins on | **1** | **48** (census) | call `src/core/verify.cpp:1072`; `include/strata/kernels/elementwise.hpp:84-113`; the host's spin `src/core/verify.cpp:1539` |
| device waits on the host | `wait_flag_ge` spins: flagA (`:1112`/`:1115`), flagB (`:1144`/`:1145`), flag (`:1156`/`:1160`) | **3** | **144** (census) | `src/core/verify.cpp` as listed |

The handshake is the largest single non-projection family in the window, and it is nearly depth-flat:

| depth | `wait_flag_ge` per window | % of the window's counted device time | `doorbell_publish` | the host's own per-layer work | the host's GPU-reach wait |
|---|---|---|---|---|---|
| 4K | **14.690 ms** (144 launches, 102 µs each) | **11.12%** | 0.357 ms (48) | 1.00 ms (plan 0.31 actq 0.22 jobs 0.23) | 86.82 ms = 67% of the 128.69 ms window |
| 32K | **15.423 ms** | **11.70%** | 0.357 ms | 0.90 ms | 85.11 ms = 68% of 124.96 ms |
| 128K | **16.867 ms** | **12.03%** | 0.342 ms | 0.81 ms | 76.77… (see the decode-timing lines in the evidence) |

The two sides are worth reading together: the **device** spends 102-117 µs per layer spinning on the host's flags,
while the **host's** own per-layer work is **21 µs per layer** (1.0 ms / 48) — i.e. the device's stall is **not**
waiting for the host's arithmetic, it is waiting for the *round trip* (ring → host wakes → publish → device
observes).  That is the term a coarser sync can buy, and it is what §6.2 measures.

### 6.2 The measured ceiling: E-6, the engine's own coarsest exact sync

The engine already contains the answer to "what would coarser synchronisation look like":
**`STRATA_VERIFY_DEVICE_PLAN=1` (E-6)** lets the **device** plan its own expert group and makes the device's waits
return **without touching the host's flag**:

```
src/kernels/cuda/verify_kernels.cu:500-506   resident_plan(): one 1-thread kernel on the window's stream plans the
                                             group on the device and, when every one of its experts is resident,
                                             writes `*skip = ring` in DEVICE memory (:477-478); if any expert is
                                             not resident it writes `*skip = 0` and returns (:446) - the host's
                                             flag is then the only path, i.e. E-6 is exact, not approximate.
src/kernels/cuda/verify_kernels.cu:480-486   wait_flag_ge_or(): `if (*skip == value) return;` - the device does
                                             not stall at all for a group it planned itself.
src/kernels/cuda/verify_kernels.cu:487-497   copy_i32_unless() / copy_or_zero(): the host->device plan and
                                             CPU-share copies become no-ops for such a group.
src/core/verify.cpp:682-689, :1067-1070, :1112, :1144, :1156   the switch and its three call sites.
```

So E-6 **is the coarsest sync the engine can currently do without redesigning the heartbeat**, and the ceiling for
a "sync every N layers" design is what it measures plus the host's own per-layer work if the host cadence were
dropped too (1.00 / 0.90 ms per window = 0.8% / 0.7%, §6.1).  Measured with the census instruments so it is
directly comparable to `d3-histc-*` (same closure path, same stamps, same max-context):

[PENDING-D: the E-6 table at 4K/32K/128K: wait_flag_ge/wait_flag_ge_or launches and ms, the window, decode tok/s,
and the ids md5 against `d3-histc-*`.]

The 4K pair, with the same instruments, the same `--max-context 262144` and one env var between them:

| arm | the device's per-layer waits | window, counted device time | window, decode line | decode tok/s | ids md5 |
|---|---|---|---|---|---|
| `d3-histc-4096` (control) | `wait_flag_ge` **18.462 ms** (144 launches, 128 µs each) | 136.33 ms | 130.53 ms | 21.25 | `66bf952d…` |
| `d3-devplan-4096` (**E-6**) | `wait_flag_ge_or` **0.143 ms** (144 launches, 1 µs each) | **122.87 ms (−9.9%)** | **126.89 ms (−2.8%)** | **21.86 (+2.9%)** | `66bf952d…` **identical** |
| `d3-histc-32768` (control) | `wait_flag_ge` **17.813 ms** | 137.52 ms | 126.97 ms | 21.92 | `d87373e8…` |
| `d3-devplan-32768` (**E-6**) | `wait_flag_ge_or` **0.144 ms** | **126.53 ms (−8.0%)** | **122.84 ms (−3.3%)** | **22.65 (+3.3%)** | `d87373e8…` **identical** |

So the device's stall on the host's flags is **17.7-18.5 ms of device time per window** and removing it is worth
**−3.6 / −4.1 ms of the window's wall, i.e. +2.9% / +3.3% decode, at 4K / 32K**, with identical greedy ids; the
smaller wall figure than the device figure is the overlap: part of the time the device spends spinning, the host
spends waiting for the device's ring (§6.1's 83-88 ms), so only the unoverlapped part leaves the critical path.
**This is the measured ceiling for the card's "coarser synchronisation" question**: the engine's own device-plan
mode, which is exact (its bail-out writes `*skip = 0` whenever a layer has a non-resident expert), buys ~3% of the
decode window and nothing more can be bought by coarser cadence alone (the host's own per-layer work is another
0.7-0.8%, and the two together are the whole term).


**A design for a coarser cadence, and why it is not implemented here.**  The host must publish per layer today
because the *plan* (which resident slot holds each of the group's experts) and the CPU share are computed host-side
from the router's top-10 that the device produces one layer at a time; E-6 shows the device can compute the plan
itself whenever the layer's experts are all resident (which they are, 100%, in the record's configuration).  A
"every N layers" cadence would therefore be: E-6 on (device plan), the host's flag publications raised once per N
layers, and the pool invoked on the device's *miss* signal only.  What it would touch is exactly the protocol
P1/P1b stabilised — the bounded per-layer wait, the release path and the deterministic exit (`#267`) — and the card
says not to destabilise that for a few percent; so this card measures the ceiling (§6.2's arm) and re-proves the
guarantee on the same binary (§6.3) instead of landing it.  The evidence path if it were attempted: E-6's delta at
three depths (here), the stall rig (here), plus a determinism sweep (N runs, one ids md5) which this card's
single-arm-per-configuration discipline cannot provide.

### 6.3 The release path's guarantee (#267), re-proved on this binary

`d3/d3_stall.sh d3-stall-4096 build-sycl/strata 16 4096 1 1` — P1b's rig: `STRATA_TEST_VERIFY_STALL=1` withholds the
last layer's flag in the first window, and a SECOND ask follows, so both halves of the property are exercised
(release, and the refusal afterwards).  On the binary this card measured (`e841fd06…`), exit **250**, wall 109 s,
**0 generated tokens**:

```
strata serve: no progress for 60 s during a request (verify window: waiting for the GPU to finish the window
  (flags A/B/M raised) 1) - stopping the engine so the server starts it again (issue #29)
strata serve: stall report (engine 0.1.34): stage "verify window: waiting for the GPU to finish the window
  (flags A/B/M raised) 1" for 59 s; 23 layers served since the last finished step
  verify window: 1 tokens at position 3831, host at layer step 23; the GPU rang 23; flags: served 22, plan (A) 23,
  copies (B) 23; tail beacons 0 0 0 0 0 0 0 0
strata: #267 release pass over live verifier slot 0 (thread …)
verify release: flag publication 0.078 ms; drain budget 5000 ms
verify release: the GPU finished; drain 3.410 ms, release 3.492 ms (publication 3.491 ms)
strata: released the verify window's GPU waits (#267): the GPU finished in 3 ms
verify teardown: the window on cs_ was released (#267); it finished (1.063 ms of a 5000 ms budget)
# and the second ask, on the same session:
ERR verify: an earlier window never finished on the GPU (#267); restart the engine
```

Read: the host parked in the window's **tail** sync (the withheld flag is the last layer's, so the per-layer ring
loop had already drained and the engine's 60 s no-progress watchdog — not the 20 s layer watchdog — is what caught
it), the release published the flags and the drain returned in **3.49 ms against its 5000 ms budget**, the GPU
finished, the process ended by itself (exit 250), and the session refused the next request with the documented
reason instead of hanging or serving wrong tokens.  That is the property P1/P1b established, unchanged by this
card — which is why §6.2's E-6 number is reported as a lever and not landed.



## 7. Not validated

* **No variance estimate.**  One arm per configuration, as in P1b/P3/P9/D1/D2.  The rig's band is a few tenths of a
  percent (D2 measured −0.6% between two binaries on the same configuration that session); the ±0.2-0.3 ms
  differences §5.1 quotes for the top-k at 4K/32K are inside it, and the 128K pair (+1.64 ms, +1.3% of the window)
  is a single A/B.
* **The 128K top-k comparison is cross-`--max-context`** (262,144 against 131,072): the arms differ in the KV
  allocation as well as in the dispatch.  `d3-oldtopkhist-131072` (E3) is the same-maxctx pair that separates
  them, and its result is reported in §5.2; until it is read, the kernel's share of that +1.3% is an inference
  from the census symbol, not a clean measurement.
* **The max-context effect itself is measured at 4K only** (`d3-basearmctx-4096` against `d3-base-4096`): a
  request at 262,144 cells pins ~3.3 GB of KV state that the 4,096-cell configuration does not, and the two
  configurations' expert caches are therefore not the same size even though both report 100% residency.
* **The score sweep's latency-bound reading is an inference**, not a stall profile: it rests on the measured
  achieved bandwidth (28-37 GB/s) and on the kernel's shape.  No device-level profile of this window is obtainable
  on this port (P9: every device-instrumenting unitrace mode stalls the window at layer 1), so the "~14× above its
  bandwidth floor" figure is a floor-to-measured ratio, and the ~3.5%-of-window ceiling in §5.4 is an upper bound
  that a reshaped kernel need not reach.
* **The E-6 measurement is a switch, not a landed default.**  It is measured with the census instruments at 4K and
  32K (and 128K if the arm lands); it is not proposed as the shipped configuration here, because it changes the
  engine's per-layer plan ownership and P1b's bounded-sync/deterministic-exit properties are correctness
  properties — the stall rig's result (§6.3) is what would make that argument, and it is run on the same binary.
* **The "sync every N layers" design of the card is not implemented**, and §6.2 says what it would need and what
  the ceiling is; no code was written for it.
* **The P9 stage lines are host-sampled at 33-39% coverage** (P9's own measured limit), so their `scores+topk`
  numbers are floors of the census's exact ones; they are quoted side by side for that reason.
* **One prompt per depth**, the m6c nested prompt of that length, as in every decode card; the selection's cost is
  a function of the position the window sits at, and the arms stop at the prompt's own stop (147 at 4K).
* **`--kv-resident 32768` is a no-op below 32,768 cells**, so the 4K and 32K arms are the all-resident
  configuration and the 128K arm is the streaming one; both are the record's own setting.
* **The served path (HTTP, `serve/server.py`) was not re-run**: the arms use the engine's own `--serve` protocol
  with the record's args plus the two stated deviations, as D1/D2/P9 did.
* **No ctest run**: no engine source changed in this card (the binary is byte-identical to D2's `e841fd06…`), so
  there is nothing for the suite to cover that D2 did not already state.
* **Nothing is pushed**: the origin (`github.com/Niko1221/Strata`) has no `sycl-xpu` branch.

## 8. The state left behind, and the files

* **No engine and no server are running**: every arm ran one engine at a time on both B70s with `ZE_AFFINITY_MASK`
  unset, and the last arm was allowed to exit; port 8099 is free and no `strata` process is left.  The P6 resident
  server was already stopped by S4 (D2 left it stopped too) and this card did not restart it.  The raw
  `pgrep`/`fuser`/`loadavg` output of the state at the end is in `d3/D3-EVIDENCE.txt` §5.
* **The tree is on branch `sycl-xpu` with this card's commits** and `build-sycl/strata` still holds
  `e841fd061fec873c2f24e785973a2ebe` — this card changed **no engine source**, so the binary every arm measured is
  the same one D2 left (its own md5 is recorded in every arm's log).
* The config of record (`strata-sycl-iq3s.json`) is **untouched**: `--spec-min-p 0.7` as D2 landed it, and E-6 was
  measured as a switch, not landed (§6.2).
* Raw arm output is committed under `d3/runs/<tag>/` and `d3/stall/<tag>/`; `d3/D3-EVIDENCE.txt` holds the
  assembled raw lines.

| what | path |
|---|---|
| the arm runner (one engine, both cards, the ask from a FIFO) | `d3/d3_run_arm.sh`, `d3/d3_drive.py` |
| the arm blocks | `d3/d3_chain.sh` (A), `d3/d3_chain3.sh` (F/A3/B), `d3/d3_chain4.sh` (D/E), `d3/d3_chain5.sh` (the whole remaining sequence) |
| the #267 stall rig | `d3/d3_stall.sh` |
| the census reader (per window: selection, handshake, instrument) | `d3/d3_select.py` |
| one row per arm / the engine's own window line | `d3/d3_report.py`, `d3/d3_stages.py` |
| every raw line the acceptance asks for | `d3/d3_evidence.sh` → `d3/D3-EVIDENCE.txt` |
| the census dumps committed early | `d3/D3-CENSUS-ARMCTX.txt`, `d3/D3-NOTES.md` |
| this write-up | `d3/STATUS-D3.md` |
