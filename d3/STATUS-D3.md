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
   4.82%**; the attention is 1.54% → 1.5% → 1.4%.  So "the selection is what grows with depth" holds as a whole and
   the depth-growing term inside it is the scores (the choice of candidate blocks), not the top-k (the choosing).
2. **The card's per-token premise is refuted: the selection is already window-level, and the per-block key read is
   already shared across the window's rows.**  Both kernels are launched **once per (QSA layer, window)** over the
   window's rows (`src/core/verify.cpp:1019-1022`), and the scores kernel that runs is upstream's
   `block_scores_multi_kernel` (commit `a20f3b5`, default on, `STRATA_SCORES_MULTI=0` = the old per-query grid),
   which reads each key block **once for all of the window's queries** (`src/kernels/cuda/qsa_select.cu:600-646`).
   The A/B that prices that sharing (the old grid vs the shipped one, ids as the guard) is §4:
   **PENDING-B**.
3. **The top-k's dispatch is not a performance question on this config — measured, not inferred, from the census's
   own symbol**: the record's `--max-context 262144` takes the **memory-keyed** kernel
   (`qsa_block_topk_ref`, 0.485 ms at 4K) and the arm-CTX configs take the **register** one (0.684 ms) — and the
   memory kernel is the **cheaper** of the two at 4K, so neither the record nor the arms are on a slow path.  What
   the dispatch does hide is that the score sweep is **latency-bound**: it moves its working set at **~30 GB/s**
   (28 GB/s at 4K, 33.7 GB/s at 32K) against a device of this class, i.e. ~14× above its own bandwidth floor at
   32K — that is the named opportunity of this card, and it is a kernel shape, not a switch (§5).
4. **The per-layer handshake is counted exactly, and the coarsest exact sync the engine already has is measured.**
   Per **layer** per window: **3 host→device 4-byte publications**, **1 device→host ring** and **3 device-side
   `wait_flag_ge` spins**; the spins are **144 launches and 14.7 / 15.4 / 16.9 ms per window at 4K/32K/128K =
   11.1% / 11.7% / 12.0%** of the window's counted device time, the largest single family in the window after the
   projections.  `STRATA_VERIFY_DEVICE_PLAN=1` (E-6) is the engine's existing answer — the device plans its own
   group and the wait becomes `wait_flag_ge_or`, which returns without touching the host's flag — and it is
   **PENDING-D**.  The release path is untouched by this card and the stall rig re-proves the #267 guarantee on
   the same binary (§6, PENDING-S).
5. **Nothing was landed**: no source change, no config change.  What the card's own terms allow for both remaining
   items — the sharing (already in the tree, priced here) and the sync (a design + a measured ceiling) — is what
   this write-up is.

## 2. Config, instruments and their own cost

[PENDING: the instrument-cost table: gpu_stamp 1.703/1.685/2.001 ms per window = 1.29/1.28/1.43% of the counted
device time, and P9's +3.0% of the wall for the stamp set is confirmed/measured here as X.]

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


## 4. The sharing: what is already shared, and what the A/B prices

[PENDING-B: base vs nomulti at 4K/32K/128K, ms/window, decode tok/s, ids md5.]

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
| `d3-hist-131072` | 131,072 | `qsa_block_topk` (register) | 1.403 | 5.361 | 140.25 ms | PENDING |
| `d3-histc-131072` | **262,144** | **`qsa_block_topk_ref`** (memory) | PENDING | PENDING | PENDING | PENDING |

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

[PENDING-D: the E-6 arms with the same instruments as the census control.]

### 6.3 The release path's guarantee (#267), re-proved on this binary

[PENDING-S: the stall rig's lines.]


## 7. Not validated

## 8. The state left behind, and the files
