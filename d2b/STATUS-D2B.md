# D2b — the decode path builds its (quant type × ncols) specializations inside the first decode windows, and a load-phase warm-up moves them there

Card `t_f93760a1` (D2b), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`, parent `t_8306429a` (D2a).
One session, both B70s, `ZE_AFFINITY_MASK` unset, one engine at a time, the config of record's flags with D2's two
documented deviations (`--prefill auto`, `--prompt-cache 0`) and `--spec-min-p 0.7` (the value the config of
record carries since D2; D2 measured that 0.5 and 0.7 give the same 4K and 32K ids).
**Engine source touched**: `src/kernels/cuda/native_mmvq.cu` (+`native_mmvq_warmup`), its header
`include/strata/kernels/native_mmvq.hpp`, and one call in `src/program/generate.cpp`'s serve path. The engine
binary is `f49fe6669704561bb8155f9c4c9acaed`, built from the pre-change `e841fd061fec873c2f24e785973a2ebe` (kept
at `/home/michael/strata-xpu/d2b/strata-e841fd06`), which every pre-fix arm measured. No config change, no
default of the multi-column layout touched (D2a closed that question).

Rig under `d2b/`: `d2b_run_arm.sh` (D1/D2/D2a's `d2/d2_run_arm.sh` plus the two cache directories as explicit
knobs and a before/after cache census), `analyze_b.py` (per-arm residual, ids md5, programs written inside the
ask, load wall), `programs_b.py` / `classify.py` / `name_other.py` / `cache_names_b.py` / `cache_devices.py` /
`decode_builds.py` (which programs, when, and in which phase), `ids_b.py`, `d2b_chain_a.sh` (the cold shipped
run + its warm control), `d2b_chain_b.sh` (the five warm-up arms), `d2b_ctest.sh`, `d2b_evidence.sh` →
`d2b/D2B-EVIDENCE.txt`. Raw engine output is under `d2/runs/d2b-*` (the same rig D2/D2a used, so every arm's log
carries its own binary md5, cache directories and device check).

## 1. Verdict

1. **The card's premise is confirmed on the SHIPPED layout, as the user-visible case.** A cold first run (both
   `SYCL_CACHE_DIR` and `NEO_CACHE_DIR` empty = a fresh install) of the shipped layout builds **153 SYCL
   programs while its own ask is in flight**, of which **32 are the dense `native_mmvq` (type × ncols)
   specializations**, and the part of `verify` no counter owns is **127.93 ms/window × 54 windows = 6908 ms** —
   83% of the run's total window time (the same run's warm control: 3.56 ms/window, 192 ms). §2.
2. **The ids do not move**: every arm of this card, cold or warm, warm-up or not, produced the 4K canonical
   `66bf952d445e330c974c49e4f220b4b1` (147 tokens) and the 32K one `d87373e84417dab60732a95eb47666a6` (256
   tokens) — the shipped layout's numbers do not depend on the window size, so the cost was never the cause of the
   ids (D2a's §4 finding, re-confirmed here from a genuinely cold side). §5.
3. **A load-phase warm-up removes the dense family from the decode phase and nothing else.** With the warm-up the
   cold run's decode phase builds **61 programs, 0 of them mmvq** (it was 93 with 32 mmvq), the residual falls to
   **36.27 ms/window (1959 ms)** and the decode wall from **13696 ms to 8587 ms**. The 32.27 ms/window that remain
   are *not* the dense family: they are 6 routed-expert `gu`/`down` kernels plus 55 other one-off kernels (bf16
   `mmvf`, `quantize_q8_0_scaled`, `gdn_*`, `qsa attn_chunk`/`block_scores`, `mtp_select`, `sampler_greedy`, …),
   ~one program per decode-path kernel site. Warm, the residual stays at the shipped ~3.5 ms (2.50 / 2.85 ms with
   the warm-up on/off). §3, §4.
4. **The warm-up costs what the card asked to price**: 192 launches (6 types × ncols 1..8 × 2 launch shapes, plus
   the Q8_1 quantization each time), **+17.5 s on a fully cold first run**, **+7.3 s** in a session where the dense
   programs were only partly cached, **+41 ms** once they exist (so the steady-state load wall is the shipped
   31-32 s). On a fresh install the first *request* is 5.5 s faster (33.77 → 28.22 s) but the first *answer* is
   12.4 s later overall (31 s + 33.8 s → 49 s + 28.2 s), because the warm-up builds ~60 programs the request would
   not have needed while the other 61 decode-path programs still build in the request either way. §6.
5. **Tests**: `ctest -R mmvq_multi_parity` passes; the S4 ctest set is **5 failed of 49 — the same five names as
   the recorded baseline** (`platform_memory_test`, `elementwise_parity`, `quantize_act_parity`,
   `iq_multi_parity`, `expert_multi_test`). The warm-up launches kernels a request may never use, and that is
   exactly what these two runs bound: it changes no test outcome. §7.

## 2. The user-visible case: a cold first run of the shipped layout

Both caches fresh (`SYCL_CACHE_DIR` and `NEO_CACHE_DIR` pointed at new /tmp directories; the project's shared
warm caches were **never moved or deleted** — a fresh directory is the same thing as moving the entries aside,
with nothing to restore afterwards). 4K, 256 new tokens, `--spec-min-p 0.7`, `--no-capture`, both GPUs:

| arm | binary | caches | warm-up | ms/win | verify | wait | tail | **RESID** | **RESID×win** | programs in ask | load | ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `d2b-cold-4096` | `e841fd06` (pre) | **both fresh** | (not in binary) | 253.63 | 229.68 | 88.37 | 11.12 | **127.93** | **6908 ms** | **153** | 32 s | `66bf952d…` |
| `d2b-ctloff-cold-4096` | `f49fe666` | **both fresh** | off | 254.00 | 230.11 | 88.57 | 11.11 | **128.23** | **6924 ms** | **153** | 31 s | `66bf952d…` |
| `d2b-wu-cold-4096` | `f49fe666` | **both fresh** | **on** | 159.02 | 136.79 | 87.30 | 11.10 | **36.27** | **1959 ms** | 117 | 49 s | `66bf952d…` |
| `d2b-rebase-4096` | `e841fd06` | warm | — | 112.83 | 100.64 | 83.97 | 10.94 | 3.56 | 192 ms | 0 | 31 s | `66bf952d…` |
| `d2b-ctloff-4096` | `f49fe666` | warm | off | 112.10 | 100.01 | 84.07 | 10.94 | 2.85 | 154 ms | 0 | 31 s | `66bf952d…` |
| `d2b-wu-4096` | `f49fe666` | warm | on | 111.64 | 99.58 | 84.09 | 10.94 | 2.50 | 135 ms | 0 | 39 s¹ | `66bf952d…` |
| `d2b-wu-32768` | `f49fe666` | warm | on | 113.60 | 99.65 | 85.07 | 10.95 | 1.38 | 127 ms | 0 | 32 s | `d87373e8…` |

¹ `d2b-wu-4096`'s 39 s is the first run in which the warm-up found 40 of its 96 specializations unbuilt
(`+40` new cache entries, its own line: 7298 ms); with them on disk the next arm's warm-up costs 41 ms. The
load wall is otherwise 31-32 s in every warm arm, exactly as D2/D2a recorded it.

The cold run's ask is 33.25 s (prefill 19.54 s + decode 13.70 s) against the warm control's 21.17 s (14.97 +
6.09). The new binary with the warm-up switched off reproduces the pre-fix numbers (254.00 ms/win, 128.23
ms/win residual, 153 programs, 13716 ms of decode) — so the 3.5x drop is the warm-up, not the rebuild.

**What the 153 programs are, and what the warm-up takes out of them** (`d2b/classify.py`, `decode_builds.py`):
split at the engine's own "prompt … read in 19392 ms" line, the cold decode phase builds **93** programs:
**32 dense mmvq** (the family the card names: `native_mmvq_multi_kernel<F, NCOLS, NW, ROWS>` for
Q4_K/Q5_K/Q6_K/IQ4_XS/IQ4_NL/Q8_0 at the NCOLS the window sequence visits, both launch shapes, plus the
single-column `native_q6_k_mmvq_kernel`/`q5_k`/`q4_k`/`iq4_xs`/`small_mmvq` forms at ncols = 1), **6 routed
expert `gu`/`down` kernels** and **55 other** one-off kernels. With the warm-up: **61 programs, 0 mmvq** — the
warm-up removed exactly the 32 dense mmvq builds and nothing else. This is also why the residual does not reach
zero: the 55 "other" + 6 expert builds are a second, larger family of first-launch builds inside the decode
windows, and they were not this card's scope (§8 takes it up).

## 3. The mechanism, on the shipped layout

`native_mmvq` dispatches `ncols > 1` to `launch_multi<F>`, whose `native_mmvq_multi_kernel<F, NCOLS, NW, ROWS>`
is a template over (type, NCOLS) — not over the operand sizes — and picks its NW/ROWS **at runtime** from
`n_in / F::DIV < F::BPI` (the shipped "exact" layout; `native_mmvq.cu:1046-1062`). So the shipped binary itself
has one SYCL program per (type, NCOLS, launch shape) that the window sequence reaches, and the first launch of
each builds it in the window that reaches it. A cold first run's window sequence is data-dependent (D2a §4: the
window size is chosen from measured round times through `DraftPolicy`), which is what makes the
pre-fix cost data-dependent: *which* (type, NCOLS) pairs a run needs is a function of its own window
sequence.

The cold run's residual (6908 ms) is the sum over all 93 first launches inside its decode phase; the 32
dense mmvq builds are the named part of it, and the only part the warm-up removes. That part is worth
127.93 - 36.27 = **91.66 ms per window x 54 windows = 4949 ms**, i.e. about 155 ms per dense build - the
same order as D2a's kernel-level first-launch costs (63.8-519.7 ms, mean 169 ms for 20 specializations).

Timestamps of the cold run's programs (`d2b/programs_b.py`, `d2b/D2B-EVIDENCE.txt` §3b): the first mmvq
specialization of the sequence appears at +5.1 s and the dense builds run out to +30 s of the 33.25 s ask, i.e.
they are spread across the decode phase (the prompt is 19.5 s of it) — one build per window that first reaches a
new (type, NCOLS) pair.

## 4. The warm-up

`native_mmvq_warmup(void* stream)` (`src/kernels/cuda/native_mmvq.cu`), called once from the serve path in
`src/program/generate.cpp` immediately before the "everything loaded" line — after every weight, the drafter and
the prompt paths are in place, before the first request (the ask is fed from the FIFO after that line) and before
any window has run:

* it launches, for each of the six quantization types the config of record's decode path reaches (Q4_K, Q5_K,
  Q6_K, IQ4_XS, IQ4_NL — `d2/D2-TYPES.txt`'s NATIVE-DENSE types — plus the MTP drafter's Q8_0, `mtp.cpp`'s own
  `native_mmvq` calls), `ncols = 1..8` (the kernel contract's whole range) and **two input sizes per type** — one
  below and one above the `n_in / F::DIV < F::BPI` crossing, so both launch shapes are built. 192 launches, 96
  specializations;
* it runs on the caller's stream over **scratch buffers it allocates and frees itself** (≈100 KiB), with an
  all-zero activation: nothing it computes is read by anything, which is why it cannot change a number — and the
  ids check (§5) is the evidence, not the argument;
* the type list is a compile-time table next to the traits it uses (`Q4KTraits::DIV`/`BPI`, …), so the two input
  sizes are derived from the kernel's own constants rather than hand-copied;
* `STRATA_MMVQ_WARMUP=0` is the A/B control arm (it prints `strata mmvq warmup: off (STRATA_MMVQ_WARMUP=0)`);
* the default is **ON**: the point of the card is that a fresh install's first decode windows are not paying for
  the compiler, and the cost of the warm-up is paid once per install (see §6 for the trade-off it leaves).
* the warm-up is on the `--serve` path only. A `strata generate` bench run has the same lazy first launches and
  would need the same call before its first window; that is noted for the follow-up rather than done here, because
  this card's rig and its user-visible case are the served path.

## 5. The checks the card asks for

**ids, 4K and 32K, shipped layout, plus a same-session rebuild control arm with the warm-up off.**

| arm | caches | warm-up | binary | tokens | ids md5 (`d2a/analyze.py`'s recipe) | canonical? |
|---|---|---|---|---|---|---|
| `d2b-cold-4096` | both fresh | — (pre-fix) | `e841fd06` | 147 | `66bf952d445e330c974c49e4f220b4b1` | **YES** |
| `d2b-ctloff-cold-4096` | both fresh | off | `f49fe666` | 147 | `66bf952d…` | **YES** |
| `d2b-wu-cold-4096` | both fresh | **on** | `f49fe666` | 147 | `66bf952d…` | **YES** |
| `d2b-rebase-4096` | warm | — (pre-fix) | `e841fd06` | 147 | `66bf952d…` | **YES** |
| `d2b-ctloff-4096` | warm | off | `f49fe666` | 147 | `66bf952d…` | **YES** |
| `d2b-wu-4096` | warm | **on** | `f49fe666` | 147 | `66bf952d…` | **YES** |
| `d2b-wu-32768` | warm | **on** | `f49fe666` | 256 | `d87373e84417dab60732a95eb47666a6` | **YES** |

Note the odd-looking result that the *cold* arms match the canonical too: the shipped layout's numbers do not
depend on the window size (D2a §3's `T ≤ 4` bitwise equality plus the T-independent single-column forms), so even
a window sequence distorted by 6.9 s of compiler still commits the same 147 tokens. That is a property of the
shipped layout, not of the warm-up; on the `STRATA_MMVQ_MULTI_GENERIC=1` layout (D2a's lever) a cold first run
would be expected to move the ids, and that is one more reason the layout stays refused and default-off.

**Residual per window with the caches warm**: 2.50 ms (warm-up on) / 2.85 ms (off) / 3.56 ms (pre-fix binary) at
4K and 1.38 ms at 32K — the shipped ~3.5 ms, unmoved. The residual is the part of `verify` no counter owns
(`verify − wait − per-layer host − stage − tail`), reported per window and ×windows in §2.

**Load wall before and after**: 31-32 s in every warm arm (shipped), 49 s on the cold first run with the warm-up
on (31 s without) — i.e. the build lands in the load phase exactly as intended, and it is *not* free: §6.

**The tests** (§7) — `mmvq_multi_parity` and the S4 set — pass at the recorded baseline.

## 6. What the warm-up costs, and the honest trade-off

`strata mmvq warmup:` lines, one per arm (`d2/D2B-EVIDENCE.txt` §4):

| where the dense programs stand | warm-up wall | load wall | of which the warm-up |
|---|---|---|---|
| nothing built (both caches fresh) | **17538 ms** | 49 s | +18 s |
| 56 of 96 built (a warm session that had never run those NCOLS) | **7298 ms** | 39 s | +8 s |
| all built | **41 ms** | 32 s | ±0 |

On a fresh install, start → first answer (4K, 147 tokens): **31 s + 33.8 s = 64.8 s without the warm-up**, **49 s
+ 28.2 s = 77.2 s with it** — the *request* is 5.5 s faster (decode 13.7 → 8.6 s, 10.7 → 17.1 tok/s) but the
*answer* arrives 12.4 s later, because the warm-up builds ~60 programs the request may never need while the 61
non-mmvq programs still build inside the request either way. Two things follow, and both are for the operator,
not for this card: (a) the 4K *decode* cost — the thing every D-card measures — is now data-independent and at
the shipped level; (b) the fresh-install total only pays off once the same treatment covers the rest of the
decode path (`§8`), or once the warm-up is not the first thing a new install runs. On the second and every later
request it is 41 ms.

## 7. The tests

`d2b/d2b_ctest.sh` (the S4 convention: oneAPI sourced, `ZE_AFFINITY_MASK=0`, `make -j8` first because the tests
relink the new `native_mmvq_warmup` symbol):

| what | result |
|---|---|
| `ctest -R mmvq_multi_parity` (the layout contract the warm-up launches against) | **1/1 passed**, 2.03 s |
| the mmvq family (`-R "mmvq\|iq_multi\|draft_policy\|coupled_draft\|native"`) | 3/4 passed, `iq_multi_parity` failed |
| the full suite (the S4 ctest set, 49 tests) | **5 failed of 49** — `platform_memory_test`, `elementwise_parity`, `quantize_act_parity`, `iq_multi_parity`, `expert_multi_test` |
| the recorded baseline (`s4/runs/ctest/full.log`, D1's "5 failed of 49 — the same five P3 recorded") | **the same five names** |

The warm-up is not reachable from any test (it is called from the serve path of the engine binary only) and it
adds no kernel: the only kernel-library change is the new function. The five failures are the pre-existing set.

## 8. What is left, and what a follow-up would be

The warm-up covers the (dense type × ncols) family the card names. A cold first request still builds **61**
programs inside its decode windows — 6 routed-expert `gu`/`down` specializations (types 18/20/21/22/23/42, one
per dense-expert type of the pack) and 55 one-off kernels (`bf16_f32_mmvf_multi_kernel<256,8>/<256,4>`,
`bf16_f32_mmvf_kernel<160>/<256>`, `quantize_q8_0_scaled`, `row_top_prob`, `gu_grouped_t`/`down_grouped_t`,
`group_resident`, `mtp_select`, `sampler_greedy`, `attn_chunk<1>`, `block_scores_multi`, `gate`,
`gdn_step_norm_multi`/`gdn_ab_multi`/`gdn_conv_l2_multi`, …), 36.27 ms/window = 1959 ms of the cold run. They
are the same phenomenon (a first launch that has to build) in a different family, and they are the reason the
fresh-install total is still net-negative with the warm-up. A second warm-up pass over exactly that set would
remove the rest of the 1959 ms and, on this card's own numbers, would turn the +18 s of load into a net win
(the pre-fix decode paid 6.9 s + 61 more builds). It is a separate change with its own type tables (the expert
kernels are templated per pack type and per `NativeExpertLayout`) and its own ids/parity checks, so it is left as
follow-up card **t_c7d8cd86** ("D2c") rather than folded in here; the card carries this section's numbers as its
starting point.

## 9. Machine state, and how to re-run

Machine: **left clean** — the last arm (`d2b-wu-32768`) was allowed to exit; no engine (`pgrep -x strata`
empty), no `serve/server.py`, port 8099 free, `ZE_AFFINITY_MASK` unset, both cards free (`d2b/D2B-EVIDENCE.txt`
§9). The shared warm caches were never moved or deleted; they gained the 40 dense specializations the first
warm-up launched (+40 `0.src` entries in `$R/sycl-cache/m6c`, and the matching NEO/IGC entries), which is
ordinary cache growth and is why §6's second row exists. Nothing was pushed (as in D2/D2a: `origin` has no
`sycl-xpu` branch). The pre-change binary is preserved at `/home/michael/strata-xpu/d2b/strata-e841fd06`.

```bash
# the cold first run of the shipped layout and its warm control (chain A; ~4 min)
bash d2b/d2b_chain_a.sh
# the warm-up's five arms on the rebuilt binary (chain B; ~10 min)
bash d2b/build_engine.sh && bash d2b/d2b_chain_b.sh
# the tests (needs the GPU; ZE_AFFINITY_MASK=0, the S4 convention)
bash d2b/d2b_ctest.sh
# every number of this card
bash d2b/d2b_evidence.sh          # -> d2b/D2B-EVIDENCE.txt
# the analysis on its own
/usr/bin/python3 d2b/analyze_b.py --all-cold --all-warm
/usr/bin/python3 d2b/classify.py d2b-cold-4096 d2b-ctloff-cold-4096 d2b-wu-cold-4096
```
