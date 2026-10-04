# D2a — the generic multi-column MMVQ layout: not a race, and its 60% window-cost spread is a cold program cache

Card `t_8306429a` (D2a), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`, parent `t_0416a0c0` (D2).
One measurement session, both B70s, `ZE_AFFINITY_MASK` unset, one engine at a time, the config of record verbatim
(`--spec-min-p 0.5`, the value D2's lever table was taken at) with D2's own two documented deviations
(`--prefill auto`, `--prompt-cache 0`).  **No engine source, no config, and no default changed**: the engine
binary is still D2's `e841fd061fec873c2f24e785973a2ebe`, and the only commits this card adds are a new kernel-level
harness, its CMake target, and the `d2a/` rig (the git status the evidence file captured before this card's
commit is `d2a/D2A-EVIDENCE.txt` §5).

Rig under `d2a/`: `d2a_chain.sh` (the 17 engine arms, through D1/D2's `d2/d2_run_arm.sh`), `analyze.py` (per-arm
window cost, the part of `verify` no counter owns, new SYCL programs, the ids md5 **as `d2/d2_report.py` computes
it**), `arm_cache.py` / `program_names.sh` (which programs, when), `tseq.py` / `windowtable.py` /
`divergence.py` / `stream_vs_window.py` (the window-size sequences and the token streams), `run_spread.sh` /
`run_spread_cold.sh` / `build_spread.sh` (the kernel-level harness), `d2a_evidence.sh` →
`d2a/D2A-EVIDENCE.txt`.  Raw engine output is under `d2/runs/d2a-*` (the rig D1/D2 used, so every arm's own log
carries the binary md5, the device check and the engine's lines).

## 1. Verdict

1. **The generic layout is NOT racy.**  At kernel level, on the config's own shapes, **every one of 16 repetitions
   produced a bitwise-identical output** for both layouts, in three separate runs (both caches cold, SYCL cache
   cold, warm) — 24 (type, T, layout) combinations, 384 timed launches and 384 hash launches per run, `reps same
   = yes` in every row (§3).  The device time of a built kernel is also stable: e.g. Q6_K 2560x6144 T=4
   0.194/0.195/0.197 ms min/med/max over the 16 reps, Q5_K T=4 0.159/0.159/0.164.
2. **The 60% window-cost spread D2 measured is the generic layout's own SYCL programs being built inside the
   decode windows, and it disappears once they exist.**  The three D2 runs wrote **25 / 5 / 0** new SYCL program
   files *while their own ask was in flight*, and the part of `verify` that no counter owns is **67.29 / 21.27 /
   2.42 ms per window = 3095 / 1085 / 160 ms of the run** (§2).  Eight fresh repeats of the same lever with those
   programs already on disk: **121.19-123.98 ms/window (2.3% spread), residual 2.56-3.89 ms, 0 new programs**.
   At kernel level the build is not subtle: with both caches fresh, a specialization's **first** launch costs
   **63.8-519.7 ms host wall / 65.2-521.1 ms by the stream's own events**, against 0.2-1.2 ms for the same kernel
   built.
3. **The ids movement is not the kernel either, and it is not caused by the cost.**  It is the **window-size
   sequence**.  Over eight identical warm repetitions of the lever there were **three distinct window-size
   sequences and exactly three token streams, 1:1** (4 + 3 + 1 runs, no exception: `d3aea6c8`→`3b0519ca`,
   `b52f6426`→`be7a9062`, `7f849df4`→`70182abf`), at a cost spread of 2.3%.  And with the prompt-lookup drafter
   off — i.e. with the *timing-fed* `DraftPolicy` out of the loop — the **same generic layout produced the shipped
   layout's own canonical stream, `66bf952d…`, in all 8 of 8 runs**, because that arm's window sequence has no
   window wider than 4 tokens (§4).  So: cost and ids are not cause and effect; both are downstream of the window
   sequence, and the reason a changed window sequence can move the ids at all is that the generic layout is only
   bitwise-equal to the single-column call for `T <= 4` (`native_mmvq.cu:1050`).
4. **The culprit lines**, in the order the chain runs: `src/kernels/cuda/native_mmvq.cu:1050`
   (`constexpr int NW = NCOLS <= 4 ? 4 : 2;` — the layout the whole effect hangs on),
   `include/strata/spec/draft_policy.hpp:36` (`observe(..., double round_ms)` — the window-size choice is a
   function of measured round times; its own header at `:15` says "It only chooses which drafts to verify: the
   output is unchanged", which holds only for a layout that is bitwise-equal at every T), and
   `src/program/generate.cpp:5494-5496` (the round time that is fed to it).  The SYCL program build has no line in
   this repo: nothing in the engine builds the selected layout's specializations before the first decode window
   (the graph path only *captures* per T; there is no kernel warm-up pass).

## 2. The mechanism: the layout's own programs are built inside the decode windows

The engine's window timing line names `verify = GPU-reach wait + per-layer host + stage + tail`; the rest is real
time inside the window that no counter owns (D2's `STATUS-D2.md` shows the shipped layout at ~3 ms/window).  That
residual, the run's own count of **new** SYCL program files written between its ask and its finish, and the
programs themselves (`d2a/D2A-EVIDENCE.txt` §1, `d2a/programs.txt`):

| arm (binary `e841fd06…`, `STRATA_MMVQ_MULTI_GENERIC=1`) | ms/window | residual ms/win | residual × windows | new programs | ids md5 |
|---|---|---|---|---|---|
| `d2-mg-4096` (D2, 1st run of the layout) | 191.79 | **67.29** | **3095 ms** | **25** | `10160c38…` |
| `d2-mg-4096-warm` (D2, 2nd) | 140.39 | **21.27** | **1085 ms** | **5** | `3b0519ca…` |
| `d2-mg-4096-warm2` (D2, 3rd) | 120.68 | **2.42** | **160 ms** | **0** | `70182abf…` |
| `d2a-mg-4096-r1..r8` (this card, warm) | 121.19-123.98 | 2.56-3.89 | 151-198 ms | **0** | 3 streams |
| `d2a-rebase-4096` (this card, control, shipped layout) | 125.36 | 3.47 | 177 ms | 0 | `66bf952d…` |

**The 25 programs are named, and they are exactly the (quant type × column count) specializations that run's own
window sequence needed.**  Every one is `native_mmvq_multi_kernel<...>`: Q6_K/Q5_K/Q4_K/IQ4_XS/IQ4_NL at
`NCOLS = 2, 3, 4` (the `NW = 4, ROWS = 2` shape) and at `NCOLS = 6` (the `NW = 2, ROWS = 2` shape), plus the MTP
drafter's Q8_0 at `NCOLS = 2, 3, 4, 6, 8` — 5+4+4+4+4+4 = 25.  The second run's **5** new programs are the same
five dense types at **`NCOLS = 5`**: `d2-mg-4096-warm`'s window sequence contains a 5-token window that
`d2-mg-4096`'s does not, and nothing else is new.  The third run needed none, because it saw no window size the
first two had not.  Timestamps: the first program is written 12.9 s after the ask is fed and the rest by +20.7 s,
i.e. inside the run (its prompt is 15.7 s of it, its decode 8.8 s).

**The build cost, measured at kernel level** (`d2a/mmvq_multi_spread.cpp`, `d2a/spread-cold-fully.txt`): with
`SYCL_CACHE_DIR` *and* the Intel NEO/IGC native-binary cache (`NEO_CACHE_DIR`) both fresh, the first launch of
each specialization costs 63.84 / 81.35 / 70.94 / 107.35 ms (Q6_K), 78.98-146.93 ms (Q5_K), 75.85-143.27 ms
(Q4_K), 203.39-519.72 ms (IQ4_XS), 114.12-336.46 ms (IQ4_NL) — and the stream's own event delta agrees
(65.2-521.1 ms), which is why it lands in a *verify* window rather than in the load phase.  **20 specializations,
3384 ms of build, 169 ms mean** — and 25 programs × ~124 ms = 3095 ms is the same arithmetic on the engine's own
numbers.  A fresh `SYCL_CACHE_DIR` **alone** is not a cold compiler (the NEO cache still holds the binaries:
`d2a/spread-cold.txt`, no first launch above 50 ms), which is exactly why the first run of a *new* layout pays and
later ones do not.

## 3. The kernel level: 16 reps × 24 combinations, both layouts

`d2a/mmvq_multi_spread.cpp` (new, `add_executable`, no `add_test`: it is a measurement), synthetic weights in the
parity harness's own style, one quantization shared by all calls, the config of record's shapes (n_in 2560,
n_out 10240/6144; `d2/D2-TYPES.txt`), one run per (type, T, layout) with T = 4 and T = 6:

| what the 16 reps say | result |
|---|---|
| **value determinism** — every rep's whole output hashed (FNV-1a) | **`reps same = yes` in all 24 rows of all three runs** (both caches fresh / SYCL cache fresh / warm). No race, at either T, for either layout. |
| **device time spread** (min/med/max over the 16 built launches) | tight: Q6_K 6144 T=4 exact 0.194/0.195/0.197, generic 0.198/0.200/0.202; Q4_K T=4 0.159/0.160/0.164; IQ4_NL 10240 T=4 0.541/0.543/0.548 ms. The generic layout's T=6 numbers are within 1-3% of their own median. |
| **first launch = build** | 63.8-519.7 ms host wall, 65.2-521.1 ms event-measured (§2), vs 0.2-1.2 ms warm. |
| **the two layouts' difference** | **T = 4: 0 words differ, in every case** (the `NW = 4` coincidence the parity harness documents). **T = 6: 15441-42078 words differ** out of 36864-61440, and the largest of them is **1.2e-4 … 6.1e-4 absolute = 5.8e-7 … 7.2e-7 of the exact layout's output RMS**. Last-bit rounding, not a wrong answer. |
| **the invariant, re-checked** | `build-sycl/mmvq_multi_parity` (unchanged): `multi_exact on` 148480 outputs, **0 differ**; `multi_exact off` **63708 differ, all at T > 4**, 0 at `T <= 4`. So the engine's only route from "which window size" to "different numbers" is a window wider than 4. |

One case is not tight and is named rather than hidden: the **exact** layout at n_out=10240, T=4 measured
0.55/0.95/1.22 ms med across the three runs (min-to-max inside one run 0.95-1.22) where the generic layout at the
same shape is 0.31 and both layouts are ~0.20 ms at n_out=6144.  That is a 10240-block grid of the `ROWS = 1`
shape on this device; it is not part of this card's answer and this harness is not the instrument to price the two
layouts with (D2's census is).  Whoever prices the generic layout should use a harness that holds the work fixed.

## 4. The ids: not the kernel, not the cost — the window-size sequence

D2's three runs moved the ids.  This card ran the same lever eight more times at 4K with the programs now on disk
(`d2a-mg-4096-r1..r8`; `d2a/D2A-EVIDENCE.txt` §1/§3, `d2a/stream_vs_window.py`):

* eight runs, **three distinct window-size sequences, three distinct token streams, and the map is exactly 1:1**
  (4 runs `d3aea6c8` → `3b0519ca…`, 3 runs `b52f6426` → `be7a9062…`, 1 run `7f849df4` → `70182abf…` — that last
  one is D2's own third run's stream).  Cost over those eight: 121.19-123.98 ms/window.
* **The control that decides it**: the same lever plus `--suffix-draft 0`, which takes the `DraftPolicy` out of the
  loop (the window size is then the MTP's own min-p-gated window, a function of the model's numbers only), eight
  runs (`d2a-mg0-4096-r1..r8`): **one window-size sequence (`eb4beddf`) and one token stream in all 8 of 8 runs,
  `66bf952d445e330c974c49e4f220b4b1` — the shipped layout's canonical stream — at 147 tokens and 120.09-120.89
  ms/window**.  That arm's window sequence has **no window wider than 4 tokens**, so the generic layout's one
  numeric difference never fires: the arm is the ids check the card asks for, and it says the layout's numbers are
  the shipped layout's numbers wherever the layouts coincide.
* **Where the runs part company.**  Every generic run (cold or warm) takes a **2-token window at window 5** where
  the shipped control takes the 4-token MTP window (`d2a/windowtable.py`: rebase `9+4`, all mg runs `9+2`) — a
  choice of the lookup drafter, made from measured round times, and present in the warm runs too, so it is the
  layout's own cost rather than the JIT that flips it.  That change alone moves no token (T=2 and T=4 are bitwise
  equal).  The streams then differ later, at **token 78** (r1 vs r2, r1 vs r7) or **88** (r2 vs r7), and in each
  pair the first *window-size* difference is a **5-token window at 19/21** — i.e. the divergence follows a T>4
  window by 10-25 tokens, the way a last-bit difference takes a few tokens to flip a greedy argmax.  The three
  sequences also all share the T=6 window at window 16 (tokens 40-45) that the shipped control has too; that
  window's numbers differ between the layouts (measured above) and do not by themselves move a token in these runs.

## 5. Causality, as the card asks (3)

* **The ids' movement cannot be the cause of the cost variation**: `d2a-mg-4096-r1` reproduces D2's
  `d2-mg-4096-warm` stream (`3b0519ca…`) at **122.28 ms/window against that run's 140.39** — a 13% cheaper window
  for the same tokens; and eight warm runs whose streams differ (three of them) all sit in 121.19-123.98.
* **The cost variation is not required for the ids' movement either**: the eight warm runs have 0 new programs and
  a 2.3% cost spread, and still produce three streams.
* Both are downstream of the **window-size sequence**: the cost because the sequence decides *which* (type ×
  ncols) specializations a run ever needs to build (25 / 5 / 0), the ids because the sequence decides which
  windows are wider than 4 and those compute different numbers.  The timing enters through `DraftPolicy`, which is
  fed each round's measured milliseconds; the policy is small and settles within ~3% (`margin = 0.03`), so a run's
  ordinary timing jitter is enough to flip a choice — and for the **shipped** layout that is harmless, because its
  numbers do not depend on T at all.

## 6. What a fix would look like (nothing shipped; the switch stays default-off)

1. For the cost: build the selected layout's `(type × ncols)` specializations **before the first window** (a
   warm-up pass over the set the request can reach: `ncols = 1..8` × the pack's dense types, plus the drafter's),
   the way the engine already warms expert pages (`generate.cpp:3055`).  That removes a data-dependent 0.1-3.1 s of
   build time from the measured window and, with it, this card's 60% spread.  It changes no numbers.
2. For the ids: the generic layout only becomes safe to ship when it is bitwise-equal to the single-column call at
   every T, or when `--spec`'s window choice stops being fed measured round times.  Anything less makes greedy ids
   a function of machine timing, which is the one property this project uses ids as a guard for.  D2's refusal
   ("ids move") stands, and this card adds the mechanism behind it.

## 7. What was ruled out, and the negatives that bound it

1. **A race in `native_mmvq_multi_kernel`** (`src/kernels/cuda/native_mmvq.cu:1001-1062`): ruled out at kernel
   level — 16/16 bitwise-identical outputs per (type, T, layout), both layouts, T = 4 and T = 6, three cache
   states.  The `NW = NCOLS <= 4 ? 4 : 2` / `blocks = (n_out+1)/2` shape at `:1049-1053` is a *different* (and
   correct) reduction, not an unstable one.
2. **A pathological launch configuration whose cost depends on data**: not what the engine's spread was — the
   spread is the build count, and at kernel level the built kernel's time is flat to a few percent.
3. **"The generic layout is simply slower"**: not supported.  With the lookup drafter off, the generic layout
   produces the shipped stream with a window sequence that has no T>4 window, and its 51 windows / 147 tokens cost
   120.09-120.89 ms against the shipped control's 125.36 — but that arm also verifies no T>4 window at all (the
   shipped sequence has two), so it is **not** a like-for-like price of the two layouts.  Priced at kernel level with
   the work held fixed, on the engine's dominant 6144-row shapes the two are within 2-6% (Q6_K T=4
   0.198/0.200/0.202 generic vs 0.194/0.195/0.197 exact; Q4_K 0.167-0.172 vs 0.159-0.163; Q5_K 0.168-0.174 vs
   0.159-0.164), while at n_out=10240 the generic layout is faster (Q6_K T=4 0.307-0.310 vs 0.549-1.223, the exact
   number being the one that is not tight — §3).
4. **Not measured, and said so**: whether the *JIT spike itself* flipped a policy choice (the window-5 flip is
   present in warm runs, so it is not needed to explain the ids), and the exact token-level path from "a T>4
   window's committed rows differ" to "token 78 differs".  What is measured is the 1:1 map from window sequence to
   stream over 8 runs, the layout's difference confined to T>4 at a relative magnitude of ~6e-7, and the fact that
   the streams agree for 77 tokens across sequences that had already diverged at window 5 and had already verified
   a T=6 window at tokens 40-45.

## 8. Machine state, and how to re-run

Machine: **left clean** — the last arm (`d2a-mg0-4096-r8`) was allowed to exit, no engine (`pgrep -x strata`
empty) and no `serve/server.py`, port 8099 free, `ZE_AFFINITY_MASK` unset, both cards free; the engine binary is
unchanged at `e841fd06…`.  Nothing was pushed (as in D2: `origin` has no `sycl-xpu` branch).

```bash
# the kernel level (no engine; GPU-only, ~5 s)
bash d2a/build_spread.sh          # builds build-sycl/mmvq_multi_spread only
bash d2a/run_spread.sh 16         # SYCL cache cold then warm
bash d2a/run_spread_cold.sh       # SYCL *and* NEO caches fresh: the per-specialization build cost
# the engine arms (one engine at a time, ~60 s each; skips arms already complete)
bash d2a/d2a_chain.sh
# every raw line of this card
bash d2a/d2a_evidence.sh          # -> d2a/D2A-EVIDENCE.txt
```
