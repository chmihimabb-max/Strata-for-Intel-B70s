# D1 — the decode window's economics: the graph path is the default, the spec width is a sweep, and the kernels are ranked

Card `t_d9ffcf38` (D1), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`.  Config of record in every arm:
IQ3_S GSQ-RCO snapshot `ed59f920…`, **both B70s** (`ZE_AFFINITY_MASK` unset, `--layer-split auto`), `--kv int8
--kv-resident 32768 --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts --prefill auto
--spec-min-p 0.5 --no-capture --stats --prompt-cache 0 --prompt-cache-every 0`, MTP drafter `~/strata-xpu/mtp/rt`
(P7's canonical build: `draft_vocab.bin` present, 425,196 B, the head over 106,299 tokens / 212.9 MiB), one engine
at a time, warm program cache (`sycl-cache/m6c`), greedy, `STRATA_DECODE_TIMING=1`, `STRATA_SUBMIT_COUNT=1`.

Two engine binaries, and every arm names the one it ran (the rig copies it into `build-sycl/strata` first):

| binary | md5 | what it is |
|---|---|---|
| `d1/strata-before` | `9eff0675059cf7e8145ae7b4bca984a4` | the pre-change build (P9/P10's binary): `STRATA_SYCL_GRAPH` opt-in, so its default is the closure path |
| `d1/strata-after` | `8419d58f9d020327d5cc00259b07d960` | this card's build: the graph path is the default, the launch-site histogram added (off) |

Rig under `d1/`, arm data outside the repo at `/home/michael/strata-xpu/d1/runs/<tag>/`.  `d1/d1_report.py`
prints one row per arm out of that arm's own logs; `d1/d1_hist.py` reads a window's histogram;
`d1/D1-EVIDENCE.txt` is the raw output.  The resident P6 server was stopped before the first arm
(`p6/p6_stop.sh`; nothing was listening on 8099) and has NOT been restarted.

## 0. Verdict

1. **The graph path is the default now** (`ce01ff3`), `STRATA_SYCL_GRAPH=0` is the off-switch, and the guards
   re-run on this binary agree with P3's: **124.57/120.49/125.71 ms per window against the pre-change binary's
   129.68/124.20/131.86 (-3.9/-3.0/-4.7% at 32K/128K/4K)**, decode **22.69 -> 23.62, 20.82 -> 21.46, 22.75 ->
   22.93 tok/s**, and the **greedy token ids are byte-identical at all three lengths** (`d87373e8…`,
   `511a89be…`, `66bf952d…` — and those are the md5s P9's arms produced on the unchanged paths).  Not a
   correctness change: a speedup.  The suite did not move: the six capture/replay tests are **6/6 with the new
   default**, and the full suite is **5 failed of 49 — the same five P3 recorded** (and the same five with
   `STRATA_SYCL_GRAPH=0`, i.e. the closure path is untouched).
2. **The spec width was swept instead of assumed, and the config of record's own `--spec 4` is the best width at
   all three lengths** (4K: 2 and 4 tie at 22.79 tok/s; 32K: 4 at 23.69 against 2's 22.28; 128K: 4 at 21.47
   against 2's 21.12).  **The card's hypothesis — that a mostly-fixed window divides when a wider one banks more
   accepted tokens — is refuted by its own sweep**: a linear fit of ms/window against tokens/window has a
   NEGATIVE intercept at 4K/32K/128K (-85.5 / -26.3 / -10.3 ms), i.e. the marginal token costs MORE than the
   average one, and the marginal cost rises at every step (4K: 44 -> 105 -> 122 ms per token; 32K: 38 -> 67 ->
   184; 128K: 45 -> 78 -> 81).  Acceptance falls at the same time (0.830 -> 0.609 at 4K), so `--spec 6`/`8` cost
   23-29% at 4K and 7-19% at 32K/128K.  The card's own counter-hypothesis is what holds.  **The variant that does
   help is the drafter's confidence floor: `--spec 4 --spec-min-p 0.7` is +5.8% at 4K and +3.3% at 32K with
   byte-identical ids** (a wash at 128K); `--mtp-max-t 2` is a negative control that reproduces `--spec 2`.
3. **The kernels are ranked by count and by time, from a new instrument** (`STRATA_LAUNCH_HIST=1` in the shim,
   because P9 measured that a chrome device trace cannot see a decode window at all).  It names launches by their
   own mangled type — `launch_multi_n<Q6KTraits, 4>#3`, `wait_flag_ge#1` — and prices each with the event's device
   timestamps.  One T=4 window is **3 810 submissions in 54 sites** (= P3's 2 504 kernels + 1 306 copies exactly),
   and **the graph and closure paths have the identical census**.  The ranking: **the native expert chain is
   65.5% of the window's device time in 493 launches** (MMVQ 40.1% + DOWN 13.9% + GU 12.2%), **the device-side
   flag handshake is 12.3% in 192 launches**, and the copies — 39% of the submissions — are **3.0%** of the time.
   QSA grows with depth (5.1% -> 11.5%), all of it in the indexer/score work.
4. **What did not help:** a wider spec (6/8), `--mtp-max-t 2`, chasing the copies, `--spec-min-p 0.7` at 128K,
   and the histogram as a routine instrument (it costs 8-9% of the window on the closure path, 1.5 ms per dumped
   window — though only 0.4% in graph mode).

## 1. Task 1 — the graph path is the default, and its guards

### 1.1 The change

`graph_mode()` in `include/strata/sycl_compat/cuda_runtime.h` used to be off unless `STRATA_SYCL_GRAPH=1`; it is
now **on unless `STRATA_SYCL_GRAPH=0`**, and the device check stays (a device without
`sycl::aspect::ext_oneapi_graph` falls back to the launch-closure replay, with a line on stderr).  The two
messages are now the right way round: the path that is NOT taken says so, and the default prints what it is
doing:

```
# the new default, as the engine prints it (every arm below)
strata/sycl: the graph path is ON (default since D1; STRATA_SYCL_GRAPH=0 turns it off) - a window's capture
  records a SYCL command_graph and cudaGraphLaunch submits it in ONE submission
# and with the off-switch (d1-guard-*-closure)
strata/sycl: STRATA_SYCL_GRAPH=0 - the launch-closure replay is used (a window costs one submission per kernel)
```

Commit **`ce01ff3`** — "D1 (t_d9ffcf38): the graph path is the default, and a launch-site histogram for the
decode window".  Two binaries are the whole A/B, and every arm names the one it ran (md5 in its own log):

* `d1/strata-before` = `9eff0675059cf7e8145ae7b4bca984a4` — the pre-change build (P9/P10's binary);
* `d1/strata-after` = `8419d58f9d020327d5cc00259b07d960` — this card's build (`build-sycl/strata`, i.e. what the
  config of record names).

### 1.2 Per-length decode, before against after

Every arm: config of record, both B70s, greedy, 256 max-new, one engine at a time, nothing else changed.  The
whole table comes out of each arm's own stderr (`d1/d1_report.py`; the raw lines are in §1.4).

| length | arm (binary) | path | windows | avg T | tokens/window | **ms/window** | **decode tok/s** | drafts accepted | **greedy ids md5** |
|---|---|---|---|---|---|---|---|---|---|
| 4K | `d1-guard-4k-before` (pre-change) | closure | 49 | 3.76 | 3.00 | **131.86** | **22.75** | 98/135 (0.726) | `66bf952d445e330c974c49e4f220b4b1` |
| 4K | `d1-guard-4k-after` (this build) | **graph (default)** | 51 | 3.59 | 2.88 | **125.71 (-4.7%)** | **22.93 (+0.8%)** | 96/132 (0.727) | `66bf952d445e330c974c49e4f220b4b1` **identical** |
| 32K | `d1-guard-32k-before` | closure | 87 | 3.48 | 2.94 | **129.68** | **22.69** | 172/216 (0.796) | `d87373e84417dab60732a95eb47666a6` |
| 32K | `d1-guard-32k-after` | **graph (default)** | 87 | 3.46 | 2.94 | **124.57 (-3.9%)** | **23.62 (+4.1%)** | 172/214 (0.804) | `d87373e84417dab60732a95eb47666a6` **identical** |
| 128K | `d1-guard-128k-before` | closure | 99 | 3.07 | 2.59 | **124.20** | **20.82** | 159/205 (0.776) | `511a89be345a6583c9bc8c580259d73e` |
| 128K | `d1-guard-128k-after` | **graph (default)** | 99 | 3.07 | 2.59 | **120.49 (-3.0%)** | **21.46 (+3.1%)** | 159/205 (0.776) | `511a89be345a6583c9bc8c580259d73e` **identical** |

P3 measured -5.2/-4.1/-2.6% per window on its own binary; this re-measurement is -4.7/-3.9/-3.0%, on a binary
that also carries the histogram.  The end-to-end tok/s is not exactly the window ratio at 4K, for P3's own
reason and it is visible in the table: the window is cheaper, the draft policy re-picks T from the measured
window cost, and the graph arm needs 51 windows for 147 tokens against the closure arm's 49 (2.88 against 3.00
tokens/window).  At 32K and 128K the window count is the same (87, 99) and the tok/s delta (+4.1%, +3.1%)
tracks the window delta.

The pre-change arms also reproduce P9's own numbers on the same pack within 0.4%: 4K 131.86 against P9's 132.38,
128K 124.20 against 124.06 — i.e. the machine state is the one the earlier cards measured.

### 1.3 The off-switch is the pre-change behaviour (post-change binary, `--graph 0`)

`d1-guard-*-closure` runs the SAME post-change binary with `STRATA_SYCL_GRAPH=0`:

| arm | ms/window | decode tok/s | ids md5 | against its pre-change arm |
|---|---|---|---|---|
| `d1-guard-4096-closure` | 131.27 | 22.85 | `66bf952d445e330c974c49e4f220b4b1` | 131.86 / 22.75 (`d1-guard-4k-before`): **-0.4%/-0.4%**, ids identical |
| `d1-guard-32768-closure` | 128.87 | 22.83 | `d87373e84417dab60732a95eb47666a6` | 129.68 / 22.69 (`d1-guard-32k-before`): **-0.6%/-0.6%**, ids identical |

So with `STRATA_SYCL_GRAPH=0` the post-change binary reads the pre-change binary's own numbers (inside the
run-to-run band the earlier cards quote), and its token ids are identical — the off-switch is the old default,
not a third behaviour.  That also makes the graph-path delta a same-machine A/B in both directions:
131.27 -> 125.71 at 4K and 128.87 -> 124.57 at 32K.

### 1.4 The raw engine lines (the acceptance asks for them)

```
# 4K, pre-change binary (closure path)
strata decode timing: 49 windows, avg T 3.76, 3.00 tokens/window, 131.86 ms/window = verify 118.33 (GPU-reach wait 93.48 + per-layer host 1.09 [plan 0.31 actq 0.25 jobs 0.26 CPU 0.00] + stage 1.27 + tail 12.84) + commit/emit 2.03 + draft 11.50; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 37.55, PCIe 0.00
# 4K, this build, graph path by default
strata decode timing: 51 windows, avg T 3.59, 2.88 tokens/window, 125.71 ms/window = verify 111.99 (GPU-reach wait 93.83 + per-layer host 1.01 [plan 0.30 actq 0.24 jobs 0.23 CPU 0.00] + stage 1.29 + tail 12.44) + commit/emit 1.91 + draft 11.80; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 35.88, PCIe 0.00
# 32K, pre-change
strata decode timing: 87 windows, avg T 3.48, 2.94 tokens/window, 129.68 ms/window = verify 114.40 (GPU-reach wait 90.63 + per-layer host 1.05 [plan 0.30 actq 0.24 jobs 0.26 CPU 0.00] + stage 1.35 + tail 12.29) + commit/emit 2.02 + draft 13.25; ... VRAM hits 34.83
# 32K, this build
strata decode timing: 87 windows, avg T 3.46, 2.94 tokens/window, 124.57 ms/window = verify 109.26 (GPU-reach wait 92.93 + per-layer host 0.99 [plan 0.29 actq 0.23 jobs 0.22 CPU 0.00] + stage 1.33 + tail 12.12) + commit/emit 1.93 + draft 13.37; ... VRAM hits 34.60
# 128K, pre-change
strata decode timing: 99 windows, avg T 3.07, 2.59 tokens/window, 124.20 ms/window = verify 109.90 (GPU-reach wait 87.30 + per-layer host 0.96 [plan 0.29 actq 0.21 jobs 0.22 CPU 0.00] + stage 1.57 + tail 11.29) + commit/emit 1.95 + draft 12.35; ... VRAM hits 30.71
# 128K, this build
strata decode timing: 99 windows, avg T 3.07, 2.59 tokens/window, 120.49 ms/window = verify 106.16 (GPU-reach wait 90.97 + per-layer host 0.93 [plan 0.29 actq 0.21 jobs 0.20 CPU 0.00] + stage 1.59 + tail 11.13) + commit/emit 1.86 + draft 12.47; ... VRAM hits 30.71
# and the submission census that explains it, one window per stage (4K)
#   pre-change binary, closure path is the default:
#     (P9's arm, same binary) 1764 (kernel 1197 memcpy 556 barrier 11) + 2066 (kernel 1307 memcpy 750 barrier 9) = 3 841
#   this build, graph path by default:
strata submit: stage window T=4 pos0=3850 layers 0..22 (23): submitted 91 (kernel 0 memset 0 memcpy 72 barrier 18 graph 1) + recorded 0; 4.0 per layer
strata submit: stage window T=4 pos0=3850 layers 23..47 (25): submitted 96 (kernel 0 memset 0 memcpy 78 barrier 17 graph 1) + recorded 0; 3.8 per layer
#   this build, STRATA_SYCL_GRAPH=0 (d1-hist-*-closed): 1764 (kernel 1197 memcpy 556) + 2066 (kernel 1307 memcpy 750) = 3 841 kernels+copies again
```

### 1.5 The suite, and what the number was before

`d1/d1_ctest.sh` (`ZE_AFFINITY_MASK=0`, oneAPI env sourced; logs under `/home/michael/strata-xpu/d1/runs/ctest/`):

| run | result |
|---|---|
| the six capture/replay tests (`gr_parity`, `qsa_parity`, `ple_parity`, `sampler_parity`, `sampler_parity_one_block`, `sampler_parity_old`, `shared_expert_parity`), **graph path at its new default** | **100% tests passed, 0 tests failed out of 6** (15.28 s) |
| the full suite, **graph path at its new default** | **90% tests passed, 5 tests failed out of 49** (113.85 s) |
| the full suite, `STRATA_SYCL_GRAPH=0` | **90% tests passed, 5 tests failed out of 49** |

The five are the pre-existing ones P3/P9 recorded — `platform_memory_test`, `elementwise_parity`,
`quantize_act_parity`, `iq_multi_parity`, `expert_multi_test` — **the number did not move**, in either path.
So P3's own acceptance list stands: the default flip is guarded by the same suite P3 used to guard the flag.

## 2. Task 2 — the spec-width sweep

`--spec 2/4/6/8` at all three lengths, the new default (graph path) and nothing else changed, `--spec-min-p 0.5`
as in the config of record.  The four numbers the card asks for are per arm, together, and `d1/d1_sweep.py`
prints them plus the fit (§2.3) and `ms/token` (= what decode tok/s is).  Raw lines: §2.4.

### 2.1 The sweep, per length

| length | `--spec` | windows | avg T | **tokens/window** | **ms/window** | ms/token | **decode tok/s** | drafts accepted | **acceptance** | vs best |
|---|---|---|---|---|---|---|---|---|---|---|
| 4K | 2 | 74 | 2.19 | 1.99 | 87.18 | 43.89 | **22.79** | 73/88 | 0.830 | **best (tie)** |
| 4K | **4** | 51 | 3.59 | 2.88 | 126.45 | 43.87 | **22.79** | 96/132 | 0.727 | **best (tie)** |
| 4K | 6 | 40 | 4.88 | 3.68 | 210.66 | 57.32 | 17.44 | 107/155 | 0.690 | -23.5% |
| 4K | 8 | 37 | 5.97 | 3.97 | 246.18 | 61.96 | 16.14 | 112/184 | 0.609 | -29.2% |
| 32K | 2 | 138 | 1.97 | 1.86 | 83.25 | 44.88 | 22.28 | 118/134 | 0.881 | -6.0% |
| 32K | **4** | 87 | 3.46 | 2.94 | 124.22 | 42.36 | **23.69** | 172/214 | 0.804 | **best** |
| 32K | 6 | 72 | 4.75 | 3.56 | 165.97 | 46.66 | 21.42 | 188/270 | 0.696 | -9.6% |
| 32K | 8 | 69 | 5.59 | 3.71 | 193.49 | 52.20 | 19.17 | 191/317 | 0.603 | -19.1% |
| 128K | 2 | 144 | 1.87 | 1.78 | 84.17 | 47.35 | 21.12 | 112/125 | 0.896 | -1.6% |
| 128K | **4** | 99 | 3.07 | 2.59 | 120.42 | 46.57 | **21.47** | 159/205 | 0.776 | **best** |
| 128K | 6 | 96 | 3.76 | 2.91 | 145.34 | 49.96 | 20.02 | 176/253 | 0.691 | -6.8% |
| 128K | 8 | 84 | 4.06 | 3.05 | 156.65 | 51.36 | 19.45 | 172/257 | 0.669 | -9.4% |

**The answer is negative, and it is the counter-hypothesis that holds.**  Banking more accepted tokens per window
does buy tokens/window (1.99 -> 3.97 at 4K, 1.86 -> 3.71 at 32K, 1.78 -> 3.05 at 128K) and the window costs more
than proportionally: the marginal cost of one more token in the window is **44.1 / 105.3 / 122.4 ms** at 4K,
**37.9 / 67.3 / 183.5 ms** at 32K and **44.8 / 77.9 / 80.8 ms** at 128K, taken in steps of width 2 -> 4, 4 -> 6,
6 -> 8 (straight from the table: (ms/win difference) / (tokens/window difference)).  Every step costs more than
the one before it.  Acceptance falls as the card's ceiling says it can (0.830 -> 0.609 at 4K, 0.881 -> 0.603 at
32K, 0.896 -> 0.669 at 128K, against the ceilings 1/(1-p) = 5.87 -> 2.56, 8.38 -> 2.52 and 9.62 -> 2.99
tokens/window — and note that at spec 8 the engine still banks 3.97/3.71/3.05 tokens per window, ABOVE the MTP's
own geometric ceiling, which is the suffix-draft path doing its job), and the two effects together leave
**`--spec 4` — the config of record's own value — the best width at all three lengths**, with `--spec 2` a tie at
4K (22.79 tok/s both) and 1.6% / 6.0% behind at 128K / 32K.

**No width moved a token id**: all four widths produced the identical greedy id list at each length
(`66bf952d…` at 4K, `d87373e8…` at 32K, `511a89be…` at 128K — the same md5s as the config-of-record arms), i.e.
the widths are a pure speed/acceptance trade, not a correctness change.  (The 4K widths also hit the same
147-token stop, so the comparison is like for like.)

### 2.2 The hypothesis, priced

The card's hypothesis needs the window to have a large per-window FIXED part, because that is the part a wider
window divides.  The measurement says the fixed part is not there to divide: a linear fit of ms/window against
tokens/window over each length's four arms gives a NEGATIVE intercept at all three lengths —

```
4K:    ms/window = -85.48 + 80.86 * tokens/window   (R^2 0.957)
32K:   ms/window = -26.31 + 55.69 * tokens/window   (R^2 0.940)
128K:  ms/window = -10.28 + 52.30 * tokens/window   (R^2 0.981)   [spec 2/4/6; with spec 8: -1.61 + 51.23, R^2 0.986]
```

i.e. the marginal token costs more than the average one, so there is no fixed cost being amortised anywhere in
the measured range: the window's cost is convex in the tokens it verifies, and the card's premise (a window is
mostly per-window fixed cost) does not hold for the quantity a wider window would divide.

Where the per-token cost actually lands, arm by arm (the engine's own parts, 4K):

| `--spec` | ms/window | verify | of which GPU-reach wait | tail | draft | VRAM hits/layer-window |
|---|---|---|---|---|---|---|
| 2 | 87.18 | 80.87 | 68.71 | 8.62 | 4.59 | 21.89 |
| 4 | 126.45 | 112.63 | 93.87 | 12.44 | 11.92 | 35.88 |
| 6 | 210.66 | 190.94 | 123.03 | 14.79 | 17.61 | 48.75 |
| 8 | 246.18 | 220.82 | 150.20 | 17.30 | 23.11 | 59.73 |

**The GPU-reach wait — the card's "71% of the window" — is itself per-token, not per-window**: it grows 68.71 ->
150.20 ms as the window verifies 1.99 -> 3.97 tokens, because the wait is the host waiting for the GPU to finish
each layer's rows, and more rows per layer is more GPU work per layer.  `VRAM hits` (the resident experts'
compute per layer-window) grows with it (21.89 -> 59.73).  The draft phase grows too (4.59 -> 23.11 ms).  So the
lever the card proposed — divide a fixed cost by banking more tokens — has nothing to divide: the window's cost
is ~proportional to the tokens it verifies, and the marginal token gets *more* expensive as the window widens.

### 2.3 The variants, at the best width (4)

Both were run at `--spec 4` (the sweep's best width = the config of record's value), one change each, all three
lengths.  `--spec-min-p` is the policy's confidence floor: with it set, the window starts at T=1 and grows while
the drafter's probability for the next position is `>=` the floor (`generate.cpp:6194-6197`), so 0.7 is a
"shorter, surer window" policy than the record's 0.5.  `--mtp-max-t 2` caps the MTP's own windows at 2 tokens
(`generate.cpp:6179`), leaving the longer windows to the suffix-draft path.

| variant (all at `--spec 4`) | length | windows | avg T | tokens/window | ms/window | **decode tok/s** | vs the config of record | acceptance | ids md5 |
|---|---|---|---|---|---|---|---|---|---|
| `--spec-min-p 0.7` | 4K | 54 | 3.04 | 2.72 | 112.86 | **24.12** | **+5.8%** (22.79) | 0.845 (0.727) | identical |
| `--spec-min-p 0.7` | 32K | 92 | 3.01 | 2.78 | 113.66 | **24.48** | **+3.3%** (23.69) | 0.886 (0.804) | identical |
| `--spec-min-p 0.7` | 128K | 118 | 2.36 | 2.17 | 101.84 | 21.30 | -0.8% (21.47) | 0.875 (0.776) | identical |
| `--mtp-max-t 2` | 4K | 74 | 2.22 | 1.99 | 87.67 | 22.66 | -0.6% (22.79) | 0.811 (0.727) | identical |
| `--mtp-max-t 2` | 32K | 138 | 1.97 | 1.86 | 83.17 | 22.30 | -5.9% (23.69) | 0.881 (0.804) | identical |
| `--mtp-max-t 2` | 128K | 144 | 1.87 | 1.78 | 84.81 | 20.96 | -2.4% (21.47) | 0.896 (0.776) | identical |

**`--spec-min-p 0.7` is the one that helps, and it helps for the reason the sweep gives**: it buys the same or
better acceptance (0.845 vs 0.727 at 4K, 0.886 vs 0.804 at 32K) on slightly SMALLER windows (2.72 vs 2.88
tokens), which is exactly the direction §2.1's marginal-cost curve rewards — **+5.8% / +3.3% decode tok/s at
4K / 32K with byte-identical greedy ids** (a speedup, not a correctness change).  At 128K it is a wash (-0.8%,
inside the run-to-run band; its window is 101.84 ms against 120.42 and its tokens/window 2.17 against 2.59, so
the two effects cancel there).  `--mtp-max-t 2` is the negative control: it just makes the MTP windows 2, i.e.
it reproduces `--spec 2`'s numbers (22.30 vs 22.28 at 32K) and buys nothing.

### 2.4 The raw engine lines (the acceptance asks for at least one length)

```
# 4K, the four widths (spec 2 / 4 / 6 / 8), the graph path (the new default)
strata decode timing: 74 windows, avg T 2.19, 1.99 tokens/window, 87.18 ms/window = verify 80.87 (GPU-reach wait 68.71 + per-layer host 0.77 [plan 0.26 actq 0.15 jobs 0.13 CPU 0.00] + stage 1.06 + tail 8.62) + commit/emit 1.71 + draft 4.59; ... VRAM hits 21.89, PCIe 0.00
strata decode timing: 51 windows, avg T 3.59, 2.88 tokens/window, 126.45 ms/window = verify 112.63 (GPU-reach wait 93.87 + per-layer host 1.02 [plan 0.29 actq 0.24 jobs 0.23 CPU 0.00] + stage 1.32 + tail 12.44) + commit/emit 1.89 + draft 11.92; ... VRAM hits 35.88, PCIe 0.00
strata decode timing: 40 windows, avg T 4.88, 3.68 tokens/window, 210.66 ms/window = verify 190.94 (GPU-reach wait 123.03 + per-layer host 1.26 [plan 0.33 actq 0.32 jobs 0.32 CPU 0.00] + stage 1.49 + tail 14.79) + commit/emit 2.11 + draft 17.61; ... VRAM hits 48.75, PCIe 0.00
strata decode timing: 37 windows, avg T 5.97, 3.97 tokens/window, 246.18 ms/window = verify 220.82 (GPU-reach wait 150.20 + per-layer host 1.50 [plan 0.36 actq 0.41 jobs 0.41 CPU 0.00] + stage 1.74 + tail 17.30) + commit/emit 2.25 + draft 23.11; ... VRAM hits 59.73, PCIe 0.00
# and the request's own line, same four arms (the acceptance is on it)
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15164 ms (252.7 tok/s), 147 generated in 6451 ms (22.8 tok/s), drafts accepted 73 of 88, 0 checkpoints
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15202 ms (252.1 tok/s), 147 generated in 6449 ms (22.8 tok/s), drafts accepted 96 of 132, 0 checkpoints
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15277 ms (250.8 tok/s), 147 generated in 8426 ms (17.4 tok/s), drafts accepted 107 of 155, 0 checkpoints
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15194 ms (252.2 tok/s), 147 generated in 9109 ms (16.1 tok/s), drafts accepted 112 of 184, 0 checkpoints
```

### 2.5 MTP's state, per arm

The drafter is `~/strata-xpu/mtp/rt` in every arm (P7's canonical build: `draft_vocab.bin` 425,196 B present,
`strata mtp: draft head over 106299 tokens (212.9 MiB)`, `795 MiB of VRAM`).  Its state is visible in every
arm's own line: the draft phase (`+ draft X ms`) and the acceptance (`drafts accepted A of O`).  P7 measured
11.65 ms/window for the draft phase at 4K; the config of record's arm here reads **11.50-11.92 ms** (spec 4),
and the sweep prices the draft phase directly: **4.59 ms at spec 2, 11.92 at 4, 17.61 at 6, 23.11 at 8** — i.e.
the drafter is a real part of the widening window, not a constant, and at spec 6/8 it is 8-9% of the window
while the extra tokens it buys are worth less than they cost.


## 3. Task 3 — the kernels, ranked by count and by time

### 3.1 What the instrument is, and why it had to be built here

P3 counted a window's submissions (2 504 kernels + 1 306 copies at 4K) but not WHICH kernels, and P9 measured
that the obvious instrument cannot see a decode window at all: a chrome DEVICE trace stalls the verify window at
layer 1 (both graph paths, four arms), because the window is a per-layer host↔device flag handshake and device
instrumentation breaks it.  So the census had to come from inside the shim, where every launch passes (D2) —
and `STRATA_LAUNCH_HIST=1` is what this card added there.  Each launch is named from its own launcher lambda's
type via `__cxa_demangle`:

```
strata::kernels::(anonymous namespace)::launch_multi_n<…Q6KTraits, 4>(void const*, void const*, float*, int, int,
  strata::sycl_compat::stream_t*)#3        ->  `launch_multi_n<Q6KTraits, 4>#3`
```

i.e. the enclosing host wrapper plus the index of the launch site inside it, which identifies the kernel that
site launches without renaming anything (the port launches unnamed lambdas on purpose).  The per-submission
microseconds come from the events' own device timestamps, read at the dump after the window's tail sync — which
is why the µs column exists on the closure path (each kernel a real submission) and not in graph mode (a
recorded node has no device timestamp until the graph executes; the counts are complete in both).

### 3.2 The census: the two paths run the same kernels, launch for launch

| arm (4K, T=4 window) | stage 0 | stage 1 | total | sites |
|---|---|---|---|---|
| `d1-hist-4096-closed` (closure path): 1 197 kernels + 556 copies | 1 753 | 2 057 | **3 810** | 54 |
| `d1-hist-4096-graph` (graph path, the window that RECORDS): | 1 753 | 2 057 | **3 810** | 54 |

3 810 = P3's 2 504 kernels + 1 306 copies exactly (the histogram counts kernels and copies; the 31 barriers are
not counted, which is the 3 841 - 3 810 difference).  Same at 32K (3 810 / 54 sites, both paths).  So the graph
path is a re-replay of the identical kernel set — which is why §1's token ids are identical.

### 3.3 The window's kernels at 4K, by device microseconds (closure path, window 2, T=4)

`d1/d1_hist.py` (`--auto` picks the window with the most counted submissions, i.e. the T=4 window here);
full per-site tables and CSVs for all lengths: `d1/D1-HISTOGRAMS.txt`, `/home/michael/strata-xpu/d1/runs/<arm>/sites.csv`.

```
#    count     us         share   kernel (sites merged)
1    129       31511.7    23.8%   launch_multi_n<Q6KTraits, 4>
2    144       16716.0    12.6%   wait_flag_ge
3    78        15317.3    11.6%   launch_down<20>
4    288       12754.9    9.7%    launch_multi
5    40        6393.2     4.8%    launch_gu<22>
6    47        6383.1     4.8%    launch_multi_n<SmallTraits<IQ4NLBlock, 4>, 4>
7    34        5707.4     4.3%    launch_gu<18>
8    35        5292.5     4.0%    launch_multi_n<Q5KTraits, 4>
9    42        5091.4     3.9%    launch_multi_n<IQ4XSTraits, 4>
10   47        4260.0     3.2%    launch_multi_n<Q4KTraits, 4>
11   20        3313.2     2.5%    launch_gu<21>
12   18        2982.8     2.3%    launch_down<42>
13   41        2318.1     1.8%    copy_from_mapped
14   120       2050.0     1.6%    bf16_gemv_fp32_mmvf_multi
15   24        2005.9     1.5%    qsa_decode_attn_batch
16   1306      1183.3     0.9%    <memcpy>
    3810      132153.6            (132.15 ms of the 141.6 ms window, both cards summed)
# -- top by COUNT --
1    1306      1183.3     0.9%    <memcpy>
2    288       12754.9    9.7%    launch_multi                       <- the fused GDN recurrence
3    193       711.0      0.5%    native_quantize_q8_1
4    192       527.3      0.4%    native_expert_grouped
5    144       16716.0    12.6%   wait_flag_ge
6    144       165.9      0.1%    shared_expert_multi
7    129       31511.7    23.8%   launch_multi_n<Q6KTraits, 4>
8    120       2050.0     1.6%    bf16_gemv_fp32_mmvf_multi
9    78        15317.3    11.6%   launch_down<20>
10   54        238.1      0.2%    copy_i32_from_mapped
```

### 3.4 The same window rolled up into families — the ranking that names the targets

`d1/d1_families.py` (kernel -> family, counts and µs summed):

| family (what it is) | 4K count | 4K µs | 4K share | 32K share | 128K share |
|---|---|---|---|---|---|
| **expert MMVQ** (`launch_multi_n<…>`: the native expert mat-vec, one launch per expert group) | 301 | 52 501 | **40.1%** | **39.8%** | **37.9%** |
| **expert DOWN** (`launch_down<N>`) | 96 | 18 246 | **13.9%** | **13.7%** | **13.1%** |
| **flag handshake, device side** (`wait_flag_ge` + `doorbell_publish`) | 192 | 16 032 | **12.3%** | **12.2%** | **10.7%** |
| **expert GU** (`launch_gu<N>`) | 96 | 15 935 | **12.2%** | **12.0%** | **11.4%** |
| GDN recurrence (`launch_multi` = fused_gr, `gdn_*`, `shared_expert_multi`) | 572 | 14 687 | 11.2% | 11.1% | 10.5% |
| QSA attention/indexer (`qsa_*`, `kv_*`, `bf16_gemv*`) | 328 | 6 726 | 5.1% | 6.2% | **11.5%** |
| copies/mapped staging (`<memcpy>`, `*_from_mapped`) | 1 449 | 3 903 | 3.0% | 3.0% | 2.9% |
| expert grouping/quantize (`native_expert_grouped`, `native_quantize_q8_1`, router, combine) | 577 | 2 322 | 1.8% | 1.8% | 1.7% |
| other | 199 | 465 | 0.4% | 0.4% | 0.3% |
| (total counted submissions / µs) | 3 810 / 130.8 ms | | | 3 810 / 132.5 ms | 3 834 / 138.2 ms |

The three lengths are near-identical except in one family, and it is the same one P9's stage profiler found:
**QSA grows with depth (5.1% -> 6.2% -> 11.5%)** — at 128K `qsa_block_scores` alone is 12 launches and 5 564 µs
(the index/score work over the paged KV, which is the depth-growing part), while the expert families shrink in
share only because the total grows.

### 3.5 The two or three targets this names

1. **The native expert MMVQ chain — 65.5% of the window's counted device time in 493 launches**
   (`launch_multi_n` 39.8% + `launch_down` 13.7% + `launch_gu` 12.0%).  Its shape is one launch per expert GROUP
   per layer (the T=4 window's NCOLS=4 templates), so it is the batching/fusion target with evidence behind it:
   the work is already grouped per expert (P10 measured 10 experts x 48 layers per window), and the launches are
   the per-group granularity, not the per-token one.  Fusing `gu`+`down` into one grouped kernel, or raising
   NCOLS, is where the 65% is.  This is also the family the machine's per-token cost lives in — which is why
   §2's "wider window" is not free: those kernels' rows grow with T.
2. **The device-side flag handshake — 192 launches, 12.2% of the time** (`wait_flag_ge` alone is the 12.6% row
   in §3.3 at 16.7 ms, 144 launches = 48 layers x 2 stages + 48 `doorbell_publish`).  This is P9's "GPU-reach
   wait" seen from the device: one spin kernel per layer per group, whose 116 µs average is the GPU waiting for
   the host to publish the next flag.  Fewer, coarser handshakes (G = 1, or a ring that carries several layers)
   is the second target, and it is worth more than everything in the copies family put together.
3. **Not the copies**, and this is a measured negative that saves the work: 1 306 `<memcpy>` + 143 mapped-staging
   launches are **39% of the window's submissions and 3.0% of its time** (0.9 µs per copy).  P3 reached the same
   conclusion from the counters ("152 copies + 31 barriers = 0.2%"); the histogram prices it directly.  The same
   goes for the grouping/quantize bookkeeping (577 launches, 1.8%).

### 3.6 What the instrument costs, its own crash, and the fix

**Its price, measured against the same binary with the histogram off (both on the closure path, so the only
difference is the instrument):**

| length | with `STRATA_LAUNCH_HIST=1` | same binary, hist off (`d1-guard-*-closure`) | cost |
|---|---|---|---|
| 4K | 143.14 ms/window, 20.96 tok/s (`d1-hist-4096-closed`) | 131.27 ms/window, 22.85 tok/s | **+11.87 ms/window, +9.0%** |
| 32K | 139.47 ms/window, 21.59 tok/s (`d1-hist-32768-closed`) | 128.87 ms/window, 22.83 tok/s | **+10.60 ms/window, +8.2%** |
| 128K | 133.60 ms/window, 19.35 tok/s (`d1-hist-131072-closed`) | (no same-binary control ran at 128K) | against the pre-change closure arm's 124.20: +7.6%, cross-binary |

The cost is the dump: it asks the driver for two device timestamps per retained submission (3 810 x 2 per
window) and does that for `STRATA_LAUNCH_HIST_WINDOWS=8` windows per request — ~7 600 driver round trips per
window, which is where 8-10 ms goes.  Every histogram number above is therefore a count from an instrumented
arm, and the µs are the closure path's; the SPEED numbers in §1 and §2 are from arms with the histogram off.

**And it crashed once, which is in the record as a measurement too.**  `d1-hist-131072-graph` (the 128K
graph-mode arm, first attempt) died in the PREFILL with `double free or corruption (out)`, exit -6, 2 PP chunks
of 16 in, no window yet (its log is kept at `/home/michael/strata-xpu/d1/runs/d1-hist-131072-graph/log.txt`,
its `hist.txt` is empty).  Two defects were behind it, both now fixed in the shipped instrument:

1. **the table was not thread-safe.**  The engine does submit from more than one host thread (the window's
   finalizer thread, `verify.cpp:433`; the prompt threads, `generate.cpp:4574/4691`), and the first form let
   whichever thread was launching push into the shared `std::vector` — a torn vector, which is exactly what
   glibc's "double free or corruption" reports.  Fixed with a `std::mutex` held by every table mutation.
2. **events were retained outside a window.**  A 129 024-token prefill is ~680 000 launches at this port's ~5
   per token, and the first form kept a `sycl::event` for every one of them (a prefill's worth of LIVE driver
   events, growing to the 400 000-entry cap).  Nothing outside a window's slice is ever priced, so the shipped
   `hist_take` retains only while a window is open and only counts otherwise.

The fix is an instrument-only change (nothing on the default path: `hist_take`/`hist_count` are called only
under `STRATA_LAUNCH_HIST`), and the 128K graph-mode arm was re-run with it — §3.7.

### 3.7 The histogram at the other lengths, the census in both paths, and the re-run

**The census is the same in both paths at all three lengths** (the window the engine spends its time in, T=4):

| length | closure path (`d1-hist-*-closed`) | graph path, the recording window (`d1-hist-*-graph`) |
|---|---|---|
| 4K | 1 753 + 2 057 = **3 810** submissions, 54 sites | 1 753 + 2 057 = **3 810**, 54 sites |
| 32K | 1 753 + 2 057 = **3 810**, 54 sites | 1 753 + 2 057 = **3 810**, 54 sites |
| 128K | 1 765 + 2 069 = **3 834**, 56 sites | 1 763 + 2 071 = **3 834**, 56 sites |

**The µs by length** (closure path, `d1/D1-HISTOGRAMS.txt` has the full per-site tables and CSVs):

| kernel | 4K µs | 32K µs | 128K µs |
|---|---|---|---|
| `launch_multi_n<Q6KTraits, 4>` | 31 512 | 31 524 | 31 319 |
| `wait_flag_ge` | 16 716 | 15 865 | 14 454 |
| `launch_down<20>` | 15 317 | 15 167 | 15 129 |
| `launch_multi` (fused GDN recurrence) | 12 755 | 12 706 | 12 594 |
| `launch_multi_n<SmallTraits<IQ4NLBlock,4>,4>` | 6 383 | 6 377 | 6 315 |
| `launch_gu<22>` | 6 393 | 6 343 | 6 201 |
| `launch_gu<18>` | 5 707 | 5 623 | 5 620 |
| `launch_multi_n<Q5KTraits,4>` | 5 293 | 5 278 | 5 229 |
| `launch_multi_n<IQ4XSTraits,4>` | 5 091 | 5 165 | 5 065 |
| `launch_multi_n<Q4KTraits,4>` | 4 260 | 4 258 | 4 306 |
| **`qsa_block_scores`** (the depth-growing indexer) | 49 | 1 530 | **5 564** |
| `<memcpy>` (1 306 of them) | 1 183 | 1 178 | 1 218 |
| counted total | 132.2 ms | 132.5 ms | 138.2 ms |

**The re-run, and what the fix cost.**  `d1-hist-131072-graph` was re-run with the fixed instrument
(`d1/strata-after-histfix`, `72ec74ae…`) and **completes**: 99 windows, 120.85 ms/window, 21.40 tok/s, ids
`511a89be…` — the config of record's own numbers at 128K (120.42 / 21.47), and its census is the table above.
Two further things that number gives:

* **in graph mode the histogram costs +0.4%** (120.85 against the non-instrumented 120.42 ms/window) — because
  only the recording window's launches are retained and priced, a few thousand events per request, against the
  closure path's ~3 810 events per window for 8 windows;
* **the closure path's cost is per DUMP and measured**: `d1-hist-4096-w1` (the same closure arm with
  `STRATA_LAUNCH_HIST_WINDOWS=1`) reads **133.00 ms/window against the 8-window arm's 143.14 and the
  no-histogram 131.27** — i.e. **~1.5 ms per dumped window** (3 810 submissions x 2 driver timestamp queries),
  0.3 ms of fixed cost, and 8 dumps is what makes the total +9%.


## 4. What did not help

Everything here is a measured negative, with the number that closes it:

1. **A wider spec window.**  `--spec 6` / `--spec 8` at the config of record: **-23.5% / -29.2% at 4K**,
   **-9.6% / -19.1% at 32K**, **-6.8% / -9.4% at 128K** (decode tok/s).  The window's per-token cost is convex
   (§2.2), so the tokens a wider window banks cost more than they save.  **The card's fixed-cost hypothesis is
   refuted by its own sweep**: the fit's intercept is negative at all three lengths (-85.5 / -26.3 / -10.3 ms),
   i.e. there is no per-window fixed part for a wider window to divide.  The card's own counter-hypothesis — that
   longer drafts lose acceptance faster than they add tokens — is what the numbers show (acceptance 0.830 ->
   0.609 at 4K while tokens/window only go 1.99 -> 3.97).
2. **`--mtp-max-t 2`** (the MTP capped at 2 tokens with `--spec 4`): **-0.6% at 4K** (a wash: 22.66 against
   22.79), **-5.9% at 32K**, **-2.4% at 128K**.  It is `--spec 2` by another name (32K: 22.30 against 22.28), so
   it neither loses nor gains anything the width sweep did not already price.  What it does NOT do is what it
   was tried for: with the MTP capped, the suffix-draft path does not make up the difference — the average T
   falls to 1.87-2.22 and the window shrinks with it.
3. **Chasing the copies.**  Already refuted by P3 from the counters and re-priced here: 1 306 `<memcpy>` + 143
   mapped-staging launches = 39% of the window's submissions and **3.0% of its counted device time** (0.9 µs per
   copy).  The same for the grouping/quantize bookkeeping (577 launches, 1.8%) — a lot of launches, very little
   time.
4. **The `--spec-min-p 0.5 -> 0.7` direction is not a win at depth** — it is a wash at 128K (-0.8%) though it is
   a clear +5.8%/+3.3% at 4K/32K.  It is listed here as much as in §2.3 because "helps at short contexts" is not
   the same claim as "helps".
5. **The histogram as a routine instrument.**  It costs **+8-9% of the window** when on (§3.6), so it is a
   one-off census, not something to leave on; and its first form crashed the 128K arm (a torn table — the fix is
   a mutex and a retention rule, §3.6).

## 5. Not validated, and the state left behind

**Not validated**

* **No variance estimate.**  One arm per configuration, as in P1b/P3/P9/P10.  The arm-to-arm band on this rig has
  historically been a few tenths of a percent; the deltas quoted here are 0.4% (the closure control against the
  pre-change binary) to 29% (spec 8), so the small ones are inside the band and are labelled as such — the 4K
  `--spec-min-p 0.7` +5.8% and the graph path's +3.0..+4.7% are well clear of it, the 128K -0.8% is not.
* **The histogram's microseconds are the closure path's, and only for kernels, memcpys and memsets.**  A recorded
  node has no device timestamp until the graph executes, so the graph-mode arms carry counts only (their census
  is identical, launch for launch — §3.2); and 31 barriers per window are not counted (3 841 - 3 810), nor are
  the event/host-fn submissions.
* **The prefill's ~680 000 launches at 128K is an estimate**, not a count: it scales P9's measured 20 336 dequant
  launches for a 3 832-token prompt (~5.3 per token) to 129 024 tokens.  The instrument never dumped a prefill
  slice (its windows start at the first decode window).
* **The 128K instrument cost is cross-binary** (no same-binary closure control ran at 128K): 133.60 against the
  pre-change binary's 124.20.  The 4K/32K figures are same-binary.
* **`--spec-min-p 0.7`'s win is measured, not landed.**  It moves no token id at any length, but making it the
  config of record is a maintainer's call (it also changes what `strata-sycl-iq3s.json` means for every future
  card) — this card only prices it: +5.8%/+3.3% at 4K/32K, a wash at 128K.
* **The instrument's thread-safety fix is not stress-tested.**  The evidence is that the arm that crashed at 128K
  completes with it (and that the 4K arms still produce the same census), not a race detector: no TSan run was
  made, and the mutex's cost when the histogram is ON was not separately measured (it is inside §3.6's +8-9%).
* **The two linear fits are over four points each.**  What is robust is their NEGATIVE intercept (no fixed cost
  to amortise) across three lengths and two independent parts of the window (the wait and the experts' VRAM
  hits both grow with T); the intercept's VALUE is an extrapolation and is not quoted as a measurement.
* **The 4K arms stop at 147 tokens** (a stop token) in every arm, including the sweep's, so their tok/s is over
  147 tokens rather than the 256 the protocol of record asks for; 32K and 128K run the full 256.  The protocol's
  floor is 256 "usage-counted" tokens and the 4K prompt's text ends there — P9's own 4K arms read 147 as well.
* **Nothing is pushed**: the origin (`github.com/Niko1221/Strata`) has no `sycl-xpu` branch.

**The state left behind**

* **No engine and no server are running**: the P6 resident server was stopped before the first arm (nothing was
  listening on 8099 and `p6/p6_stop.sh` had nothing to kill) and it was **NOT restarted**, so 8099 is free and
  both cards are free.  Every arm ran one engine at a time with `ZE_AFFINITY_MASK` unset (two B70s,
  `--layer-split auto`), and the closure-control/histogram arms that need one path set it per arm.
* The tree is on branch `sycl-xpu` with this card's commits; `build-sycl/strata` holds the shipped binary
  `72ec74ae…` (graph path default + the histogram, off), which is also `d1/strata-after-histfix`.  The binaries
  the arms ran are on disk (`d1/strata-before` `9eff0675…`, `d1/strata-after` `8419d58f…`) and are NOT committed
  (they are 35 MB each; `d1/.gitignore`).
* The arm data (logs, timelines, histograms, CSVs) is at `/home/michael/strata-xpu/d1/runs/` — outside the repo,
  as in P9/P10 — including the crashed arm, kept as `d1-hist-131072-graph-crash`.
* `p6/server.pid` was already gone (P9 removed it); nothing else on this box was touched.
