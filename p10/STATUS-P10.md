# P10 — one card with the CPU computing the expert overflow, and (if it holds) two one-card instances

Card t_1d052912. Rig: `p10/` in this repo, arms in `/home/michael/strata-xpu/p10/runs/<tag>`, raw output per arm
in `p10/P10-EVIDENCE.txt`. Engine `build-sycl/strata` at HEAD `83d1e82` (md5 9eff0675059cf7e8145ae7b4bca984a4),
pack IQ3_S `ed59f920`, faithful drafter `~/strata-xpu/mtp/rt`, MTP on, greedy, `--prompt-cache 0`, one card or the
two-card split, `--stats` for every number below.

**The answer in one line:** the CPU expert path is real and it is busy — on one B70 `--expert-cache auto` keeps
**52-54%** of the 24,576 profiled experts in VRAM and the pool computes the rest on **14.6-15.8 cores** during
decode — and it costs **-4.9% decode at 32K** and **-15.5% at 128K** against the two-card split, but it costs
**-25.8% / -40.3% of prefill**, comes with 8.7-15.3 GB of file reads per request, and the answers are **no longer
bit-identical** to the model of record (the CPU rounds differently). Two one-card instances, one per card, do
**not** double the workers: they reach **24.5 tok/s** of aggregate decode against **22.6** for one two-card
instance (+8%, not +100%), each request runs **1.8x slower**, and for two concurrent 32K prompts the pair takes
**353.5 s against 303.5 s** — the parallelism is eaten by the shared RAM path, so the honest answer to Mike's
"might as well run two instances" is **no**.

## 1. What was run, and the mask rule that is the only difference between the arms

| arm | cards | `ZE_AFFINITY_MASK` | `--layer-split` | prompt tokens | generated |
|---|---|---|---|---|---|
| `1c-4k` | card 0 | **0** | not passed | 3,832 (needle) | 150 (stop) |
| `2c-4k` | both | unset | `auto` | 3,832 | 147 (stop) |
| `1c-32k` | card 0 | **0** | not passed | 32,256 | 256 |
| `2c-32k` | both | unset | `auto` | 32,256 | 256 |
| `1c-128k` | card 0 | **0** | not passed | 129,024 | 256 |
| `2c-128k` | both | unset | `auto` | 129,024 | 256 |
| `1c1-4k` | card **1** | **1** | not passed | 3,832 | 150 (stop) |
| `2i-32k-9`, `2i-32k-19`, `1x2-32k` | see §6 | per instance | `auto` (1x2 only) | 32,252 (text) | 256 |

**On the mask (the card asks that this be stated):** for a *single-card instance* the mask is the correct mechanism
— it is how one engine is given one card — and `--layer-split` is then not passed at all, because there is nothing
to split. The rule that forbids the mask (PLAN §11 U11) is about a **split run inside one engine**, where hiding the
second card from the split search is the failure mode. The two cases are different, and no arm here carries the mask
into a two-card run: every two-card arm was checked to see both devices (`layer split auto: K=23`, cards hold
layers 0-22 / 23-47).

Every arm is the config of record otherwise: `--kv int8 --expert-cache auto --expert-profile data/expert-profile.bin
--mmap-experts --prefill auto --spec 4 --spec-min-p 0.5 --kv-resident 32768 --max-context <length>`. The 4K prompt
is the needle one (its answer is short, so the model stopped by itself at 150/147 tokens — the same early stop on
both card counts, and the per-window decode figures are averaged over 49-50 windows). 32K and 128K produced the
full 256 tokens on both. Per-arm commands: `p10/P10-EVIDENCE.txt`.

## 2. What produced the spill, and the residency it produced (task 1)

`--expert-cache auto` did it; **no explicit cache size was needed** (the card's fallback — `--expert-cache <N>`
— was not used, so nothing here is capped by hand). The two-step is visible in the engine's own load lines: `auto`
first grants slots of the **largest** blob, then the native-pack per-pair sizing re-cuts them, which is why the
final count is higher than the auto line:

| arm | `expert cache auto:` | final cache | residency of 24,576 | cache MiB | VRAM free at READY |
|---|---|---|---|---|---|
| `1c-4k` | 26.07 GiB free, 700 MiB reserved, +218 MiB draft head -> 10,153 slots | **13,258** | **54.0%** | 25,778 | 315 MiB |
| `1c1-4k` | 26.81 GiB free -> 10,450 slots | **13,649** | **55.5%** | 26,531 | 317 MiB |
| `1c-32k` | 25.47 GiB free -> 9,911 slots | **12,943** | **52.7%** | 25,164 | 527 MiB |
| `1c-128k` | 25.31 GiB free -> 9,845 slots | **12,860** | **52.3%** | 24,996 | 537 MiB |
| `2c-4k` | card0 27.46 GiB -> 10,797; card1 12,800 | **24,576** | **100%** | 47,962 | 2,610 MiB |
| `2c-32k` | — | **24,576** | **100%** | 47,962 | 2,440 MiB |
| `2c-128k` | — | **24,576** | **100%** | 47,962 | 2,218 MiB |

The draft head's 218 MiB is visible in the auto arithmetic, so it is inside this compare. The one card is asked to
hold the *same* profile with half the VRAM, and it lands at 12,860-13,258 pairs, i.e. the single-card spill is
forced by the cache sizing the card asked for, not by an artificial cap.

## 3. The CPU path is real, not nominal (task 2)

**The engine's own counters.** `CPU experts X (Y entries)` is per layer-window over 48 layers; two cards report
exactly zero in every arm:

| arm | per layer-window | decode cache hit rate | host ms/window [breakdown] |
|---|---|---|---|
| `1c-4k` | **CPU experts 1.98 (2.43 entries)**, VRAM hits 35.57 | 93.6% (85,371 / 91,200) | 25.40 [plan 0.54 actq 0.29 **jobs 15.70 CPU 4.13**] |
| `1c1-4k` | **CPU experts 1.71 (2.10 entries)**, VRAM hits 34.06 | 94.2% (85,005 / 90,240) | 22.33 [plan 0.35 actq 0.27 **jobs 13.36 CPU 3.76**] |
| `2c-4k` | **CPU experts 0.00 (0.00 entries)**, VRAM hits 37.55 | 100.0% (88,320 / 88,320) | 1.08 [plan 0.31 actq 0.25 jobs 0.26 CPU 0.00] |
| `1c-32k` | **CPU experts 1.22 (1.45 entries)**, VRAM hits 33.84 | 95.9% (141,322 / 147,360) | 16.52 [plan 0.44 actq 0.26 **jobs 9.46 CPU 2.90**] |
| `2c-32k` | **CPU experts 0.00 (0.00 entries)**, VRAM hits 35.06 | 100.0% (146,400 / 146,400) | 1.04 [plan 0.30 actq 0.23 jobs 0.25 CPU 0.00] |
| `1c-128k` | **CPU experts 1.75 (2.15 entries)**, VRAM hits 29.83 | 93.3% (144,636 / 155,040) | 22.56 [plan 0.43 actq 0.25 **jobs 13.69 CPU 3.39**] |
| `2c-128k` | **CPU experts 0.00 (0.00 entries)**, VRAM hits 30.71 | 100.0% (145,920 / 145,920) | 0.96 [plan 0.30 actq 0.21 jobs 0.22 CPU 0.00] |

`jobs` is the host waiting on / dispatching the pool and `CPU` is the host thread's own expert compute (the bracket
sums to the window's host total). On one card the host-side expert work is **16.5-24.3 ms a window**; on two cards
it is **1.0 ms**. 1.22-1.98 experts per layer-window is well past the 1.70 the card called plenty of evidence.

**The workers are busy.** `top -b -H`, busiest sample of the `1c-4k` request (13:08:51, during decode):

```
[top] top-H.txt: busiest sample 13:08:51: 74 engine threads, 20 at >=50% of one core, 21 above 1%
[top]    tid  1104204   92.0% cpu     18.10s cpu-time     <- the host thread
[top]    tid  1104300   56.0% cpu      0.58s cpu-time
[top]    tid  1104303   56.0% cpu      0.58s cpu-time
... 19 pool workers at 50-56% (the full sample is in P10-EVIDENCE.txt)
```

and by phase, from per-thread `/proc/<pid>/task/<tid>/stat` deltas (`p10_threads.py`; the phase boundary is the
engine's own `DONE` line):

| arm | load | prefill | decode | decode, threads >=50% |
|---|---|---|---|---|
| `1c-4k` | — | 14.5 s, **0.89 cores** (1 thread) | 7.7 s, **15.57 cores** | 20 of 74 |
| `1c-32k` | — | 123.8 s, **0.99 cores** (1 thread) | 12.3 s, **15.78 cores** | 20 of 75 |

**Prefill does not use the pool at all** on one card (0.89-0.99 cores: the host loop and its I/O threads), and
decode uses **15.6-15.8 of the 20 cores** — that is the whole point of the arm, and it is a finding in its own
right for anyone tuning prefill.

**The instrument was cross-checked, because two tools disagree on this process.** `/proc/<pid>/stat` for the
process and the sum of its per-thread files agree exactly during decode (16.42-16.95 cores aggregate against
16.34-16.91 cores summed, 20 threads above 0.5 core, `p10_threadcheck.py`), and `pidstat -t`'s own **aggregate**
row in the same second reads `1,782%` (17.82 cores) next to a per-thread maximum of `90.1%` — i.e. pidstat's
thread rows are consistent with top and with `/proc` once the whole sample is read (the pool threads are not the
first rows of its output; reading the first eight rows of a 48-row sample shows only parked threads and reads as
"0.00%" — the first pass at this did exactly that and was wrong).

**AVX2, per the operator note:** this host has no AVX-512 and the engine says so at load —
`strata generate: this CPU has no AVX-512: the expert kernels run on AVX-2 (multi-token for the i-quant gate/up
rows)` — so the AVX2 kernels are the ones under test and `STRATA_FORCE_AVX2=1` would pin nothing new here. Not run.

## 4. Depth, VRAM, RSS and the SSD tier (task 3)

Every number below is the engine's own line for that arm (`p10_table.py` reads them all back).

| arm | prompt tok | prefill tok/s | decode tok/s | ms/window | tok/window | peak VRAM card0 | card1 | peak RSS | file read this request |
|---|---|---|---|---|---|---|---|---|---|
| `1c-4k` | 3,832 | 263.5 | **20.1** | 149.04 | 3.00 | 30.83 GiB | – | 48.18 GiB | **8,698.8 MB** |
| `1c1-4k` | 3,832 | 257.7 | **21.2** | 136.27 | 2.88 | – | 31.56 GiB | 48.18 GiB | **7,601.9 MB** |
| `2c-4k` | 3,832 | 251.8 | **22.7** | 132.32 | 3.00 | 28.58 GiB | 31.14 GiB | 48.24 GiB | **0.0 MB** |
| `1c-32k` | 32,256 | 260.6 | **21.5** | 136.84 | 2.94 | 30.62 GiB | – | 48.24 GiB | **8,562.3 MB** |
| `2c-32k` | 32,256 | 351.1 | **22.6** | 129.94 | 2.94 | 28.75 GiB | 31.43 GiB | 48.29 GiB | 0.0 MB |
| `1c-128k` | 129,024 | 206.8 | **17.5** | 144.63 | 2.53 | 30.67 GiB | – | 48.29 GiB | **15,264.7 MB** |
| `2c-128k` | 129,024 | 346.5 | **20.7** | 124.94 | 2.59 | 28.97 GiB | 31.54 GiB | 48.31 GiB | **0.0 MB** |

- **Decode**: one card is -11.5% (4K), -4.9% (32K) and -15.5% (128K). The per-window figure grows with depth
  exactly as the card predicted, because the spill fraction does not shrink while the window's own work does:
  `1c-128k` shows the most CPU work (1.75 experts/layer-window, host 22.56 ms) and the widest window (144.63 ms).
- **Prefill**: one card is -25.8% at 32K and -40.3% at 128K, but **+4.6% at 4K** — a small prompt on one card
  avoids the split's hand-offs entirely, and it is the depth that hurts.
- **VRAM/RSS**: the single-card instance sits at 30.6-30.8 GiB of its 32.28 GiB card (peak, driver fdinfo) with
  only 315-537 MiB free at READY, and its RSS peaks at 48.2-48.3 GiB — the same RSS as the two-card arms, because
  the RAM-resident part of the pack grows as the VRAM cache shrinks.
- **The SSD/file tier**: the one-card requests read **8.7 GB (4K) and 15.3 GB (128K)** through the mmap'd GGUF
  during the request (`DONE ... 0 0 8698.8` / `15264.7`) against **0.0 MB** on two cards; since start the one-card
  process had read 63.1 GB against the two-card 54.3 GB. The single-card configuration therefore pays a file-tier
  cost the two-card one does not. Whether that traffic reached the SSD is **not measured in these arms** (the
  direct rig has no device-level sampler); in the serve.server arms where it *was* measured (§6) the device read
  **0.00 GiB** and every byte came from the page cache, so the volume is a RAM-bandwidth cost at minimum.
- **Reproducibility across sessions**: `2c-4k` measured 132.32 ms/window (3.00 tok/window) here against P9's
  132.38 ms/window (3.00 tok/window) on the same pack and binary — 0.05%.

## 5. Do the two configurations even produce the same answer?

No, and that is worth stating before any throughput comparison: the greedy token ids differ between the card
counts at every length (`1c-32k` vs `2c-32k`: 14 of 512 `T` lines differ; the md5 of the 256 generated ids differs
in all three pairs). The engine's own load line predicts this — "the GPU computes the experts in the cache; it
rounds differently from the CPU, so a reply can differ slightly from a run without the cache" — and on one card
~half the experts take the CPU path. The two sequences are the same length and both are plausible completions;
what is measured here is that they are not the same ids, so a single-card worker is not a drop-in for the model of
record where bit-reproducibility matters.

## 6. Two instances, one per card (task 4)

Two `serve.server` instances, `ZE_AFFINITY_MASK=0` and `=1`, each with its own config, engine log, domain and
drafter, both taking the same text prompt (the 32K depth-curve prompt decoded through the pack's own tokenizer:
**32,277 tokens as the server tokenized it**, since the id-list round trip is exact only up to the 8-id digit
nonce, 32,252 vs 32,256) and 256 generated tokens. The one-instance control is one two-card `serve.server` with
the same prompt and **two concurrent requests**, which it serves one at a time (server.py: "one sequence at a
time behind a FIFO") — so its two wall clocks are queue positions, not an aggregate.

**First finding: the plain config of record does not survive a 32K request on the split.** The one two-card
instance, started exactly as `strata-sycl-iq3s.json` reads (checkpoints ON), died mid-request and returned
`HTTP 400 saving a checkpoint part failed` / `HTTP 503 the engine stopped unexpectedly (exit code 1)`:

```
strata/sycl: memcpy failed: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
strata serve: checkpoint save: conversation snapshot running-state copy: invalid argument
strata serve: saving a checkpoint part failed
```

That is the split's cross-card running-state copy failing, which S1 already measured for this pair (a
device0 -> device1 copy fails with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`: the pair has no peer path). The
single-card instances save their two checkpoints without trouble, so both 1x2 and 2i were re-run with
`--prompt-cache 0 --prompt-cache-every 0` — the setting every direct-serve arm of this card already carried — and
the pair below is like-for-like (same prompt, same 256 tokens, no checkpoints, `--prefill 512` from the config of
record).

| arm | pool workers | prefill tok/s | decode tok/s | wall, two 32K requests | CPU busy during decode | file tier, cumulative | peak VRAM | peak RSS |
|---|---|---|---|---|---|---|---|---|
| `2i-32k-9-pc0` two instances | **9 / 9** | **97.6 / 96.8** | **11.8 / 12.7** | **352.7 / 353.5 s** | 4.89 / 4.62 cores | 694.5 / 661.9 GB | 30.61 / 31.36 GiB | 47.34 / 47.35 GiB |
| `1x2-32k-pc0` one two-card | 19 | **233.8 / 234.3** | **18.3 / 18.9** | **152.2 / 303.5 s** (serialized) | 0.69 / 0.75 cores | 58.3 / 66.4 GB | 25.69 + 31.39 GiB | 48.26 GiB |
| `2i-32k-9` two instances, checkpoints on | 9 / 9 | 99.6 / 96.6 | 12.0 / 15.9 | 345.6 / 350.3 s | 4.85 / 5.60 cores | 694.9 / 665.3 GB | 30.60 / 31.36 GiB | 47.57 / 47.59 GiB |
| `2i-32k-19` two instances, checkpoints on | **19 / 19** | 99.7 / 96.5 | 11.2 / 13.9 | 346.6 / 352.8 s | 9.16 / 11.32 cores | 695.2 / 662.3 GB | 30.60 / 31.36 GiB | 47.55 / 47.57 GiB |
| `1x2-32k` one two-card, checkpoints on | 19 | — | — | **66.4 / 66.9 s, both FAILED** | — | — | 25.68 + 31.38 GiB | 48.28 GiB |

For reference, the same prompt on one card **solo**: 260.6 tok/s prefill and 21.5 tok/s decode; on two cards
**solo**: 351.1 and 22.6.

**What the numbers say.**

- **Concurrency costs each instance more than half of its rate.** Decode falls from 21.5 to 11.8-12.7 tok/s and
  prefill from 260.6 to 96.8-97.6 tok/s. The theoretical aggregate of two untouched instances would be 43 tok/s
  of decode; the measured aggregate is **24.5 tok/s** (summed, while both decode) — the interference eats 43% of
  it. Per request the pair is worse than the two-card instance by **+55% decode time and +140% prefill time.**
- **The pool split buys nothing; the extra cores are spent and not recovered.** 9+9 workers against 19+19 (38
  workers on 20 cores): same wall (352.7 vs 352.8 s), prefill identical (97.6 vs 99.7 tok/s — the prefill path
  does not use the pool at all on one card), decode slightly **worse** with more workers (11.2/13.9 vs 11.8/12.7),
  while cores busy during decode rise from **10.5 to 20.5**. That is the shape of a workload stalled on a shared
  resource, not on worker count.
- **The shared resource is RAM bandwidth, and the SSD is not the limit.** Each instance pulled **608-640 GB**
  through the engine's file tier during the one 32K request (cumulative 661.9-695.2 GB against 54.3-63.1 GB for
  the solo arms), and `/proc/diskstats` for the pack's device (`nvme0n1`) shows **0.00 GiB read from the device**
  during either pc0 arm — every one of those bytes was a page-cache hit, i.e. RAM traffic between two processes
  streaming the same mmap'd experts. The MTP draft also degrades under that contention (drafts accepted 128/178
  and 124/182 against 172/218 solo), which is why tokens/window falls from 2.94 to 1.98.
- **The asymmetry between the cards is not what the PCIe probe suggests.** Card 1's host link measures half of
  card 0's (engine's own probe: 26.4-26.5 GB/s against 13.4 GB/s, S1's measurement), yet with one card each the
  two instances' rates are close (card 0: 97.6 prefill / 11.8 decode; card 1: 96.8 / 12.7; the 4K single-card arms
  measured 263.5/20.1 on card 0 against 257.7/21.2 on card 1). Compute-bound decode does not pay the link
  difference, and the aggregate must not be halved.
- **Memory footprint**: 2 x ~47.4 GiB of RSS and 2 x ~31 GiB of VRAM for two instances, against 48.3 GiB and
  57.1 GiB (both cards) for one — the shared page cache is what makes the second instance's load cheap.


## 7. Verdict (task 5)

**The CPU expert path works — that part of the question is answered yes — but running two one-card instances does
not buy double the workers, and the honest answer to the second question is no.** One B70 with
`--expert-cache auto` holds **52.3-54.0%** of the 24,576 profiled experts (12,860-13,258 pairs, 24.4-25.2 GiB),
the pool computes the rest on **14.6-15.8 of the 20 cores** during decode (19 workers plus the host, 70-88% of a
core each; `CPU experts 1.22-1.98 (1.45-2.43 entries)` per layer-window where the two-card config reports exactly
0.00), and the price is **-11.5% decode at 4K, -4.9% at 32K and -15.5% at 128K**, plus **-25.8% / -40.3% of
prefill** at 32K / 128K and **8.6-15.3 GB of file-tier traffic per request** the two-card configuration never
pays (0.0 MB) — and the answers are no longer bit-identical (14 of 512 token ids differ at 32K). Sequentially
that is a clear loss: a single-card instance is slower on every axis than the two-card split, and it is a
different (not better or worse-verified) answer. Under concurrency the arithmetic does not rescue it either:
two instances together reach **24.5 tok/s** of aggregate decode against **22.6 tok/s** for one two-card instance
serving one request at a time — **+8%**, not +100% — because each instance loses half its own rate to the shared
RAM path (prefill 260.6 -> 97 tok/s each) and because doubling the pool to 19+19 workers on 20 cores moves the
cores-busy figure from 10.5 to 20.5 while making the wall clock no better. For two concurrent 32K prompts the
pair finishes in **353.5 s against 303.5 s** for the single two-card instance, i.e. **16% worse end-to-end**, and
per request it is 1.8x slower to decode. The trade is therefore: **two instances give up real per-request QoS
(1.8x slower decode, 2.4x slower prefill under load, 52% expert residency instead of 100%, ~9-15 GB of file-tier
traffic per request and a different token stream) for a measured aggregate gain of about 8% — which is not worth
a production change on this evidence.** The one regime the arms did not test, and where two instances could still
win, is decode-dominated traffic (long generations on short prompts) with two or more concurrent requests: there
the aggregate is the only thing that matters and the prefill collapse that decided this comparison does not
occur — that is a follow-up measurement, not an assumption. Two other measured facts belong with the verdict:
the resident server's own 32K path on the split **fails outright** with checkpoints on (the cross-card
checkpoint copy, §6), and `--pool-workers` should stay at the default or be *reduced* — 19+19 is strictly worse
than 9+9 and no better than 19 on one card.


## 8. Not tested

- **One prompt family.** 4K is the needle prompt (16,146 B of text, 3,832 tokens, and the model's own stop at
  150 tokens), 32K/128K are the depth-curve code-review prompt. No other family was run, and the text arms (§6)
  re-tokenize to 32,277 because the id-list round trip differs on the 8-id digit nonce.
- **No expert cache smaller than `auto`.** `--expert-cache auto` produced the spill on its own, so the numeric
  form (`--expert-cache <N>`) and `--resident-budget-gib` / `--expert-cache-per-layer` were not exercised: a
  deliberately tiny cache (deeper spill, more CPU work) and a per-layer one are both unmeasured here.
- **`STRATA_FORCE_AVX2=1` was not run** — this host has no AVX-512, so the engine already dispatches to the AVX2
  kernels (its own load line says so); the switch would pin the same arm.
- **Concurrency beyond 2 requests**, and no mixed-length or mixed-arrival-rate load. Concurrency 2, both requests
  at 32K, both arriving at the same instant, is the only concurrent measurement.
- **One two-instance configuration only**: two 32K prompts, one per instance. The decode-dominated regime
  (short prompts, long generations), where the two-instance arrangement could still win on aggregate, was not
  measured.
- **The prompt-path chunking difference between the two rigs is named, not isolated.** The direct arms
  (`p10_run_arm.sh`) pass `--prefill auto`; the `serve.server` arms carry the config of record's `--prefill 512`.
  That is a candidate explanation for the file-tier volume difference (8.6 GB per 32K request in the direct arms
  against 608-640 GB per instance in the server arms), and it was not A/B'd.
- **The 8% aggregate decode gain is one measurement per arm, not a distribution.** No variance estimate was
  taken; the 9/9 against 19/19 pair differs by 2.6% in wall clock, which is the run-to-run band the arms show.
- **The 1x2 wall clock includes the server's queue, not a concurrent service.** The two-card instance serves one
  sequence at a time by construction, so "aggregate throughput under concurrency" for it is measured as the
  completion time of two queued requests, not as two simultaneous streams.
- **No quality review of the divergent tokens.** The single-card answers differ from the model of record's ids
  (§5); whether the divergent text is better or worse was not read by anyone.
- **Nothing pushed** to a remote: the arms, the rig and this record are local commits on `sycl-xpu`.

## 10. How to reproduce

```
bash p10/p10_chain.sh      # the six like-for-like arms: 1c/2c x 4K/32K/128K  (about 30 min)
bash p10/p10_chain2.sh     # 1c1-4k (card 1), 2i-32k-9, 2i-32k-19, 1x2-32k  (about 25 min)
python3 p10/p10_prompt_text.py 32768                      # the text form the serve arms need
bash p10/p10_chain3.sh     # the pc0 concurrency pair: 1x2-32k-pc0, 2i-32k-9-pc0
python3 p10/p10_table.py                                  # the comparison table
bash p10/p10_evidence.sh > p10/P10-EVIDENCE.txt            # the raw output per arm
```

Rig files: `p10_run_arm.sh` (one engine, ids protocol), `p10_two.sh` (serve.server arms, one or two instances),
`p10_drive.py` (FIFO driver), `p10_threads.py` + `p10_cpu_report.py` + `p10_threadcheck.py` (per-thread CPU and
the instrument cross-check), `p10_top_evidence.py` (top/pidstat raw), `p10_disk.py` (/proc/diskstats),
`p10_summary.py` + `p10_table.py` (the engine's own numbers), `p10_two_configs.py` + `p10_two_report.py` +
`p10_client.py` (the server arms), `p10_prompt_text.py` (the id-list prompt as text),
`p10_pidstat_check.py` (which per-thread tool to trust). Arm data lives in `/home/michael/strata-xpu/p10/runs/`.

## 9. The machine as this card leaves it

**All engines and servers stopped, no `ZE_AFFINITY_MASK` exported, nothing resident.** `p6_stop.sh` finds no
engine and no server, port 8099 (and 8101-8103) free, card 0 at 965 MiB (the desktop) and card 1 at 0.0 MiB. The
resident server this card was told to stop was already stopped before the first arm and was **not restarted** --
whatever card runs next should start it if it needs it (`bash p6/p6_start.sh`).

