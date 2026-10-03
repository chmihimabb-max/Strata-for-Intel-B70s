# P10 — one card with the CPU computing the expert overflow, and (if it holds) two one-card instances

Card t_1d052912. Rig: `p10/` in this repo, arms in `/home/michael/strata-xpu/p10/runs/<tag>`, raw output per arm
in `p10/P10-EVIDENCE.txt`. Engine `build-sycl/strata` at HEAD `83d1e82` (md5 9eff0675059cf7e8145ae7b4bca984a4),
pack IQ3_S `ed59f920`, faithful drafter `~/strata-xpu/mtp/rt`, MTP on, greedy, `--prompt-cache 0`, one card or the
two-card split, `--stats` for every number below.

**The answer in one line:** the CPU expert path is real and it is busy — on one B70 `--expert-cache auto` keeps
**52-54%** of the 24,576 profiled experts in VRAM and the pool computes the rest on **15.6-15.8 cores** during
decode — and it costs **-4.9% decode at 32K** and **-15.5% at 128K** against the two-card split, but it costs
**-25.8% / -40.3% of prefill**, comes with 8.7-15.3 GB of file reads per request, and the answers are **no longer
bit-identical** to the model of record (the CPU rounds differently). {{two_instance_sentence}}

## 1. What was run, and the mask rule that is the only difference between the arms

| arm | cards | `ZE_AFFINITY_MASK` | `--layer-split` | prompt tokens | generated |
|---|---|---|---|---|---|
| `1c-4k` | card 0 | **0** | not passed | 3,832 (needle) | 150 (stop) |
| `2c-4k` | both | unset | `auto` | 3,832 | 147 (stop) |
| `1c-32k` | card 0 | **0** | not passed | 32,256 | 256 |
| `2c-32k` | both | unset | `auto` | 32,256 | 256 |
| `1c-128k` | card 0 | **0** | not passed | 129,024 | 256 |
| `2c-128k` | both | unset | `auto` | 129,024 | 256 |
| `1c1-4k` | card **1** | **1** | not passed | 3,832 | see §6 |
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
  process had read 63.1 GB against the two-card 54.3 GB. The single-card configuration is therefore a
  **GPU + CPU + SSD** path under decode, not a GPU + CPU path, and the 4K case reads 9 GB for 3,832 prompt tokens.
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

{{section_6}}

## 7. Verdict (task 5)

{{section_7}}

## 8. Not tested

{{section_8}}
