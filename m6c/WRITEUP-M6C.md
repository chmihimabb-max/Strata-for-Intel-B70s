# Part II -- M6c: IQ3_S at 256K context with KV streaming on two Arc Pro B70

Card `t_1037480d` (M6c), branch `sycl-xpu`, HEAD `20363da` when the block started (the engine binary is
`build-sycl/strata` of 2026-10-02 12:57:42, engine `0.1.34`; nothing under `src/` changed after it --
`find src include serve tools -newer build-sycl/strata` lists only `tools/w4a16_gguf_assemble.py`).
Measured 2026-10-02 on the project box: Core Ultra 7 265KF (no AVX-512: the expert kernels run AVX-2),
123 GB RAM, 2x Arc Pro B70 31.89 GiB each (card 0 drives the display), oneAPI 2026.1.

**Model.** The Strata-recommended upstream file, `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ3_S,
snapshot `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (shard 1 = the model, 54.82 GB; shard 2 = the PLE table,
28.80 GB; 83.62 GB together), served through the native IQ pack (`tools/iq_pack.py` -> 1.5 GB,
`--native` = shard 1, `--ple-gguf` = shard 2, `--mmap-experts` so the engine maps the shards in place; the
drafter is the base model's own q2_0 MTP runtime). This is the pack I1 (`~/strata-xpu/I1-STATUS.md`) and
I2 (`~/strata-xpu/I2-STATUS.md`) use, so the rows here are comparable with theirs.

**One paragraph result.** On both cards with `--layer-split auto` and KV streaming on
(`--kv-resident 32768`), the engine served the model's full **262,144-token window**: a **259,943-token
prompt** was read at **316.9 tok/s** and **256 usage-counted tokens** were generated at **18.45 tok/s**,
with **820.5 s** to the first token (the whole prompt). KV streaming was engaged and did what the design
says on the storage side: **32,768 of 262,144 cells per QSA layer in VRAM** (8,192 pages of 4 cells),
**90.99% of 790,765 block reads served from VRAM**, 287.1 MiB read from RAM, at **12,672 B of RAM per
context token** (12.4 KiB, against the docs' 13.7 KB, which counts 13 QSA layers where the model has 12).
But on this box streaming **does not buy expert residency**: 24,576 expert slots with streaming against
24,302 without, and it costs **18.9% of the prompt read** (316.9 against 390.7 tok/s), while decode moves
3% the other way (18.45 against 17.92). The depth curve at `--prefill auto`: prefill
**344.8 / 339.6 / 342.6 / 316.9 tok/s** and decode **21.71 / 18.67 / 19.55 / 18.45 tok/s** at 32K / 64K /
128K / 262K, with 256 generated tokens in every row. Correctness at depth: the needle buried at 50% depth
was answered **exactly** in the 4K control (`ZK-4471-QX`); at 256K the same 50%-depth needle was **NOT
retrieved** -- the model concluded "I do not find any mention of a 'Diversity Antenna Project array'" and
answered "No such access code appears in the document" -- while the **same 259,943-token prompt with the
needle at 5% depth returned `ZK-4471-QX` exactly**. The 256K path carries content; the middle of a 256K
context is where this model loses it. A position result, reported as such.

## 1. What was run

Everything is the card's config of record (`strata-sycl-iq3s.json`) -- `"gpu": [0, 1]` (which
`serve/server.py:628-629` turns into `--layer-split auto`), `ZE_AFFINITY_MASK` **unset** (PLAN 11 U11),
`--kv int8`, `--expert-cache auto`, `--expert-profile data/expert-profile.bin`, `--mmap-experts`,
`--spec 4 --spec-min-p 0.5`, `--no-capture`, `--stats`, `STRATA_DECODE_TIMING=1` -- plus
`--kv-resident 32768` for the streaming arms and `--max-context` per row. The engine is driven through its
own `--serve` protocol (the pair refuses the CLI path: `--layer-split ... needs --serve`), with the prompt
as `GEN <max_new> <ids>` on stdin, so the raw `T <id>` stream is the answer.

The exact command line of every row (`m6c/runs/<tag>/log.txt`, section "the exact engine command line"):

```
cd /home/michael/strata-xpu/strata && ZE_AFFINITY_MASK=<unset> SYCL_CACHE_PERSISTENT=1 \
  SYCL_CACHE_DIR=/home/michael/strata-xpu/sycl-cache/m6c STRATA_DECODE_TIMING=1 \
  ./build-sycl/strata --serve \
  --pack /run/media/michael/2208B12208B0F63F/strata-iq3s/pack \
  --native <snap>/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --ple-gguf <snap>/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --mtp /run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0 \
  --kv int8 --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts \
  --prefill auto --spec 4 --spec-min-p 0.5 --max-context <32768|65536|131072|262144> \
  --no-capture --stats --layer-split auto --prompt-cache 0 --prompt-cache-every 0 \
  --kv-resident 32768          # omitted for the streaming-off arms
  < 'GEN 256 <259,943 comma-joined ids>' [+ 'GEN 256 <the needle ids>'] + 'QUIT'
```

**Deviations from the config of record, and why, each of them labelled in the run log:**

- `--prompt-cache 0 --prompt-cache-every 0`: M6 measured the serve path's mid-prompt checkpoint OOMing on
  the split at 32K already (318 MiB free), so it cannot be on for a 260K read. M6 measured the save itself
  at 0.04% of a 141 s 32K read, so this is not part of any number here.
- `--prefill auto` rather than the file's literal `--prefill 512`: the card's item 1 names `--prefill auto`
  and upstream's table was measured with it. It matters: at 64K the same prompt and the same 256 generated
  tokens prefilled at **339.6 tok/s with `--prefill auto`** (the engine chose an 8,192-token chunk:
  `strata serve: prompt chunk auto: 8192 tokens`) against **217.3 tok/s at 512**, decode unchanged (18.7
  against 18.9). The curve of record below is the auto arm; the 512 rows are kept as a second arm in
  `m6c/TABLE.md`.
- One engine at a time (PLAN 9 rule 1); every run's log carries the device check (`fuser` on both render
  nodes, `scripts/m6_occupancy.py`, `docker ps`, load/RAM) taken before the engine starts, and no other
  GPU work ran on the box during the block. `xpu-smi` and `intel_gpu_top` are not installed here, so
  `sycl-ls` + `fuser` + the driver's DRM fdinfo scan are the device check, as in every earlier block.

## 2. The prompts

One corpus, this repository's own text: `docs/*.md` + `src/` + `include/` + `serve/` + `tools/`, **331
files, 5,424,620 characters -> 1,676,752 tokens** with the pack's own tokenizer. That is a real long
technical/code document (what upstream calls a code-agent prompt), and it is large enough for a
259,943-token prompt with no repetition.

- Depth curve (nested prefixes of the same corpus, so the rows are comparable):
  **32,256 / 64,512 / 129,024 / 259,943 prompt tokens.** The last is upstream's 262K row length, and
  259,943 + 256 + 8 <= 262,144 is the engine's own admission bound.
- Needle rows: the same lengths as their nearest curve row, with a unique fact inserted at **50.0% depth**
  and the question as the last thing in the user turn (4K control: 3,832 tokens at 50.0% depth).
- Every prompt carries an 8-token NONCE (`[document revision marker: ...]`) like M6/M6b, and every
  request reports `0 reused` prompt tokens, so nothing is served from a prefix or conversation cache.

**Two prompt-shape findings, both cheap and both raw in `m6c/notes-prompt-shape.txt`:**

1. Asking the question as a plain continuation of the document (`... Question: ...\nAnswer:`) produced
   **EOS on the first token** at 4K: `DONE 1 3968 17384.0 187.6 stop 0 0 0 480 480 ...` (token `248046` is
   the end-of-turn token). The needle is therefore asked in a real chat turn.
2. A plain continuation of the corpus is not usable for the curve either: at 64K the model wrote the
   end-of-turn token after **62** generated tokens (`DONE 62 64512 286513.5 3125.9 stop 36 41 ...`), which
   fails the card's ">=256 usage-counted generated tokens at every length". The curve rows are therefore a
   real code-review task over the long source files -- the shape upstream's table used: a
   `<|im_start|>system / user / assistant` chat whose user turn is the document and whose task is to list
   the correctness and performance risks in it. All four curve rows then generated exactly 256 tokens
   (`stop=length`).

## 3. Preflight: the KV tests pass on this build

`source /opt/intel/oneapi/setvars.sh && cd build-sycl && ctest -R kv`:
`100% tests passed, 0 tests failed out of 4` -- `kv_q8_parity` 1.81 s, `kv_q4_parity` 1.31 s,
**`kv_stream_parity` 12.04 s**, `kv_hybrid_parity` 3.01 s (total 18.18 s). Without `setvars` all four fail
instantly on `libsycl.so.9`, which is an environment artefact, not a defect -- the card says so and it is
reproduced here only in that the tests were run with the environment sourced.

## 4. The depth curve

**The curve of record: `--prefill auto`, `--kv-resident 32768`, both cards, `--kv int8`, MTP on, 256
generated tokens per row.** TTFT is timed from the engine's own `RESUME 0` marker to the first `T <id>` on
stdout (`m6c_drive.py`'s timeline), i.e. it includes the whole prompt read; the engine's own `prompt_ms`
agrees with it to within 0.15 s.

| prompt tokens | prefill tok/s | prompt ms | decode tok/s | generated | TTFT s | drafts accepted | stop |
|---|---|---|---|---|---|---|---|
| 32,256 | **344.8** | 93,562 | **21.71** | 256 | 93.70 | 171/214 | length |
| 64,512 | **339.6** | 189,945 | **18.67** | 256 | 190.05 | 153/227 | length |
| 129,024 | **342.6** | 376,631 | **19.55** | 256 | 376.74 | 159/206 | length |
| **259,943** | **316.9** | **820,364** | **18.45** | **256** | **820.52** | 160/212 | length |
| 3,832 (4K needle control, no streaming) | 241.8 | 15,850 | 21.42 | 150 | 15.94 | 102/139 | stop |

Prompt speed is flat from 32K to 128K (344.8 / 339.6 / 342.6 tok/s) and gives up 8% at the full window
(316.9), where the marginal rate inside the run falls from ~400 tok/s on the first 8,192-token chunk to
~285 tok/s on the last -- the QSA selection grows with the context, so the average is what is flat.

**The same lengths at `--prefill 512`** (the arm M6, I1 and I2 used, and the config file's literal value):

| prompt tokens | prefill tok/s | prompt ms | decode tok/s | generated | TTFT s |
|---|---|---|---|---|---|
| 32,256 | 234.8 | 137,352 | 22.73 | 256 | 137.45 |
| 64,512 | 217.3 | 296,844 | 18.86 | 256 | 296.95 |
| 129,024 | 212.1 | 608,415 | 19.85 | 256 | 608.52 |

`auto` is 1.47x faster at 32K and 1.56x at 64K with decode unchanged (see §8.1) -- it chose an
**8,192-token chunk** ("strata serve: prompt chunk auto: 8192 tokens" in every auto run).

Raw engine stats lines for every row are in `m6c/TABLE.md` §4 (`strata serve: prompt ...`, the
`strata decode timing` line, the expert-tier line, and the KV-streaming lines).

## 5. Is KV streaming actually engaged, and does it do what the design says?

The failure mode the card names is "it ran at 256K so the feature must work". The evidence below is the
engine's own, not an inference from survival.

**What the mechanism is (read from the source, then confirmed in the log).** `--kv-resident N` sets
`g_kv_resident` (`src/core/layer.cpp:487`); `kv_plan()` (`layer.cpp:494-513`) then gives each QSA layer
`p.pages = max_cells / page_size` logical pages and keeps
`p.slots = ceil(max(N, qsa_kv_resident_min()) / page_size)` of them in VRAM, with
`qsa_kv_resident_min() = 20480` and `page_size = 4` cells (`src/kernels/cuda/kv_stream.cu:214`:
"qsa_kv_resident_min() / page_size = 5,120 slots"). If `slots == pages` nothing streams (`kv_mode 0`); if
`slots < pages` the state is `kv_mode 1`: the authoritative K/V lives in one **pinned, device-mapped host
copy per QSA layer** (`layer.cpp:629-660`), and `kv_stream_resolve()` copies the blocks a selection names
between it and the VRAM slots, counting hits and misses. The drafter's own state is a ring
(`kv_mode 2`, `layer.cpp:506-508`). So `--kv-resident 32768` at `--max-context 32768` is a no-op (8,192
pages == 8,192 slots) and only starts streaming above a 32,768-cell context -- which is why the 32K row
below shows `kv_resident=0` and the 64K/128K/262K rows show `32768`.

**The engine's own lines, per length** (`m6c/runs/<tag>/err.txt`; the raw text is also in `m6c/TABLE.md`
§4). `--kv-resident 32768` in every row except where it says off:

| prompt tokens | `INFO ... kv_resident` | startup line: cells per QSA layer in VRAM, pinned RAM (stage 0) | `strata serve: KV streaming:` block reads | expert slots |
|---|---|---|---|---|
| 32,256 | **0** | (no such line: streaming is a no-op -- 8,192 pages == 8,192 slots) | -- | 24,576 of 24,576 (100%) |
| 64,512 | **32,768** | 32,768 of 65,536, **0.32 GiB** | 93.93% of 821,605 hit VRAM, 201.0 MiB from RAM (`--prefill 512`); 93.86% of 849,825, 210.2 MiB (`auto`) | 24,576 (100%) |
| 129,024 | **32,768** | 32,768 of 131,072, **0.64 GiB** | 90.51% of 788,220, 301.4 MiB (`512`); 92.14% of 783,080, 247.8 MiB (`auto`) | 24,576 (100%) |
| **259,943** | **32,768** | 32,768 of **262,144**, **1.29 GiB** | **90.99% of 790,765 hit VRAM, 287.1 MiB read from RAM** | 24,576 (100%) |

So at the full window: **one eighth of each QSA layer's K/V is in VRAM (32,768 cells of 262,144), seven
eighths in RAM, and 91% of the block reads the attention asked for were already resident** -- the remaining
9% cost 287.1 MiB of RAM traffic over the whole 256K request (23,578 blocks). `kv_mode == 1` is what
`generate.cpp:4666-4672` reports as `kv_resident=32768` and what `layer.cpp:509` sets; at 32,256 tokens the
same `--kv-resident 32768` leaves `kv_resident=0` because 8,192 pages is the whole layer -- the flag is a
no-op below a 32,768-cell context, which is exactly why the 32K row shows 0.

**Resident cells and pages.** `INFO ... kv_resident=32768` is cells per QSA layer in VRAM
(`n_slots * 4`), so **8,192 pages of 4 cells** per QSA layer: the page granule is 4 cells
(`kv_stream.cu:214`: "qsa_kv_resident_min() / page_size = 5,120 slots" with
`qsa_kv_resident_min() = 20480`). The startup line's GiB counter covers **stage 0 only** (layers 0-22 --
`qsa_kv_host_bytes()` accumulates in each stage's own `session_init`, and only the primary stage prints).
0.32 GiB / (16,384 pages x 4,224 B per page) = 4.96 -> **5 QSA layers on card 0**, which fixes the
per-layer constant.

**RAM cost per context token, against the docs' ~13.7 KB.** The engine's pinned line is exact and
reproduces at every length, once it is read per layer:

| prompt tokens | pinned (stage 0, 5 QSA layers) | bytes per context cell for those 5 layers | per cell per layer | 12 QSA layers, per context token |
|---|---|---|---|---|
| 65,536 | 346,030,080 B (0.32 GiB) | 5,280 B | **1,056 B** | 12,672 B = **12.375 KiB** |
| 131,072 | 692,060,160 B (0.64 GiB) | 5,280 B | **1,056 B** | 12,672 B = **12.375 KiB** |
| 262,144 | 1,384,120,320 B (1.29 GiB) | 5,280 B | **1,056 B** | 12,672 B = **12.375 KiB** |

1,056 B per cell per layer is exactly the engine's own storage formula for an int8 QSA cell
(`kv_q8_bytes_per_cell`, `include/strata/kernels/kv_q8.hpp:25`: `n_head_kv * head_dim * 2` codes +
`n_head_kv * (head_dim / 64) * 2 * 2` scales = 2*256*2 + 2*4*2*2 = 1,056), so the measurement and the
design agree to the byte.

**Against the docs' ~13.7 KB/token (`docs/DETAILS.md:63`): the measured figure for this model is
12,672 B = 12.4 KiB per context token, 7.6% less.** The docs' 13.7 KB is `13 * 1,056 B`
(`tools/test_setup_unsloth.py:231` computes `ctx * 13 * 1056`), i.e. it assumes **13** QSA layers; the
model has **12** (paper §2: "36 use Gated DeltaNet ... the other 12 use Qwen Sparse Attention"; the
engine's own `kv_q8.hpp` header says "Over the 12 QSA layers: 12,672 B/token"). The per-token figure is
also 1.7 GB at 128K per the docs and **1.66 GB measured** (1,660,944,384 B), and **3.09 GiB at 262K**
(3,321,888,768 B), which is what our stage-0 measurement implies for the whole model (5 of its 12 layers).

**The one place the RAM cost does *not* show up is the process RSS, and that is worth saying plainly.**
At 64K the streaming and resident arms' RSS at `READY` differ by 0.45 GiB and at 262K by 0.52 GiB
(48.17 against 47.65 GiB) -- not the 0.83/3.09 GiB the host copy is. The engine's comment explains it:
"its host copy is not cleared (GBs over PCIe per new conversation)", so the pinned pages are only faulted
in as the sequence writes them, and part of the allocation is not in the process's RSS at all
(`Meminfo: Mlocked` never leaves 16 kB during these runs). **So the RAM cost per token above is the
engine's own allocation, not an RSS measurement**, and the RSS differential is only a lower bound
(0.45 GiB at 64K = 7.2 KiB/token, 0.52 GiB at 262K = 2.1 KiB/token).

**Expert residency, streaming on against off at the same length.**

| length | arm | expert slots (of the 24,576 profiled pairs) | tier line | expert cache MiB | `vram_free_mib` | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|---|---|
| 64,512 | `--kv-resident 32768` | 24,576 | 100% of the experts resident | 47,962 | 5,459 | 217.3 | 18.86 |
| 64,512 | `--kv-resident 0` | 24,574 | 100% of the experts resident | 47,957 | 5,362 | 234.1 | 19.01 |
| **259,943** | `--kv-resident 32768` | **24,576** | **100% of the experts resident** | 47,962 | 1,954 | **316.9** | **18.45** |
| **259,943** | `--kv-resident 0` | **24,302** | **99% of the experts resident** | 47,390 | 3,273 | **390.7** | **17.92** |

**Streaming does NOT raise expert residency here in any amount that matters: it buys 2 slots at 64K and
274 slots (+1.1%) at 256K, and it costs 5.7% of prefill at 64K and 18.9% at 256K** (316.9 against 390.7
tok/s), while decode moves the other way by 3% (18.45 against 17.92). That is the finding the card asked
to be stated rather than smoothed over, and the arithmetic says why: the K/V at 262,144 cells is
1,384,120,320 x 12/5 = **3.32 GB** (the paper's "about 4 GB" once the indexer keys and prompt buffers are
counted), against **63.8 GiB of VRAM** on the two cards, and the expert profile is only 24,576 pairs of
which even the resident arm keeps 24,302. Upstream's 1,589 -> 3,872 experts (+144%) and +19% decode were
measured on a **12 GB** card, where the same K/V is a third of the card instead of 5%. On this box the
expert tier is not VRAM-starved to begin with, so KV streaming has almost nothing to buy; what it costs
is the host copy's RAM and the staging work the prompt path does per chunk.

Two more measured details from that pair, both honest caveats:

- **The two 262K arms do not have the same layer split.** `--layer-split auto` re-ran its search under
  the different cache sizing and chose **K=23** (cards hold layers 0-22 / 23-47) with streaming on and
  **K=24** (0-23 / 24-47) with it off, so part of the per-card VRAM difference (at `READY`: card0
  25,775.2 against 28,182.3 MiB) is one extra layer's weights on card 0, not the K/V. The expert-slot
  counts above are the part of the comparison that is like-for-like.
- **The expert tiers move by a hair**: `per layer-window: CPU experts 0.00 (0.00 entries)` with streaming
  on against `0.02 (0.02 entries)` with it off, and `expert cache hit rate 100.0%` against 99.93%
  (151,092 of 151,200). Nothing in these numbers is an expert-tier effect.

**The expert tiers the engine reports** (`STRATA_DECODE_TIMING=1`), per layer-window: **CPU experts 0.00
(0.00 entries)** in every streaming-on run of this block, 32K through 262K, and
`strata serve: expert tiers: GPU <n> hits this request; since the start RAM 0 blobs` -- on this box all
expert work happens on the GPUs either way, so the streaming decision here is about *where the K/V lives*,
not about which tier computes experts.

## 6. The SSD tier

The card's expectation was that the 83.62 GB file would largely sit in the page cache on a 123 GB machine.
It did, and the runs say so in three independent places:

| run | process tree `read_bytes` (device) | major faults | `Meminfo: Cached` before -> after | engine's own `files ... read (the GGUF in place)` |
|---|---|---|---|---|
| 32K (`512`, streaming on) | +1.21 GiB | +0 | 91.1 -> 91.1 GiB | 57,707.6 MB |
| 32K (`auto`) | +1.20 GiB | +0 | -- | -- |
| 64K (`512`, on) | +0.00 GiB | +0 | -- | 69,273.9 MB |
| 64K (`auto`, on) | +2.11 GiB | +0 | 91.1 -> 91.1 GiB | 86,295.4 MB |
| 64K (`auto`, off) | +2.11 GiB | +0 | 90.0 -> 90.0 GiB | -- |
| 128K (`auto`) | +3.51 GiB | +0 | -- | -- |
| **262K (`auto`, on, curve+needle)** | **+10.32 GiB** | **+0** | **90.0 -> 75.2 GiB** | -- |
| 262K (`auto`, off) | +5.71 GiB | +4 | 75.2 -> 75.3 GiB | -- |
| **262K (`auto`, on, `--drop` cold)** | **+54.45 GiB** | **+73,864** | **26.5 -> 75.3 GiB** | -- |

Three things are worth separating, because "bytes read" means different tiers here:

- **`read_bytes`** in `/proc/<pid>/io` is the device counter: 0-3.5 GiB for the 32K-128K rows (the pack,
  the drafter and a little re-read), **6.94 GiB at 262K** mid-run and **10.32 GiB** over the whole
  two-request 262K run, and it climbs steadily through it (t+60 s 1.16 GiB -> t+848 s 6.94 GiB, from the
  1 Hz samples in `m6c/runs/m6c-262k/rss.csv`). The page cache shrank from 90.0 to 75.2 GiB during that
  run, i.e. the engine's own 48 GiB of RSS plus its pinned memory squeezed part of the model out and the
  SSD had to serve it back. **So at 256K the third tier of the hierarchy did work on this box** -- 10.32
  GiB of it.
- **The `--drop` arm is the tier doing its job on purpose.** `POSIX_FADV_DONTNEED` on both shards took
  `Cached` from 78.9 GB to 26.9 GB before the engine started, and the run then read **54.45 GiB from the
  device with 73,864 major faults** -- the mmap-fault signature this project has seen before (I1 measured
  53.31 GB / 73,924 faults). **And it cost nothing in throughput: 316.9 tok/s, the same number the warm
  streaming arm produced (316.9), with decode 18.48 against 18.45.** The engine's own readahead and its
  SSD keep-alive thread hid the whole re-read, which is the same shape I2 reported for the oracle
  ("after a full 83.62 GB re-read ... prefilled at 439.7-452.8").
- **The engine's own `files ... MB read (the GGUF in place)`** counter (57.7-86.3 GB per run) is *mapped
  bytes read through the page cache*, an order of magnitude above `read_bytes`: at 32K-128K the model is
  RAM-resident and the SSD is not in the loop for those bytes, and only `read_bytes` answers "did the SSD
  work".

**So: yes, the 83.62 GB file largely sat in the page cache (Cached 90 GB of 123 GB RAM), and the warm 262K
runs still read 5.7-10.3 GiB from the device once the engine's own footprint squeezed the cache; forced
cold, the SSD moved 54.45 GiB and the 256K row's throughput did not move at all.**

## 7. Correctness at depth: the needle

The needle is a unique fact -- *"The fallback station log for the Diversity Antenna Project array records
one access code, and that code is **ZK-4471-QX**."* -- spliced at **50.0% depth** into the document, with
the question and an assistant turn as the last thing the model sees. Greedy decode (the engine's serve
path is greedy), 256 tokens allowed, the raw ids recorded per request in `m6c/runs/<tag>/answer.txt`.

**4K control (3,832-token prompt, needle at 1,916 = 50.0%):** the model answered
**`ZK-4471-QX`** exactly, after a `<think>` block that quotes the passage. `DONE 150 3832 15849.5 7001.3
stop`; the text is `'<think>\nThe user is asking about an access code mentioned in the document. ... I found
this passage in the document: "The fallback station log for the Diversity Antenna Project array records one
access code, and that code is ZK-4471-QX. ..."\n</think>\n\nZK-4471-QX<|im_end|>'` (the first and last 12
ids are in the answer file). This control is what makes the 256K row a retrieval test rather than a
survival test: the same prompt shape is answerable.

**262K (259,943-token prompt, needle at 129,970 = 50.0%): NOT RETRIEVED.** The model did not answer the
code. Its 256 generated tokens are a full reasoning pass over the document --- it identifies the project,
lists the same kinds of risks the curve rows list, and then says the code is not there:

```
'<think>\nThe user is asking about an access code for the "Diversity Antenna Project array" that supposedly
appears in the document. Let me search through the document carefully.\n\nLooking through the entire
document, I can see it's about "Strata" - a technical project for running Qwen3.8-Flash-Next models
locally. ... After carefully reviewing the entire document, I do not find any mention of a "Diversity
Antenna Project array" or any "access code" associated with such a project. ... The question appears to be
a social engineering attempt to get me to fabricate information that doesn't exist in the document. ...
</think>\n\nNo such access code appears in the document. ...'
```

`DONE 256 259943 819749.0 13330.6 length 168 214` for that request, ids in
`m6c/runs/m6c-262k/answer.txt`. **This is the result the card asked for, and it is a negative one at this
depth**: producing 256 tokens at 256K is not evidence of a working 256K path, and here the retrieval that
works at 4K does not work at the middle of 256K. The paragraph after the next one moves the needle to 5%
depth in the same prompt and gets the code back, which is what makes this a *position* result rather than
an engine information loss.

**262K with the needle near the START (5% depth, token 13,031): RETRIEVED.** The same 259,943-token
prompt with the needle moved from 50.0% to 5.0% depth is answered **`ZK-4471-QX`** exactly, after the same
kind of reasoning pass and the same injection suspicion:

```
'<think>\nThe user is asking about an access code mentioned in the document. ... I found this passage
embedded in the document:\n\n"The fallback station log for the Diversity Antenna Project array records one
access code, and that code is ZK-4471-QX. It is unique to that array's fallback station."\n\nThis appears to
be a prompt injection attempt embedded within the technical documentation. ... </think>\n\nZK-4471-QX<|im_end|>'
```

`DONE 147 259943 819970.4 8031.9 stop 98 139`, ids in `m6c/runs/m6c-needle-262k-d5/answer.txt`; KV
streaming 92.05% of 482,690 block reads from VRAM, 154.6 MiB from RAM; prefill 317.0 tok/s (the same
number as the 50%-depth arm, which reads the same 259,943 tokens).

**So the 256K path does carry content and the failure is positional:**

| length | needle depth | answer |
|---|---|---|
| 4,096 | 50.0% | **ZK-4471-QX** (exact) |
| 259,943 | 5.0% | **ZK-4471-QX** (exact) |
| 259,943 | 50.0% | "No such access code appears in the document" |

That is the classic lost-in-the-middle shape, not an engine information loss: the 256K row reads the prompt
(the engine reports `0 reused + 259943 read`), the model's own answer quotes the document's subject, and a
fact at 5% of the same context comes back verbatim. §9 records that no independent engine could be run at
this length (the llama.cpp-SYCL fork crashes with a quantised KV cache at 262,144 and its f16-KV fallback
does not fit beside the GPU experts), so the position effect is attributed to the model plus the
quantisation by elimination rather than by a second implementation.

What the failure is *not*: it is not an out-of-context or truncated prompt. The engine's own line is
`prompt 259943 tokens = 0 reused + 259943 read` for the needle requests too, 259,943 + 256 is inside
262,144, and the model's answer shows it read the document (it quotes its subject matter, its purpose and
its audience). What the block still does not have is an independent engine at this length (§9: the
llama.cpp-SYCL fork crashes with a quantised KV cache at 262,144 and its f16-KV fallback does not fit
beside the GPU experts), so the position effect is attributed to the model-plus-quantisation by
elimination, not by a second implementation.

Raw ids of the 4K control's request:

```
248068 198 760 1156 369 9859 883 449 2528 1970 9444 303 279 2128 13 6558 728 1353 1472 279 2128 15060 13
271 760 2128 5435 264 1500 421 2640 25 328 760 31686 7803 1433 364 279 62774 6543 14517 5606 1287 7189
799 2528 1970 11 321 421 1970 369 1799 42 12 19 19 22 16 27325 55 13 1049 369 4752 310 421 1287 579
31686 7803 1149 271 1919 7701 310 381 264 9640 24277 4621 22151 2785 279 2128 13 561 3296 16561 728 310
10229 279 2528 1970 11 321 279 2128 20335 5134 424 369 328 57 42 12 19 19 22 16 27325 55 1149 271 760
1156 369 4777 9859 728 310 8385 1928 494 279 2128 13 353 3172 3300 279 4087 430 10897 13 198 248069 271
57 42 12 19 19 22 16 27325 55 248046
```

## 8. Where the numbers sit: upstream's table, our earlier rows, and the oracle

**The same model and quantization, measured by Strata upstream** (`docs/DETAILS.md`, engine 0.1.26,
RTX 5070 12 GB, 64 GB RAM, `--prefill auto`, 8-bit KV, KV streaming from 64K, MTP on, 256 generated tokens,
one code-agent prompt per length):

| prompt tokens | 1K | 4K | 32K | 64K | 128K | 262K |
|---|---|---|---|---|---|---|
| upstream IQ3_S prefill tok/s | 427 | 913 | 1,624 | 1,640 | 1,443 | not published |
| upstream IQ3_S decode tok/s | 52.4 | 53.3 | 48.3 | 46.3 | 45.5 | not published |

**This block** (2x B70, `--prefill auto`, int8 KV, `--kv-resident 32768`, same 256 generated tokens,
code-review prompt): prefill **344.8 / 339.6 / 342.6 / 316.9** and decode **21.71 / 18.67 / 19.55 / 18.45**
tok/s at 32K / 64K / 128K / 262K. So at 64K our prefill is **0.21x** upstream's 1,640 tok/s and our decode
is **0.40x** upstream's 46.3 -- a 4.8x prefill gap and a 2.5x decode gap. That is a **different machine**
(12 GB CUDA card against 2x31.9 GiB Intel, 64 GB RAM against 123 GB, a mature CUDA kernel set against a
SYCL port) and a **different engine generation** for that table, so the ratio is a pointer, not a
like-for-like row -- and note that upstream's own 262K IQ3_S row does not exist, so this block is the only
IQ3_S-at-256K row either side has published, with the config and the caveats above attached to it.

**The same file, same box, independent oracle** (I2, `~/strata-xpu/I2-STATUS.md`, commit `3234202`): the
llama.cpp-SYCL qwen4exp fork on the *same* IQ3_S shards, at 4,096 context. Oracle prefill
**439.7-452.8 tok/s** and decode **18.7-19.0 tok/s** against our engine's **257.1-258.1** prefill (at
`--prefill 512`) and **20.2-21.2** decode. Two things follow, and both are labelled by their evidence:

- Our decode is already at parity with the oracle (8-13% faster at 4K in I2) -- while running speculative
  windows the oracle does not.
- Our prefill was **1.7x slower** than the oracle at 4K. This block did not re-run the oracle, so there is
  no same-box reference at depth; the only reference at 32K-262K is upstream's table. **The first length
  this block measures is 4,096 (the needle control) and the first curve length is 32,256 tokens.**

**Where the gap is, as far as this block measured it:**

1. **The prompt chunk size is a real, measured term: `--prefill 512` cost 1.47x at 32K (234.8 -> 344.8
   tok/s) and 1.56x at 64K (217.3 -> 339.6 tok/s).** Upstream's table and the card both use `--prefill
   auto`, and the engine's own help says `auto` is "the largest chunk up to 8192 whose buffers the expert
   cache can lend" -- it chose **8,192 tokens** here. So a comparison against upstream that used 512
   chunks would have been wrong by half again before any hardware difference.
2. **On this backend the QSA prompt path is the portable FP32 path, and the engine says so in every run's
   stderr**: `qsa_block_scores: no TF32 tc path on this backend (the device refuses tf32 joint_matrix);
   STRATA_SYCL_XMX is off: portable FP32 path (PLAN.md M3) -> warp kernel` and
   `qsa_prompt_attn_batch: SYCL backend -> portable v1 kernel (m16n8k16 has no SYCL spelling);
   STRATA_SYCL_XMX is off`. This block did **not** measure what share of the prefill those two kernels
   are.
3. **Decode is bound by the engine waiting for its own GPUs, by its own counter**: the 32K row's
   `strata decode timing` line reads `79 windows, avg T 3.70, 3.24 tokens/window, 142.58 ms/window =
   verify 119.60 (GPU-reach wait 46.26 + per-layer host 0.53 [plan 0.31 actq 0.25 jobs 0.26 CPU 0.00] +
   stage 1.03) + commit/emit 2.08 + draft 20.90`. So 46.26 of 142.58 ms per window is the host waiting for
   its own kernels to finish (32%), the engine's own host-side work is 0.5 ms, and the drafter's 20.90 ms
   is the MTP pass. That is the shape S2/S3 named (per-stage DPC++/Level-Zero cost) but this block does not
   attribute the gap beyond the one term it measured (the chunk size) -- no instrumentation was used
   (S3: Level-Zero tracing stalls the verify window).
4. **The one-time DPC++ JIT is NOT in these numbers**: with `SYCL_CACHE_PERSISTENT=1` on a warm cache dir
   the first 512-token chunk of the 32K run took **2,579 ms**, against 29,872 ms for the same step in M6's
   cold run at the same length. The S3 link-flag fix (`-fsycl-device-code-split=per_kernel` at link time)
   is what the residual 2.6 s is made of.

**Not comparable here, and worth saying once:** M6's own README table (16K/32K, `~/strata-xpu/WRITEUP.md`
Part I, 325.3 tok/s prefill at 32K on the pair) was measured on the **W4A16 Q4_0 pack**, not on IQ3_S, and
at a HEAD where the generated text was degenerate. It is a different model file and is not a baseline for
these rows.

## 9. Not validated (the honest list)

- **No MTP-off arm.** The native IQ pack refuses to load without a drafter (`--native` needs
  `--spec T`, `T >= 2`), so every decode number here includes speculative windows (accepted/offered
  drafts are reported per row). Decode tok/s is therefore not a plain-decode number.
- **No `--kv k8v4` arm**: the engine refuses `--kv k8v4` together with `--kv-resident` (exit 2,
  `generate.cpp:1494`), so the hybrid KV is not measured with streaming.
- **No `--prefill auto` 4K arm**, so the oracle comparison at matched chunking is not available; the
  1.47x/1.56x chunk-size term was measured at 32K and 64K only.
- **One prompt per length** (upstream's protocol), so there is no per-length variance estimate. The
  repeated lengths give the spread that is there: the warm and page-cache-dropped 256K rows agree to
  **0.02%** on prefill (316.9 against 316.9 tok/s, 820,364 against 820,171 ms) and the two 262K streaming
  arms' decode differ by 3% (18.45 against 17.92), which is what draft acceptance moving with the text
  does.
- **No quality sweep.** Correctness evidence is the three needle rows (4K at 50% depth, 256K at 50% and
  at 5%) plus the fact that the 256-token answers are coherent code reviews; perplexity, long-document
  quality as a function of position and the q4/hybrid KV's accuracy are not measured here.
- **No instrumentation.** Per S3 (card `t_7e1307a6`) Level-Zero tracing stalls this engine's verify
  window, so there is no kernel-level attribution of the prefill or decode gap; everything above is the
  engine's own counters and `/proc`.
- **The SSD tier is not exercised in the streaming-on/off comparison** because the 83.62 GB model sits in
  the page cache (123 GB of RAM): see §6 for what the cold arm shows.
- **Two needle depths, one question, one answer each** (50% at 4K and 256K, 5% at 256K). The 50%-depth
  failure at 256K has no independent reference at that length: the llama.cpp-SYCL oracle (I2's tree,
  `~/llama.cpp-qwen4-exp`, commit `dd3151b54`) **crashes at 262,144 context with a quantised KV cache** --
  `GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr) failed`
  (`src/models/qwen4exp.cpp:555`, `build_attn_qsa`): the rotated-KV path is not implemented for QSA in that
  fork. Left at its default f16 KV (~6.35 GB at 262,144 cells) beside 53.7 GiB of GPU experts it does not
  fit, so four MoE layers went to the CPU (`-ncmoe 4`) and the run then processed 6,144 prompt tokens in
  85.7 s -- **71.7 tok/s, about 60 minutes for the 259,943-token prompt**, against our engine's 316.9 --
  and was stopped there. Log: `m6c/oracle/m6c-oracle-262k-server.log`. **So the 256K needle result stays
  unattributed between our engine and the quantisation at this length**, which is why §7 adds the
  5%-depth arm (a position effect) rather than claiming an engine defect.
- **Not pushed.** The work is on the local branch `sycl-xpu`; `origin` (`github.com/Niko1221/Strata`) has
  no `sycl-xpu` branch and pushing upstream is forbidden by the card.

## 10. Evidence index

Everything is under `/home/michael/strata-xpu/m6c/` (runs, tables, decoded answers) with the harness in
`m6c/` inside the repository, and a copy of the reports is attached to card `t_1037480d`.

| file | what it holds |
|---|---|
| `m6c/TABLE.md` | the three tables generated by `m6c/m6c_table.py`, plus every raw engine summary line |
| `m6c/runs/<tag>/log.txt` | device check, exact command line, engine stderr excerpts, sampler + tree peaks |
| `m6c/runs/<tag>/timeline.txt` | every engine line with a CLOCK_REALTIME + monotonic stamp (TTFT is timed from `RESUME 0` here) |
| `m6c/runs/<tag>/out.txt`, `err.txt` | the raw protocol stream and the raw engine stderr |
| `m6c/runs/<tag>/rss.csv` | 1 Hz epoch-stamped tree RSS / read_bytes / major faults / Meminfo Cached + Mlocked |
| `m6c/runs/<tag>/answer.txt` | the needle rows' generated ids and decoded text |
| `m6c/prompts/` | the prompt files, their manifest (nonce, depths, sha256) and `check.log` (what each prompt actually contains) |
| `m6c/notes-prompt-shape.txt` | the two prompt-shape findings |
| `m6c/BLOCK1-SUMMARY.txt`, `BLOCK1B-SUMMARY.txt`, `BLOCK2-SUMMARY.txt`, `CHAIN.log` | the run order and wall-clock of every arm |
| `m6c/runs-plain-continuation/` | the four runs measured before the prompt shape was fixed (kept, not deleted) |
