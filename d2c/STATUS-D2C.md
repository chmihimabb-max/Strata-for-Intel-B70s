# D2c — the other 61 decode-path programs are built in the load phase too, and what that is worth

Card `t_c7d8cd86` (D2c), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`, parent `t_f93760a1` (D2b).
One session, both B70s, `ZE_AFFINITY_MASK` unset, one engine at a time, the config of record's flags with D2's two
documented deviations (`--prefill auto`, `--prompt-cache 0`) and `--spec-min-p 0.7` (the value the config of
record carries since D2).
**Engine source touched**: `src/kernels/cuda/decode_warmup.cu` (new), `include/strata/kernels/decode_warmup.hpp`
(new), one include + one call in `src/program/generate.cpp`'s serve path, one TU in
`cmake/sycl_backend.cmake`. **No kernel source, no config, no flag set and no default touched** (D2a's
multi-column question stays closed).
Engine binary before this card: `f49fe6669704561bb8155f9c4c9acaed` (D2b's, kept at
`/home/michael/strata-xpu/d2c/strata-f49fe666`); after: `73ee6a3c385e06760d46277a1e481df0` (the binary every arm
below names in its own `log.txt`; the `ctest` run's `make -j8` relinked the same source and wrote the same md5).

Rig under `d2c/`: `enumerate_c.py` (names every program an arm built in a window phase, from the SPIR-V name
section), `site_cost.py` (per-site share of the decode phase), `unnamed_c.py`, `analyze_c.py` (D2b's `analyze_b`
plus the decode/prompt split, the second warm-up's own line and the start → first-answer total),
`d2c_chain_a.sh` (the three cold first runs and the three warm controls), `d2c_ctest.sh`,
`make_enumeration.sh` → `d2c/ENUMERATION.txt`, `d2c_evidence.sh` → `d2c/D2C-EVIDENCE.txt`. Raw engine output is
under `d2/runs/d2c-*`; every number below is in `d2c/D2C-EVIDENCE.txt`.

## 1. Verdict

1. **The 61 programs are now 55 named launch sites, and the warmable ones are warmed.** `d2c/ENUMERATION.txt`
   §1-§3 name every program the cold decode phase built, in build order, with its launch site and its exact
   kernel instantiation; §4 is the warmable/bound table with the reason for each. D2b's "55 other" bucket is
   gone: it was 55 distinct sites (one program each), not 55 unclassifiable ones. D2b's `name_other.py` saw
   "(no symbol)" for 35 of them because it looked for a `native_*kernel` at the start of a `strings` line, while
   the site is carried in a *mangled typeinfo* whose `c++filt` form names it exactly (§2).
2. **A cold 4K first run with both passes builds 15 programs in its decode phase, not 61** (it was 93 with no
   pass at all), the residual falls **35.51 → 9.46 ms/window (1918 → 511 ms)** and the ask **28.23 → 26.20 s**.
   `ctest -R mmvq_multi_parity` passes, the families of the warmed kernels pass, the S4 set is **5 failed of 49 —
   the same five names as the recorded baseline**, and every arm, cold or warm, reproduces the canonical ids
   (4K `66bf952d…`, 32K `d87373e8…`). §4, §5.
3. **The pass costs 2.29 s of load on a cold first run (5 ms warm) and saves 2.03 s of the ask, so it does NOT
   turn the trade-off positive**: start → first answer is **77.23 s without this card's pass and 78.20 s with
   it** (§4). The card's premise — that the other 61 builds are what makes the fresh install 12.4 s worse — does
   not survive the measurement, and §6 gives the arithmetic of why: those 61 programs are one-per-site, so the
   request needs all of them and a warm-up for them can only *move* the work, never remove it. The 12.9 s the
   fresh install is still paying is D2b's over-warm (96 specializations built where a 4K request uses 32), i.e.
   it is recoverable by building FEWER programs, not by warming more.
4. **What the pass does buy is the request, and it is worth stating separately**: the cold first request's decode
   windows now build 15 programs instead of 93, its residual is 9.46 ms/window instead of 128.26 (3.2x the warm
   steady state instead of 43x), and the in-request compiler time is down from ~6.9 s to ~0.5 s. §4.

## 2. The named list this card asked for

A SYCL program here IS one `strata::sycl_compat::launch(...)` call site (the device code is split per kernel,
`-fsycl-device-code-split=per_kernel` at compile AND link time). Each compiler-cache entry carries that site's
SPIR-V, whose NUL-separated name section holds the mangled C++:

```
typeinfo name for strata::sycl_compat::launch<strata::kernels::<SITE>(args)::{lambda...}>::operator()<...>
void strata::kernels::(anonymous namespace)::<KERNEL><template args>(unsigned char*, ..., launch_shape, ...)
```

`c++filt` turns the first into the LAUNCH SITE (the helper in `src/kernels/cuda/*.cu`) and the second into the
exact KERNEL instantiation when the kernel is a real function template. `d2c/enumerate_c.py` does this for every
`0.src` written between the engine's own `prompt … read in N ms` line and the ask's end, which is what turns
"61 programs" into a list auditable line by line.

Warmed by `decode_warmup()` (36 launch sites, `d2c/ENUMERATION.txt` §4 lists them with the argument each one is
called with and why it is warmable):

* the BF16 MMVF family — its block size is a **host** function of `n_in` (`mmvf_block_size`) and its column count
  a host branch of `n_tok`, so the pass spans both (`<160>`/`<256>`, `<256,4>`/`<256,8>`);
* the routed experts — one `native_expert_grouped` call per `(gu_type, d_type)` pair the pack carries
  (`pack/native_experts.txt`), because the kernel is templated per TYPE: that is what builds
  `native_gu_multi_kernel<18/21/22/23>` and `native_down_multi_kernel<20/42>`;
* the drafter's `moe_grouped_s2` (whose `gu_grouped_t_kernel`/`down_grouped_t_kernel` the alignment-selected fast
  path picks), `moe_group_resident`, `moe_hit_add`;
* the GDN family (`gdn_conv_l2_multi`, `gdn_conv_commit`, `gdn_ab_multi`, `gdn_step_norm_multi`);
* the MTP/sampler tail (`mtp_select`, `native_router_top10(+_multi)`, `sample_tokens`, `row_top_prob`, `map_ids`);
* the mixer/MoE family (`native_moe_combine(+_multi)`, the four `native_gr_*`, `native_qsa_gate_apply`);
* PLE history, and the verify window's one-off copies/flags/quantizers (15 sites).

Bound, with the reason (`d2c/ENUMERATION.txt` §4):

* **`qsa_decode_attn_batch`** (`attn_chunk_kernel<1>`, `attn_merge_kernel`), **`qsa_block_scores`**,
  **`native_qsa_indexer_append`** — they read the QSA attention pools and the indexer's tail/dead/pooled buffers
  through a per-page table whose geometry a *request* builds (`QsaAttnPools`, `QsaIndexerBuffers`, `QsaShapes`).
  A dummy pool is not a smaller pool: the table indexes it, so zeroed table and zeroed pool are only safe if
  their arithmetic agrees with the real geometry, and that geometry does not exist yet in the load phase.
  Measured share of the phase: ~200 ms of the 1918 ms residual (4 of the 15 remaining programs).
* **`shared_expert` / `shared_expert_multi`** and **`native_ple_postops`** — they dispatch on the pack's own
  weight types (`NativeSharedWeights::gate_type/up_type/down_type`, `SForm`, `PleWeights`), not on a compile-time
  table, so warming them with a guessed type builds a program the request does not use — which D2b measured to be
  strictly worse than not warming. ~11 of the 15 remaining programs. Plumb the pack's type fields into the pass
  and they become warmable the same way.
* The dense (quant type × ncols) MMVQ family is D2b's pass and is not repeated here.

## 3. Mechanism, and the two faults it took to get it right

`decode_warmup(void* stream)` (`src/kernels/cuda/decode_warmup.cu`), called from the serve path in
`src/program/generate.cpp` immediately after D2b's `native_mmvq_warmup(main_cs)` — still before the
"everything loaded" line, before the ask can arrive (the FIFO is fed after that line) and before any window or
graph capture has run. It allocates one ~8 MiB scratch arena (~4.75 MiB of f32, plus u16/u8/i32/u64/u32
regions), zeroes it once, and calls each site once on the caller's stream. Every count it steers is a **zeroed
device word**, so the kernels return from their own first guard (`g >= *n_groups`, `k < *n`, `n <= 0`,
`n_keep == 0`, `value 0`); nothing it computes is read by anything. The ids guard (§4) is the evidence, not the
argument. `STRATA_KERNEL_WARMUP=0` is its A/B arm, separate from D2b's `STRATA_MMVQ_WARMUP=0`, so the two passes
are measured apart. `STRATA_KERNEL_WARMUP_TRACE=1` adds one line per site and one stream sync per site, which is
how the two faults below were found; it is off by default because the per-site sync would contaminate the pass's
own timing (and the timing is one of the things measured).

Two things this pass got wrong first, both worth recording because both read as "the kernel is broken" and are
not:

* **A host write to device memory.** The first version set `grp_ptr[0]` from the host (`a.U64(0)[0] = …`) so the
  kernel would have a valid blob pointer. The arena is device memory, so that is an invalid access: the engine
  **segfaulted (exit -11)** right at the routed-expert site. It is unnecessary — with `n_groups` zeroed the
  kernels return before loading `grp_ptr` — and the array is now left zeroed.
* **Aliasing between the arena's control words.** The first version reused one int32 slot for several sites'
  in-counts and out-buffers. `moe_group_resident` writes `grp_start` there, and a later `fetch_blobs` read the
  same word back as its blob count and dereferenced the (zero) pointer it found: the device faulted, the context
  was poisoned, and the engine died with **exit -6** during the ask. Every word a kernel READS as a count, index
  or length now has its own slot that nothing writes (`d2c/ENUMERATION.txt`, the `I_*` table in the source).

## 4. The measurements

All arms: 4K, 256 new tokens, `--spec-min-p 0.7`, both GPUs, one engine at a time; a "cold" arm has **both**
`SYCL_CACHE_DIR` and `NEO_CACHE_DIR` pointed at fresh directories (a fresh install — the shared warm caches were
never moved or deleted).

### 4a. The cold first runs — the number this card is for

| arm | warm-ups | programs in the ask (prompt + **decode**) | residual ms/win | residual × win | load | ask | **start → first answer** | ids |
|---|---|---|---|---|---|---|---|---|
| `d2c-nooff-cold-4096` | none | 152 (59 + **93**) | 128.26 | 6926 ms | 32 s | 33.28 s | **65.28 s** | `66bf952d…` |
| `d2c-mmoff-cold-4096` | D2b's | 112 (51 + **61**) | 35.51 | 1918 ms | 49 s | 28.23 s | **77.23 s** | `66bf952d…` |
| `d2c-wu-cold-4096` | **both** | 66 (51 + **15**) | 9.46 | 511 ms | 52 s | 26.20 s | **78.20 s** | `66bf952d…` |

The pass's own lines: `strata mmvq warmup: 192 launches …, 17453 ms` + `strata kernel warmup: 54 launches over
54 sites, 2287 ms` on the cold run. Whole-run cache census: 157 → 217 → 221 `0.src` entries.

What this card's pass does, isolated: **61 → 15 programs in the decode phase** (46 removed; the 15 that remain
are exactly the bound set of §2 plus `native_gr_rms_norm_weighted -> weighted_rms_norm<1024>`, see below),
**1918 → 511 ms** of residual, **28.23 → 26.20 s** of ask, for **49 → 52 s** of load.

**One site the pass warms with the wrong instantiation, stated rather than hidden:**
`native_gr_rms_norm_weighted` calls the kernel with a tile constant derived from `n_cols`; the pass calls it with
`n_cols = 512` and the request reaches `weighted_rms_norm<1024>`, so that one program still builds in the request
(~40 ms). The remedy is one number in `decode_warmup.cu` (call it with the model's own `n_embd`), and it is left
as a named follow-up rather than folded in after this card's chain had been measured as a whole.

### 4b. The warm controls (same session, shared warm caches)

| arm | warm-up lines | residual ms/win | programs in the ask | load | ask | start → first answer | ids |
|---|---|---|---|---|---|---|---|
| `d2c-wu-4096` | mmvq 39 ms, kernel 5 ms | 2.99 | 0 | 31 s | 21.16 s | 52.16 s | `66bf952d…` |
| `d2c-mmoff-4096` | mmvq 39 ms, kernel off | 2.80 | 0 | 31 s | 21.16 s | 52.16 s | `66bf952d…` |
| `d2c-wu-32768` | mmvq 43 ms, kernel 6 ms | 1.38 | 0 | 32 s | 102.73 s | 134.73 s | `d87373e8…` |

The residual stays at the shipped ~3 ms/window (2.99 with the pass on, 2.80 with it off, 1.38 at 32K) — the pass
does not perturb the steady state, and once its programs exist it costs **5-6 ms** of load. Every arm's ids are
canonical.

## 5. Tests

`d2c/d2c_ctest.sh` (the S4 convention: oneAPI sourced, `ZE_AFFINITY_MASK=0`, `make -j8` first because the tests
relink the new `decode_warmup` symbol):

| what | result |
|---|---|
| `ctest -R mmvq_multi_parity` (the contract D2b's pass launches against) | **1/1 passed**, 1.96 s |
| the families whose kernels this pass warms (`bf16_gemv`, `elementwise`, `quantize_act`, `s2_expert_grouped`, `mmvq`, `iq_multi`, `sampler`, `qsa_parity`, `gdn_parity`, `gr_parity`, `router_top10`, `expert`) | 12/16 passed; failed: `elementwise_parity`, `quantize_act_parity`, `iq_multi_parity`, `expert_multi_test` |
| the full suite (the S4 ctest set, 49 tests) | **5 failed of 49** — `platform_memory_test`, `elementwise_parity`, `quantize_act_parity`, `iq_multi_parity`, `expert_multi_test` |
| the recorded baseline (`s4/runs/ctest/full.log`, "5 failed of 49") | **the same five names** |

The pass adds no kernel and is not reachable from any test (it is called from the serve path of the engine
binary only), and it launches kernels a request may never use — which is what these two runs bound: it changes no
test outcome.

## 6. What this card's numbers say about the trade-off

The card's premise was that the 61 remaining builds are why a fresh install is 12.4 s worse with D2b's pass than
without it, and that warming them turns that positive. The measurement says otherwise, and the arithmetic is
short:

* the 61 programs are **one program per launch site** (the enumeration shows every site appearing exactly once),
  so a cold 4K request needs all 61. A warm-up for them therefore builds the same set the request would have
  built: it **moves** work out of the ask and into the load, it does not remove it;
* and the move is not free in either direction. Measured: the pass built 46 of the request's programs in
  **2287 ms** (≈50 ms each) and removed **1407 ms** of residual (≈31 ms each). Per program, a load-phase build is
  the *same order* as an in-ask build — D2b's own pair (162 ms in-ask against 182 ms in the load phase) says the
  same — so the sum cannot come out ahead. Measured totals: **+2.29 s of load, −2.03 s of ask, +0.97 s of
  start → first answer**.
* the 12.9 s the fresh install still pays (78.20 s against the shipped 65.28 s) is not these 61 programs: it is
  D2b's pass building **96 specializations where a cold 4K request uses 32** (D2b §6's own numbers). 64 unused
  programs at ~180 ms each is ~11.5 s, i.e. essentially all of it.

So the trade-off is turned positive by **building fewer programs, not by building them earlier**: D2b's dense
pass has to be narrowed to the `(type, ncols, shape)` the window sequence actually reaches — which needs the
real tensor `n_in` per type in `native_mmvq_warmup` (the shape crossing is `n_in / F::DIV < F::BPI`, a property
of the loaded model, not of the run) and would cut the ~96 down to ~48 — or the first request has to stop paying
for programs a load phase already built. That is a `native_mmvq_warmup` change with its own ids guard, and it is
the one lever this card's measurements point at; it is **not** done here, because this card's pass is a separate
change and both were measured apart.

What is bought in exchange for the +0.97 s is the request itself, and on a fresh install that is the number a
user waits on: the first request's decode windows build **15 programs instead of 93**, its residual is
**9.46 ms/window instead of 128.26** (3.2x the warm steady state 2.99 instead of 43x), its in-request compiler
time is **~0.5 s instead of ~6.9 s**, and the decode wall is 8.6-9.7 s instead of 13.7 s. On every later request
the pass costs 5 ms.

## 7. Machine state, and how to re-run

Machine: **left clean** — the last arm (`d2c-wu-32768`) and the `ctest` run were allowed to finish; no engine
(`pgrep -x strata` empty), no `serve/server.py`, port 8099 free, `ZE_AFFINITY_MASK` unset, both cards free. The
shared warm caches were never moved or deleted; they gained this pass's programs once (ordinary cache growth, and
the reason a second run's pass costs 5 ms). Nothing was pushed (as in D2/D2a/D2b: `origin` has no `sycl-xpu`
branch). D2b's pre-change binary is preserved at `/home/michael/strata-xpu/d2b/strata-e841fd06` and this card's
"before" binary at `/home/michael/strata-xpu/d2c/strata-f49fe666`; the cold arms' caches are under
`/tmp/d2c-cold{1,2,3}-cache`.

```bash
# the three cold first runs and the three warm controls (~8 min)
bash d2c/d2c_chain_a.sh
# the engine rebuild (~1 min)
bash d2c/build_engine.sh
# the tests (needs the GPU; ZE_AFFINITY_MASK=0, the S4 convention)
bash d2c/d2c_ctest.sh
# the named list of what a cold decode phase builds (~1 min)
bash d2c/make_enumeration.sh d2c-mmoff-cold-4096 d2c-wu-cold-4096   # -> d2c/ENUMERATION.txt
# every number of this card
bash d2c/d2c_evidence.sh                                            # -> d2c/D2C-EVIDENCE.txt
```
