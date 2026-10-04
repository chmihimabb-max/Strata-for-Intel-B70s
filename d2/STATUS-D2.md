# D2 — the decode path's dispatch, its precision, and the `--spec-min-p 0.7` win landed

Card `t_0416a0c0` (D2), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`. One measurement session, both
B70s at a time, `ZE_AFFINITY_MASK` unset, one engine at a time, the config of record verbatim except the one lever
each arm names.  The file (`strata-sycl-iq3s.json`) is: IQ3_S GSQ-RCO snapshot `ed59f920…`, `--kv int8
--kv-resident 32768 --expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts --prefill 512
--spec 4 --spec-min-p 0.5 --max-context 262144 --no-capture --stats`, MTP drafter `~/strata-xpu/mtp/rt`,
`gpu [0,1]` → `--layer-split auto` → **K=23** (CUDA0 layers 0-22, CUDA1 layers 23-47 and the head), 24 576/24 576
experts resident (11 776 slots/21.20 GiB card 0 + 12 800/25.63 GiB card 1), 100% expert residency, 0.00 CPU expert
entries per layer-window.

Engine binaries, and every arm's own log carries the md5 it ran:

| binary | md5 | what it is |
|---|---|---|
| the arms up to and including chain2 | `e79b2ad632d37d32786228b660d35b29` | HEAD's `build-sycl/strata`: D1's graph path + histogram + S4's checkpoint fix (S4 measured it byte-identical in numerics to D1's) |
| the arms from `d2-rebase-4096` on | `e841fd061fec873c2f24e785973a2ebe` | the same, rebuilt with this card's ONE added switch (`STRATA_MMVQ_MULTI_GENERIC`, default off) |

Two documented deviations from the file, both inherited from D1's rig and both stated rather than hidden: the arms
pass **`--prefill auto`** (the file says `512`; auto picks the 8 192-token chunk and is 1.47-1.56x faster on the
prompt path, and the decode numbers this card is about are unaffected), and they add **`--prompt-cache 0
--prompt-cache-every 0`** (S4's split OOM at the engine's default 16 384-token mid-prompt checkpoint). The
`--max-context` is the arm's own CTX.

Rig under `d2/`, raw output under `d2/runs/` (the card asks for both inside the repo):
`d2/d2_run_arm.sh` (D1's arm runner retargeted at `d2/runs`: one engine, the ask fed from a FIFO after "everything
loaded", md5 + device check + every engine line in the arm's own log), `d2/d2_chain0.sh` (task 0), `d2/d2_chain1.sh`
(tasks 1-3), `d2/d2_chain2.sh` (the expert-kernel variant), `d2/d2_chain3.sh` (the multi-column layout variant,
after its build), `d2/d2_report.py` (one row per arm out of that arm's own logs), `d2/d2_census.py` (the window's
census, attribution read off the source), `d2/d2_types.py` + `d2/D2-TYPES.txt` (the tensor→type census),
`d2/d2_sites.py` (per-site counts), `d2/d2_evidence.sh` → `d2/D2-EVIDENCE.txt` (every raw line the acceptance
asks for), `d2/D2-CENSUS.txt`, `d2/config-before-d2.json` (the file as D2 found it).  Commits: `3da0feb` (rig +
census), `569cfdb` (the switch + every arm), plus the write-up.

## 1. Verdict

1. **`--spec-min-p 0.7` REPRODUCES in this session and is now the config of record.** Byte-identical greedy ids at
   all three lengths: **+4.6% at 4K (23.03 → 24.08 tok/s), +3.2% at 32K (23.62 → 24.38), a wash at 128K (-0.2%,
   21.47 → 21.42)**, on smaller windows with better acceptance (0.727 → 0.845 at 4K, 0.804 → 0.886 at 32K,
   0.776 → 0.875 at 128K).  D1 measured +5.8/+3.3/-0.8; the difference is the arm band.  Landed: one line in
   `strata-sycl-iq3s.json` (§2).
2. **The port IS choosing the fast path — and D1's own census was labelling it wrong.**  `launch_multi_n<F,4>` is
   not the experts' mat-vec: it is `native_mmvq.cu:1046`, the **dense per-weight-matrix multi-column MMVQ**, one
   launch per native dense matrix, and the window has exactly 300 of them plus the head — which the engine prints
   itself (`300 native projection matrices, 2018.88 MiB of weights`).  Corrected 4K census: **dense MMVQ 40.9%**,
   expert down 14.2%, expert gate+up 12.4%, flag handshake 10.7%, mixer read 9.9%, copies 3.1%, bf16 projections
   2.6%, QSA 2.6%.  The "65.5% expert chain" of the card's premise was 40.9% dense projections + 26.6% experts.
3. **Every alternative the engine can select, measured: the shipped choice wins.**  Per-token instead of
   multi-column decode dispatch: **-4.0% / -3.3%** (4K/32K, ids identical).  The old one-entry expert kernels
   (`native_gu_kernel`/`native_down_kernel`): **-7.6% / -2.0%**, ids identical.  The plain hyper-connection read (vs
   the shipped staged one): -0.9%, ids identical; split (-0.5%) likewise.  llama.cpp's generic multi-column layout
   for the dense MMVQ: **-9.7% at 4K (warm) and +5.5% at 32K, and it moves the greedy ids** — refused.
4. **Two levers move token ids and are therefore refused as correctness changes, not shipped** (P5's lesson):
   the generic multi-column layout (§5) and `STRATA_WINDOW_PLAIN_GR` (the unfused per-token mixer read: -21.7% and
   different ids).  The generic layout is worse than that: **three identical runs of it produced three different
   greedy streams (137 / 148 / 185 tokens) and three different window costs (191.79 / 140.39 / 120.68 ms), while
   every one of the eight shipped-layout 4K arms of this session produced the same stream** (`66bf952d…`) **and
   125-126 ms/window**.  Both levers are named for follow-up rather than acted on (§8).
5. **No F32 GEMV exists on this config's decode path**, so there is nothing to convert for precision's sake: every
   F32 tensor the window touches is a 1-D norm/conv/bias/gate vector consumed by an elementwise kernel, and both
   GEMV families on the path already run the low-precision contract their weights ask for (native MMVQ: int8 Q8_1
   activations; `mmvf`: FP32 activations against BF16 weights).  The one genuine "fast path not taken" is that the
   **40.9% dense MMVQ is a scalar FMA integer dot while the device has int8 XMX** — a kernel rewrite, not a
   dispatch flip, and named as such (§6).

## 2. Task 0 — the `--spec-min-p 0.7` re-verification, and the landed diff

Two arms per length, same binary (`e79b2ad6…`), `--spec 4` in both, nothing else changed.  Every number is the
engine's own (`d2/d2_report.py`; the raw lines are below and in `d2/D2-EVIDENCE.txt`).

| length | arm | windows | avg T | tokens/window | **ms/window** | **decode tok/s** | drafts accepted | acceptance | **greedy ids md5** |
|---|---|---|---|---|---|---|---|---|---|
| 4K | `d2-minp05-4096` (`--spec-min-p 0.5`) | 51 | 3.59 | 2.88 | **125.14** | **23.03** | 96/132 | 0.727 | `66bf952d445e330c974c49e4f220b4b1` |
| 4K | `d2-minp07-4096` (**0.7**) | 54 | 3.04 | 2.72 | **113.07** | **24.08 (+4.6%)** | 93/110 | 0.845 | `66bf952d…` **identical** |
| 32K | `d2-minp05-32768` (0.5) | 87 | 3.46 | 2.94 | **124.58** | **23.62** | 172/214 | 0.804 | `d87373e84417dab60732a95eb47666a6` |
| 32K | `d2-minp07-32768` (**0.7**) | 92 | 3.01 | 2.78 | **114.13** | **24.38 (+3.2%)** | 164/185 | 0.886 | `d87373e8…` **identical** |
| 128K | `d2-minp05-131072` (0.5) | 99 | 3.07 | 2.59 | **120.47** | **21.47** | 159/205 | 0.776 | `511a89be345a6583c9bc8c580259d73e` |
| 128K | `d2-minp07-131072` (**0.7**) | 118 | 2.36 | 2.17 | **101.26** | **21.42 (-0.2%)** | 140/160 | 0.875 | `511a89be…` **identical** |

Those three id streams are the SAME md5s D1's arms produced, so this session re-ran D1's token streams exactly,
and the 0.5 arms reproduce D1's own numbers (4K 125.14 vs D1's 125.71 pre-change/126.45 in the sweep, 23.03 vs
22.79; 32K 124.58/23.62 vs 124.57/23.62; 128K 120.47/21.47 vs 120.42/21.47).  Mechanism, as D1 priced it: 0.7 is a
"shorter, surer window" policy — the same or better acceptance on smaller windows (2.72 vs 2.88 tokens at 4K) —
and §2 of D1's write-up showed the window's cost is convex in the tokens it verifies, so smaller-and-surer wins at
4K/32K and cancels at 128K (its 101.26 ms window carries 2.17 tokens against 120.47 carrying 2.59).

The raw engine lines (acceptance asks for them; all six are in `d2/D2-EVIDENCE.txt`):

```
# 4K, 0.5            strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15205 ms (252.0 tok/s), 147 generated in 6382 ms (23.0 tok/s), drafts accepted 96 of 132, 0 checkpoints
# 4K, 0.7            strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15160 ms (252.8 tok/s), 147 generated in 6106 ms (24.1 tok/s), drafts accepted 93 of 110, 0 checkpoints
# 32K, 0.5           strata serve: prompt 32256 tokens = 0 reused + 32256 read in 92002 ms (350.6 tok/s), 256 generated in 10839 ms (23.6 tok/s), drafts accepted 172 of 214, 0 checkpoints
# 32K, 0.7           strata serve: prompt 32256 tokens = 0 reused + 32256 read in 91706 ms (351.7 tok/s), 256 generated in 10500 ms (24.4 tok/s), drafts accepted 164 of 185, 0 checkpoints
# 128K, 0.5          strata serve: prompt 129024 tokens = 0 reused + 129024 read in 372757 ms (346.1 tok/s), 256 generated in 11926 ms (21.5 tok/s), drafts accepted 159 of 205, 0 checkpoints
# 128K, 0.7          strata serve: prompt 129024 tokens = 0 reused + 129024 read in 372722 ms (346.2 tok/s), 256 generated in 11949 ms (21.4 tok/s), drafts accepted 140 of 160, 0 checkpoints
# and the window lines (ms/window, verify and its parts):
strata decode timing: 51 windows, avg T 3.59, 2.88 tokens/window, 125.14 ms/window = verify 111.56 (GPU-reach wait 93.55 + per-layer host 1.03 [plan 0.30 actq 0.24 jobs 0.22 CPU 0.00] + stage 1.32 + tail 12.44) + commit/emit 1.89 + draft 11.70; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 35.88, PCIe 0.00
strata decode timing: 54 windows, avg T 3.04, 2.72 tokens/window, 113.07 ms/window = verify 100.90 (GPU-reach wait 84.36 + per-layer host 0.94 [plan 0.28 actq 0.21 jobs 0.19 CPU 0.00] + stage 1.17 + tail 10.93) + commit/emit 1.87 + draft 10.30; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 30.37, PCIe 0.00
strata decode timing: 92 windows, avg T 3.01, 2.78 tokens/window, 114.13 ms/window = verify 100.12 (GPU-reach wait 85.11 + per-layer host 0.91 [plan 0.28 actq 0.20 jobs 0.19 CPU 0.00] + stage 1.38 + tail 10.95) + commit/emit 1.90 + draft 12.11; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 30.11, PCIe 0.00
strata decode timing: 118 windows, avg T 2.36, 2.17 tokens/window, 101.26 ms/window = verify 89.56 (GPU-reach wait 76.77 + per-layer host 0.81 [plan 0.27 actq 0.16 jobs 0.15 CPU 0.00] + stage 1.55 + tail 9.24) + commit/emit 1.80 + draft 9.90; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 23.56, PCIe 0.00
```

**The landed diff** (the file is local/gitignored; the pre-D2 copy is committed at
`d2/config-before-d2.json`, md5 `4ece57018c7ed9c9692d914af3d8d9b6`, the landed file md5
`e787844fd2c564817583e3ae5d34f15c`):

```diff
--- a/strata-sycl-iq3s.json   (d2/config-before-d2.json, md5 4ece57018c7ed9c9692d914af3d8d9b6)
+++ b/strata-sycl-iq3s.json   (as landed, md5 e787844fd2c564817583e3ae5d34f15c)
@@ args
-          "--spec-min-p", "0.5",
+          "--spec-min-p", "0.7",
@@ _note
-…that pre-P6 file is kept byte-identical at /home/michael/strata-xpu/p6/config-before-p6.json (md5 2fe9a4b711fbd704abc95958166f74fc).",
+…that pre-P6 file is kept byte-identical at /home/michael/strata-xpu/p6/config-before-p6.json (md5 2fe9a4b711fbd704abc95958166f74fc). D2 (card t_0416a0c0, 2026-10-03): --spec-min-p 0.5 -> 0.7, after re-verifying D1's win in a fresh session at 4K/32K/128K with byte-identical greedy ids (+4.6%/+3.2% decode at 4K/32K, a wash at 128K - strata/d2/STATUS-D2.md); the pre-D2 file is kept byte-identical at strata/d2/config-before-d2.json (md5 4ece57018c7ed9c9692d914af3d8d9b6). Nothing else in this file changed in D2.",
```

Nothing else in the file changed, and no other file of the config's meaning changed.  The 128K -0.2% is inside the
arm band and is a wash, not a win — landed anyway because the policy is the same trade at every length
(smaller/surer window, equal-or-better acceptance, identical ids) and 4K/32K are the lengths this config serves.

## 3. The dispatch map (tensor class → kernel → file:line → predicate)

The decode path is `Verifier::record_window` (`src/core/verify.cpp:708`) — one window, two stages (K=23: CUDA0
layers 0..22, CUDA1 layers 23..47 and the head), captured as ONE SYCL `command_graph` and replayed (D1's default;
`--no-capture` in the config of record is inert on a native pack, `src/program/generate.cpp:3202/3229-3239`).  The
type of every tensor below is in `d2/D2-TYPES.txt`, read from the GGUF with `tools/gguf_reader.py` — the same types
`NativeDense` reads (`src/core/native_dense.cpp:16-28`).

The pack's decisive fact: **the 302 non-expert dense tensors that a GEMV touches are all served NATIVELY**
(`--native <shard 1>`; the engine prints `300 native projection matrices, 2018.88 MiB of weights` and `302 canonical
tensors skipped: served natively`), so `gemv_quantized`'s FIRST branch — `w.native_data`
(`src/core/layer.cpp:142-158`) — is the one that fires for every quantized projection, and the canonical
`code_bits == 2 ? s2_gemv_q8 : (wants_q8k ? s_gemv_q8k_split : s_gemv_q8_0_split)` tail (`src/core/layer.cpp:163-172`)
is unreachable.  `native_of()` (`src/core/verify.cpp:898-900`, `:935-937`) would have failed the window outright if
any of them were not native.

| tensor class (count in the window) | pack type(s) | kernel | file:line | predicate |
|---|---|---|---|---|
| `attn_qkv.weight` (36 GDN layers) | Q6_K 22, Q5_K 10, IQ4_XS 2, Q4_K 2 | `native_mmvq` → `native_q{6,5,4}_k_mmvq` / `native_iq4_xs_mmvq` → `launch_multi` → **`launch_multi_n<F,4>`** (`native_mmvq_multi_kernel`) | call `src/core/verify.cpp:909`; dispatch `src/kernels/cuda/native_mmvq.cu:1475` (switch on `ggml_type`) → `:1371`/`:1332`/`:1293` → `:1064` → `:1046` | `w.native_data` ⇒ `native_mmvq`; `ncols = n = 4` (the window's rows) ⇒ the multi-column launcher |
| `attn_gate.weight` (36 GDN) | Q6_K 14, Q4_K 10, IQ4_XS 8, Q5_K 4 | same | `src/core/verify.cpp:917` | same |
| `ssm_out.weight` (36 GDN) | Q6_K 29, Q5_K 4, Q4_K 3 | same | `src/core/verify.cpp:924` | same |
| `attn_k.weight`, `attn_v.weight` (12 QSA each) | Q6_K 11 + 10, IQ4_XS 1, Q5_K 2, Q4_K 1 | same | `src/core/verify.cpp:953`, `:954` | same |
| `attn_q.weight` (12 QSA) | Q6_K 5, IQ4_XS 3, Q5_K 3, Q4_K 1 | same | `src/core/verify.cpp:988` | same |
| `attn_output.weight` (12 QSA) | Q6_K 11, Q4_K 1 | same | `src/core/verify.cpp:1044` | same |
| `ffn_{gate,up,down}_shexp.weight` (48 layers each) | IQ4_NL 47/Q8_0 1, IQ4_XS/Q4_K/Q5_K/Q6_K | same (`small_mmvq<IQ4NLBlock,4>` for IQ4_NL, `Q80Block,8` for Q8_0) | `src/core/verify.cpp:1090` (`shared_expert_multi` → `src/kernels/cuda/shared_expert.cu:251-300`) | `native.gate_data != nullptr`; else the `s2_gemv_q8` / `s_gemv_*` tail |
| **routed experts** `ffn_{gate,up}_exps` (48 layers) | IQ3_XXS 17, IQ2_S 20, IQ3_S 10, IQ4_XS 1 (`pack/native_experts.txt`) | `native_expert_grouped` → `launch_gu<TG>` → `native_gu_multi_kernel` (GRP_NC=4) | call `src/core/verify.cpp:1135`; dispatch `src/kernels/cuda/iq_kernels.cu:1590`, `:1603-1608` → `:1438` (`:1444` multi) | `expert_layout().native`; env `STRATA_OLD_IQ_MMVQ` forces `native_gu_kernel` instead |
| **routed experts** `ffn_down_exps` (48 layers) | IQ4_NL 39, Q2_0 9 | `launch_down<TD>` → `native_down_multi_kernel` | `src/kernels/cuda/iq_kernels.cu:1619-1624` → `:1447` (`:1452` multi) | as above |
| expert epilogue (96 calls) | — | `swiglu_entries_kernel`, then `quantize_q8_1_kernel` (or `_sumq` for the `Fmt<103>` types) | `src/kernels/cuda/iq_kernels.cu:1611`, `:1614-1617` | always, inside `native_expert_grouped` |
| router `ffn_gate_inp.weight` (48) | BF16 | `bf16_gemv_fp32_mmvf_multi` (llama.cpp `mmvf` = FMA dot, **not** mma/XMX) | `src/core/verify.cpp:1057`; kernel `src/kernels/cuda/native_bf16.cu:76-119` | `dec_batch && n > 1 && native_router_enabled() && NE == 512 && K == 10`; else per-token `moe_route` → `project_bf16` (`src/core/layer.cpp:94-100`) |
| router top-10 | — | `native_router_top10_multi` | `src/core/verify.cpp:1059` | same (`STRATA_DEC_BATCH=0` → `router_top10`) |
| `indexer.k_proj.weight`, `indexer.q_proj.weight` (12 QSA each) | BF16 | `bf16_gemv_fp32_mmvf_multi` | `src/core/verify.cpp:949`, `:998` | `qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled() && !st.kv_q4`; else `bf16_gemv_fp32_mmvf` per token (`:951`, `:1014`) |
| hyper-connection mixers `hc_{attn,ffn}_{down,up,inject}` (48 layers each) | BF16 | `fused_gr_read_multi` → `launch_multi` (3 kernels: the norm read, the down/up pair, the inject) | call `src/core/verify.cpp:880`; launcher `src/kernels/sycl/fused_gr.cpp:840`, `:1045` | `layer_set_fused_gr(...)`; the variant is `fused_gr_variant()` = **staged** on both cards (the engine's own on-card bit-for-bit check against the plain read passes and prints so), forced by `STRATA_HC_SPLIT` (0 plain / 1 split / 2 staged); `STRATA_WINDOW_PLAIN_GR=1` bypasses the fused read for `gr_read` (`src/kernels/cuda/gr.cu:761`) |
| head `output.weight` (1) | Q6_K | `native_mmvq` → `launch_multi_n<Q6KTraits,4>` (the 301st dense launch) | `src/core/verify.cpp:1229` | `head_->loaded()`; else `lm_head_mix`/`lm_head` |
| `ssm_alpha/beta` (36), `ssm_{dt,a,norm,conv1d}`, all norms | BF16 (α, β) / **F32** (the rest) | elementwise / norm kernels — **no GEMV** | `gdn_ab_multi`, `gdn_conv_l2_multi`, `gdn_step_norm_multi`, `native_gr_rms_norm_weighted`, `native_rope_apply` | — |
| `token_embd.weight` | IQ4_XS | `iq_embed_rows` | the window's embed step | — |

Two things this map settles, both of which the card asked for:

1. **The six shipped GEMV variants are not on this config's decode path.** `s2_gemv.cu`, `s2_gemv_fast.cu`,
   `s2_gemv_q8.cu`, `s2_gemv_quads.cu`, `s_gemv.cu` and `bf16_gemv.cu` are the CANONICAL (pack-plane) family, and
   the native pack bypasses them (above).  What IS on the path is `native_mmvq.cu`'s per-type MMVQ,
   `iq_kernels.cu`'s grouped expert kernels, and `native_bf16.cu`'s `mmvf` — so the alternatives worth forcing are
   §5's, and the canonical family's own parity tests (`s2_gemv_parity`, `s_gemv_parity`, `s_gemv_q8k_parity`,
   `s2_gemv_q8_parity`, `bf16_gemv_parity`) are the only place they run at all on this box.  `bf16_gemv`/`split`'s
   `project_bf16` (`src/core/layer.cpp:94-100`) is likewise reached only from the non-native `moe_route` path.
2. **`launch_multi_n<…>` is the DENSE projection MMVQ, not the experts'.**  D1's `d1_families.py` maps
   `launch_multi_n` to "expert MMVQ — the native expert mat-vec, one launch per expert group", and that is the
   attribution error this card found first: `launch_multi_n` is `native_mmvq.cu:1046` (the host launcher of
   `native_mmvq_multi_kernel`), reached only from `native_mmvq()`'s `ncols > 1` path (`:1064-1078`), one launch per
   (native weight matrix, window).  The engine's own count makes it unambiguous: **301 launches, and the window has
   exactly 300 native dense matrices** (36 GDN × 3 + 12 QSA × 4 + 48 × 3 shexp = 300) **plus the head**; the
   per-type table agrees to the unit (Q6_K 129 = 128 dense Q6_K + the Q6_K head).  The experts are
   `launch_gu`/`launch_down` (`iq_kernels.cu:1438/1447`), reached only through `native_expert_grouped`.  §4 is the
   corrected census.

## 4. The corrected census (one T=4 decode window, closure path, device microseconds)

`d2/d2_census.py` — D1's instrument (`STRATA_LAUNCH_HIST=1`; the closure path so every launch has a device
timestamp), the attribution read off the source instead of off the symbol's spelling.  Arm `d2-hist-4096` (3 810
counted submissions, 125.4 ms of device time, both stages summed; `d2/D2-CENSUS.txt`):

| family (what reaches it) | count | µs | %count | **%time** |
|---|---|---|---|---|
| **dense native MMVQ** (`launch_multi_n<F,4>`: one launch per native weight matrix, `native_mmvq.cu:1046`) | 301 | 51 267.5 | 7.9% | **40.9%** |
| **routed experts: down** (`launch_down<TD>`, `iq_kernels.cu:1447`) | 96 | 17 774.9 | 2.5% | **14.2%** |
| **routed experts: gate+up** (`launch_gu<TG>`, `iq_kernels.cu:1438`) | 96 | 15 559.3 | 2.5% | **12.4%** |
| flag handshake, device side (`wait_flag_ge`, `doorbell_publish`) | 192 | 13 395.3 | 5.0% | **10.7%** |
| mixer read (`launch_multi`, `fused_gr.cpp:840`: norm read + down/up + inject, 3 sites × 96 calls) | 288 | 12 404.6 | 7.6% | **9.9%** |
| copies / mapped staging (`<memcpy>` 1 306, `*_from_mapped`) | 1 449 | 3 904.7 | 38.0% | 3.1% |
| bf16-weight projections (`bf16_gemv_fp32_mmvf(_multi)`: router 48, shexp scalar gate, indexer k/q per QSA layer, PLE) | 136 | 3 258.1 | 3.6% | 2.6% |
| QSA attention / indexer / norms (`qsa_*`, `kv_*`, `native_rope*`, `native_qsa*`, `fwht`) | 228 | 3 245.0 | 6.0% | 2.6% |
| GDN recurrence / shared expert (`gdn_*`, `conv*`, `gr_*`, `shared_expert*`) | 252 | 1 824.5 | 6.6% | 1.5% |
| routing / combine (`native_router_top10_multi`, `moe_*`) | 144 | 877.8 | 3.8% | 0.7% |
| routed experts: epilogue (`swiglu_entries_kernel`, `quantize_q8_1_kernel`) | 241 | 879.0 | 6.3% | 0.7% |
| routed experts: indexing (`native_expert_grouped`'s own launches) | 192 | 514.0 | 5.0% | 0.4% |
| other | 195 | 468.6 | 5.1% | 0.4% |
| **(total counted)** | 3 810 | 125 376 | | |

**The routed-expert chain is 27.7%** (down 14.2 + gate+up 12.4 + epilogue 0.7 + indexing 0.4), **not 65.5%**; the
40.9% D1 merged into it is the dense projections, whose two biggest single items are named by type:

| dense type | launches | µs | %time | | expert kernel | launches | µs | %time |
|---|---|---|---|---|---|---|---|---|
| Q6_K | 129 | 30 718.5 | 24.5% | | down IQ4_NL | 78 | 14 853.1 | 11.8% |
| IQ4_NL | 47 | 6 213.2 | 5.0% | | gate+up IQ2_S | 40 | 6 060.9 | 4.8% |
| Q5_K | 35 | 5 131.1 | 4.1% | | gate+up IQ3_XXS | 34 | 5 624.6 | 4.5% |
| IQ4_XS | 42 | 4 940.6 | 3.9% | | gate+up IQ3_S | 20 | 3 256.2 | 2.6% |
| Q4_K | 47 | 4 171.1 | 3.3% | | down Q2_0 | 18 | 2 921.8 | 2.3% |
| Q8_0 | 1 | 92.8 | 0.1% | | gate+up IQ4_XS | 2 | 617.5 | 0.5% |

The 32K arm's census is the same table within 0.5% (`d2-hist-32768`: dense 40.4%, down 13.8%, gate+up 12.1%,
handshake 11.1%, mixer 9.7%, QSA 3.6%, copies 3.1%) — i.e. **the split between the two projection families is
depth-independent**, the same conclusion D1 reached for its mislabelled version of the table.

What the instrument costs, measured in this session: `d2-hist-4096` reads 134.41 ms/window against the
non-instrumented closure arm's 131.27 (D1's, cross-binary) = **+2.4% for two dumped windows** (D1's +9% was for
eight); its greedy ids are identical to the baseline's (`66bf952d…`), and so are the 32K arm's (`d87373e8…`), i.e.
the instrument is arithmetic-neutral as designed.  The census is a one-off: nothing here changes the shipped path.

## 5. Task 2 — the variant table (every alternative the engine can select, forced and measured)

Baseline for every row is `d2-minp05-4096` (`--spec-min-p 0.5`, the config of record as D2 found it: 125.14
ms/window / 23.03 tok/s / ids `66bf952d…`) except the two arms that ran after the rebuild, whose control is
`d2-rebase-4096` (125.93 / 22.89 / ids `66bf952d…` — the new binary on the default levers, i.e. the same arm
within -0.6%).  One lever per arm, everything else the config of record, greedy ids as the guard.

| arm | lever (what it forces) | ctx | windows | ms/window | decode tok/s | vs baseline | **ids md5** | **ids equal?** |
|---|---|---|---|---|---|---|---|---|
| `d2-minp05-4096` | — (config of record) | 4K | 51 | 125.14 | 23.03 | — | `66bf952d445e330c974c49e4f220b4b1` | — |
| `d2-rebase-4096` | — (new binary, all defaults) | 4K | 51 | 125.93 | 22.89 | -0.6% | `66bf952d…` | **YES** |
| `d2-dec0-4096` | `STRATA_DEC_BATCH=0`: per-token GEMVs/routers/combine instead of one launch over the window's rows | 4K | 51 | 130.38 | 22.11 | **-4.0%** | `66bf952d…` | **YES** |
| `d2-dec0-32768` | same | 32K | 87 | 128.76 | 22.85 | **-3.3%** | `d87373e8…` | **YES** |
| `d2-hcsplit-4096` | `STRATA_HC_SPLIT=1`: the split hyper-connection read (default is staged) | 4K | 51 | 125.80 | 22.91 | -0.5% | `66bf952d…` | **YES** |
| `d2-hcplain-4096` | `STRATA_HC_SPLIT=0`: the plain reference read | 4K | 51 | 126.29 | 22.82 | -0.9% | `66bf952d…` | **YES** |
| `d2-oldiq-4096` | `STRATA_OLD_IQ_MMVQ=1`: `native_gu_kernel`/`native_down_kernel` (one entry per pass) instead of the `_multi` pair (4 per pass) | 4K | 51 | 135.47 | 21.28 | **-7.6%** | `66bf952d…` | **YES** |
| `d2-oldiq-32768` | same | 32K | 87 | 127.14 | 23.14 | **-2.0%** | `d87373e8…` | **YES** |
| `d2-mg-4096-warm` | `STRATA_MMVQ_MULTI_GENERIC=1`: llama.cpp's generic multi-column table for the 300 dense projections | 4K | 51 | 140.39 | 20.67 | **-9.7%** | `3b0519ca8196ffff312463ae4a09c0c1` | **NO** |
| `d2-mg-4096-warm2` | same (run 3 of 3, identical binary and lever) | 4K | 66 | 120.68 | **23.23** | **+1.5%** | `70182abfd4b4f78c02d823dc4f227f80` | **NO** |
| `d2-mg-4096` | same, first run (cold program cache for the newly instantiated kernel: the identical warm re-run is 51 ms/window faster) | 4K | 46 | 191.79 | 15.53 | -32.2% | `10160c38b7e991c84ec6f4cc01e26a55` | **NO** |
| `d2-mg-32768` | same | 32K | 84 | 122.24 | **24.93** | **+5.5%** | `59c8771250d15f238e348dd981dfdccc` | **NO** |
| `d2-plaingr-4096` | `STRATA_WINDOW_PLAIN_GR=1`: the unfused per-token `gr_read` + `gr_write` instead of the fused mixer read | 4K | 52 | 160.00 | 18.03 | **-21.7%** | `be7a906263db783e1dc5ab7932b2a282` | **NO** |
| `d2-hist-4096` / `-32768` | the instrument (closure path, 2 windows dumped) | 4K/32K | 49/87 | 134.41/130.60 | 22.32/22.53 | +2.4% (cross-binary) | `66bf952d…` / `d87373e8…` | **YES** |

Raw engine lines for the winning and losing arms (4K, the request's own line; the full set is in
`d2/D2-EVIDENCE.txt`):

```
# the shipped dispatch (baseline)
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15205 ms (252.0 tok/s), 147 generated in 6382 ms (23.0 tok/s), drafts accepted 96 of 132, 0 checkpoints
# per-token instead of multi-column (-4.0%)
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15022 ms (255.1 tok/s), 147 generated in 6649 ms (22.1 tok/s), drafts accepted 96 of 132, 0 checkpoints
# the old one-entry expert kernels (-7.6%)
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15146 ms (253.0 tok/s), 147 generated in 6909 ms (21.3 tok/s), drafts accepted 96 of 132, 0 checkpoints
# the unfused mixer read (-21.7%, and ids move)
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15174 ms (252.5 tok/s), 150 generated in 8320 ms (18.0 tok/s), drafts accepted 98 of 144, 0 checkpoints
# the generic multi-column layout at 32K (+5.5%, and ids move)
strata serve: prompt 32256 tokens = 0 reused + 32256 read in 93176 ms (346.2 tok/s), 256 generated in 10268 ms (24.9 tok/s), drafts accepted 173 of 209, 0 checkpoints
# the same arm at 4K, warm (-9.7%)
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15306 ms (250.4 tok/s), 148 generated in 7160 ms (20.7 tok/s), drafts accepted 100 of 131, 0 checkpoints
# and its first 4K run, cold (-32.2%; the same arm re-run warm with the same lever and binary is 51 ms/window faster)
strata serve: prompt 3832 tokens = 0 reused + 3832 read in 15700 ms (244.1 tok/s), 137 generated in 8822 ms (15.5 tok/s), drafts accepted 91 of 125, 0 checkpoints
# the switch announcing itself (d2-mg-*/err.txt line 7)
strata mmvq: STRATA_MMVQ_MULTI_GENERIC=1 - the multi-column projections take llama.cpp's generic multi-column layout (bitwise-equal to the single-column call NO longer holds)
```

**The generic multi-column layout is refused on correctness, not just speed.**  It moves the greedy ids at both
lengths, and **three identical runs of it produced three different streams and three different window costs**:

| run (same binary `e841fd06…`, same lever) | tokens generated | windows | ms/window | decode tok/s | verify (of which GPU-reach wait) | ids md5 |
|---|---|---|---|---|---|---|
| `d2-mg-4096` (cold cache) | 137 | 46 | 191.79 | 15.53 | 177.54 (95.94) | `10160c38…` |
| `d2-mg-4096-warm` | 148 | 51 | 140.39 | 20.67 | 126.60 (91.30) | `3b0519ca…` |
| `d2-mg-4096-warm2` | 185 | 66 | 120.68 | 23.23 | 106.95 (90.56) | `70182abf…` |
| the shipped layout, 8 runs of the session (0.5/0.7/dec0/hc*/oldiq/hist/rebase) | 147 every time | 49-54 | **125.14-134.41** (125.14-125.93 for the 0.5 arms) | 22.89-24.08 | wait 84-95 | `66bf952d…` x8 |

The GPU-reach wait is flat across the three (91-96 ms) while the REST of verify is 16.4 / 35.3 / 81.6 ms — i.e. the
extra time is real device work inside the window that the timing line does not attribute to any named part, and it
swings by 65 ms between identical runs.  The shipped layout's wait is the same and its unattributed part is ~3 ms in
every arm.  So this is not a "slower variant": it is an unstable one — a candidate race or a pathological kernel
configuration in the generic multi-column path (`native_mmvq.cu:1049-1053`: `NW = NCOLS <= 4 ? 4 : 2`, one block per
two rows, `dim3(WARP, NW)`), which is worth its own card rather than a switch flip.  It also makes its own 32K
"+5.5%" uninterpretable: with a 60% spread between runs, one run cannot price it.  The switch stays in the tree
(default off, and the `d2-rebase-4096` control proves it inert) so the next card can pick it up without
re-deriving it.

## 6. Task 3 — the precision actually used per tensor class

| what | precision on this config's decode path | where it is chosen |
|---|---|---|
| dense projections (300), the head, the shared expert's three projections | the pack's own **quantized** weights (Q6_K 24.5% of the window, IQ4_NL, Q5_K, IQ4_XS, Q4_K, Q8_0) × **Q8_1 activations** (int8 + fp16 scale and fp16 warp sum) | `native_quantize_q8_1` + `native_mmvq`, `src/core/verify.cpp:908-909` etc. — llama.cpp's MMVQ contract, pinned to `3cf03257…` |
| routed experts | the pack's per-layer **i-quant / Q2_0 / IQ4_NL** weights × the same Q8_1 rows | `native_expert_grouped`, `src/core/verify.cpp:1135-1136` |
| router logits, indexer k/q, shexp scalar gate, PLE key/value | **BF16 weights × FP32 activations**, fp32 accumulate — `bf16_gemv_fp32_mmvf*` is llama.cpp's `mmvf` (`src/kernels/cuda/native_bf16.cu:12-35`), a plain `__fmaf_rn` dot product, **not** an mma/XMX kernel | `src/core/verify.cpp:949/998/1057`, `native_bf16.cu:76-119` |
| hyper-connection mixers | **BF16 weights × FP32 activations** (the fused read's own kernels) | `src/kernels/sycl/fused_gr.cpp:840` |
| SSM α/β gates | BF16 (1-D), consumed by `gdn_ab_multi` | `src/core/verify.cpp:913` |
| SSM conv/norm/dt/a, every norm weight, the PLE norms | **F32** (1-D) — elementwise/norm kernels only, **no GEMV** | `gdn_conv_l2_multi`, `gdn_step_norm_multi`, `native_gr_rms_norm_weighted`, `native_rope_apply`, `native_ple_postops` |
| KV | int8 (`--kv int8`) | the config |

**The F32 GEMV question answers itself as a negative, with the type census behind it:** the F32 tensors on the
decode path are 1-D (`d2/D2-TYPES.txt`: `ssm_conv1d` 36×, `ssm_dt.bias` 36×, `ssm_a` 36×, `ssm_norm` 36×,
`hc_*_norm` 48×2, `attn_q_norm`/`attn_k_norm` 12× each, `indexer.{q,k}_norm` 12× each, `output_hc_norm`,
`ple_norm_*`) and every one of them is consumed by an elementwise or norm kernel.  **No F32 weight matrix is used
as a GEMV anywhere in the window**, so there is no F32→BF16 conversion to measure: the two GEMV families the window
does use already run the precision their weights ask for, and the BF16 ones already take the fp32-activation
(`mmvf`) path rather than the fp16-activation one (`bf16_gemv_split`), which is exactly what
`project_bf16`'s own comment says is required (`src/core/layer.cpp:358-367`: the fp16 activation on the router is
"the 8.100e-03 error that flips a selection").

The one genuine "faster path not taken" the precision audit exposes is different in kind: **the 40.9% dense MMVQ
is a scalar FMA integer dot product, while this device has int8 XMX** (`joint_matrix` int8 8x32x16 → int32; this
port uses XMX only in the QSA score/select path, and `probe21` measured that the device refuses tf32
`joint_matrix` outright — `src/kernels/sycl/native_qsa_score.cpp:54-69`, `qsa_select.cpp:200-208`).  Rewriting MMVQ
onto int8 XMX is a kernel-engineering project, not a dispatch flip: the weight formats are
IQ2_S/IQ3_S/IQ3_XXS/IQ4_NL/IQ4_XS/Q4_K/Q5_K/Q6_K and none of them is a plain int8 matrix, so the rewrite has to
name its own accuracy statement and its own parity test before it can be measured the way this card measures
things.  It is named here as the largest measured opportunity on the decode path and is **not** attempted.

## 7. What was ruled out, and why

1. **The six shipped canonical GEMV variants** (`s2_gemv`, `s2_gemv_fast`, `s2_gemv_q8`, `s2_gemv_quads`,
   `s_gemv`, `bf16_gemv`): **not reachable on this config** — the native pack serves all 302 dense GEMV tensors
   (`src/core/layer.cpp:142-158` fires first; the engine's `300 native projection matrices` line and the census's
   absence of every one of their symbols agree, and `native_of` would have refused the window otherwise).  Nothing
   to force, nothing to ship; their parity tests remain the only coverage they have here.
2. **The per-token dispatch** (`STRATA_DEC_BATCH=0`): **-4.0% at 4K, -3.3% at 32K**, ids identical — the
   multi-column dispatch it replaces is the right one, and the code's claim that the two are row-wise identical
   is confirmed by the ids.
3. **The older one-entry expert kernels** (`STRATA_OLD_IQ_MMVQ=1`): **-7.6% at 4K, -2.0% at 32K**, ids identical —
   the shipped `GRP_NC=4` multi-entry pair is kept.
4. **The plain and split hyper-connection reads** (`STRATA_HC_SPLIT=0`/`1`): -0.9%/-0.5% (arm band), ids identical
   — the engine's own staged choice (which it verifies bit-for-bit on the card at load) stands; there is no win
   hiding in that A/B, only the confirmation that the three variants agree numerically.
5. **The unfused mixer read** (`STRATA_WINDOW_PLAIN_GR=1`): **-21.7% AND the ids move** → a correctness change,
   refused (P5's shape).  Flagged in §8: the port's own comment names the plain `gr_read` as the kernel a numpy
   script reproduces on this pack, so a divergence between the shipped fused read and that reference is a real
   question — this card measured that they differ, not which one is right.
6. **llama.cpp's generic multi-column layout** (`STRATA_MMVQ_MULTI_GENERIC=1`): **-9.7% at 4K in its warm run,
   +5.5% at 32K, ids move, and three identical 4K runs disagree with each other (137/148/185 tokens, 120.68-191.79
   ms/window)** → refused on correctness and on instability.  The shipped "exact" layout
   (`native_mmvq.hpp`: every column bitwise equal to a single-column call) is reproducible run to run and is the one
   this device wants, which is the direct answer to the card's question for the biggest family on the decode path.
7. **The launch-site histogram as a routine instrument**: +2.4% at 4K for two dumped windows (D1: +9% for eight).
   A one-off census, as intended.
8. **The copies and the flag handshake as dispatch questions**: 3.1% and 10.7% of the window, but neither is a
   *variant* choice — there is one copy path and one handshake, and D1 already priced the copies as 39% of the
   submissions for 3.0% of the time.

## 8. Not validated, and what would settle it

* **No variance estimate.** One arm per configuration throughout (as in P1b/P3/P9/P10/D1), except the generic
  multi-column layout at 4K, which was run three times and disagreed with itself.  The band on this rig is a few
  tenths of a percent and is measured here by the rebaseline control: `d2-rebase-4096` against
  `d2-minp05-4096` is -0.6% on the same configuration, different binaries.  So -0.5%/-0.9% (the HC variants) are
  inside the band and are labelled as such; -2.0% and up are not.
* **The 4K arms stop at 147 tokens** (the prompt's own stop token) in every arm whose ids are the baseline's, so
  their tok/s is over 147 tokens rather than the protocol's 256; the arms whose ids moved ran to their own stop
  (150 for `d2-plaingr-4096`, 137/148/185 for the three `d2-mg-4096*` runs).  32K and 128K run the full 256.  D1's
  and P9's 4K arms have the same shape.
* **The generic layout's run-to-run instability is measured, not diagnosed** (three runs of one configuration: 137,
  148 and 185 generated tokens; 120.68, 140.39 and 191.79 ms/window; 16.4-81.6 ms of unattributed device time inside
  verify against the shipped layout's ~3 ms).  No race detector ran, and the two hypotheses named here — a race in
  the generic multi-column kernel, or a pathological launch configuration — are untested.
* **The plain-vs-fused mixer read divergence is measured but not attributed**: forcing `STRATA_WINDOW_PLAIN_GR=1`
  changes the ids at 4K; which of the two is correct (the port's comment says the plain read is the one a numpy
  script reproduces to 1e-6) is the next card's question, not this one's.
* **The second expert share per layer is inferred, not measured as a no-op.**  Each layer calls
  `native_expert_grouped` twice (`src/core/verify.cpp:1142` and `:1153`, the resident share then the second share),
  so 96 expert launches per window per kind; with 100% residency the second share's group count is expected to be
  zero.  The histogram counts the launches, not the groups.
* **`--kv-resident 32768` at 4K is a no-op** (it streams only above 32 768 tokens), so the 4K arms are the
  all-resident configuration and the 128K arm is the streaming one; both are the config of record's own setting.
* **The served path was not re-run after the config change.**  The engine arms carry the same args (plus the
  deviations above), and S4 left the served path working on this binary's predecessor; a serve-side confirmation of
  `--spec-min-p 0.7` is not in this card's arms.
* **Nothing is pushed**: the origin (`github.com/Niko1221/Strata`) has no `sycl-xpu` branch.
* **Not attempted (named, not hidden)**: the int8-XMX MMVQ rewrite (§6); re-deriving the QSA selection sharing
  (that is D3's card).

## 8b. The follow-up cards this session created

* **`t_8306429a`** — the generic multi-column MMVQ layout is non-deterministic (three identical runs, three token
  streams, 120.68-191.79 ms/window): find the mechanism, or bound it, and never ship it on this evidence.
* **`t_2b6b6797`** — the fused hyper-connection read and the plain `gr_read` disagree on the greedy ids at 4K even
  though the engine's load-time check says they are bit-for-bit equal: find the first divergent layer/tensor class
  and say which path the m5g numpy reference supports.  This is the only correctness question this card raised that
  it could not close.
* **`t_85e61269`** — price an int8-XMX (DPAS) MMVQ against the shipped scalar MMVQ on the decode shapes: the 40.9%
  family is a scalar FMA integer dot on a device with int8 XMX, and nobody has measured the ceiling.  A microbench
  and a bandwidth floor, not a rewrite.
* `t_87aa2963` (D3, pre-existing) is the selection-sharing + sync-pricing card and was released by this card's
  completion; D3 also inherits §4's corrected census, since its premise ("the expert chain is the dominant
  device-time consumer") is corrected here to "the dense projection MMVQ is, at 40.9%".

## 9. The state this card left the machine in

* **No engine and no server are running**: every arm ran one engine at a time and the last arm
  (`d2-mg-4096-warm2`) was allowed to exit; nothing listens on 8099 and no `strata` process is left.  The P6
  resident server was already stopped by S4 and D2 did not restart it.
* The tree is on branch `sycl-xpu` with this card's commits; `build-sycl/strata` holds `e841fd061fec873c2f24e785973a2ebe`
  (HEAD's engine + the one default-off switch).  Any future card that wants the shipped behaviour gets it with the
  switch unset; the pre-D2 binary's arms are reproducible by checking out `3da0feb`'s parent and rebuilding.
* The config of record now carries `--spec-min-p 0.7` (§2), with the pre-D2 file committed at
  `d2/config-before-d2.json`.
* Raw arm data is committed under `d2/runs/<tag>/` inside the repo (this card's rule) and the write-up's every
  number is backed by those files plus `d2/D2-EVIDENCE.txt` and `d2/D2-CENSUS.txt`.

## 10. D2b (card `t_2b6b6797`) — the fused read is not bit-equal to `gr_read`; the load-time check compares the fused family with itself

The card D2 opened as "the unfused mixer read (-21.7%, and ids move)".  Full write-up:
`d2/plain/STATUS-D2B-PLAINGR.md`, raw tables under `d2/plain/`, arm data under `d2/runs/d2b-fused-4096/`,
`d2/runs/d2b-plain-4096/`, `d2/runs/d2p-1c-fused-4096/`, `d2/runs/d2p-1c-plain-4096/`.

* **The arms re-ran D2's pair exactly** (rig + D2's reader both checked): shipped `d2b-fused-4096` 125.31 ms/window,
  23.0 tok/s, 147 ids, `ac9f16fc…`/`7a7638cd…`; forced plain `d2b-plain-4096` 159.53 ms/window, 18.1 tok/s, 150 ids,
  `ad985c23…`/`61767199…` = **+27.3% window time, -21.5% decode, the stream moves**.  The ladder instrumentation
  used below changed neither stream (control).
* **The banner's claim is true but about itself.**  `fused_gr_check` -> `fused_gr_selftest` compares the fused
  variants against `fused_gr.cu`'s own variant 1, which the `what[]` table calls "the plain read"
  (`src/kernels/sycl/fused_gr.cpp:1271`).  `strata::kernels::gr_read` is never part of it — no test in the tree
  compared the two reads before this card's `gr_parity` block.
* **Measured at the real geometry (2560/4/320, T=4, one card)**: fused staged == fused plain bitwise
  (0/2560 `mixed`), and fused vs `gr_read`'s native branch = **1707/2560 `mixed` floats differ, worst 5.884e-07**,
  3/4 `inject`, 267/320 `lo`, `xn` bit-identical, and the folded write `R_out` **0/10240 (bit-equal)**.  The
  `inject` row has no activation function and still differs, so the **reduction order alone** (fused `dot8` lane
  order vs native MMVF) suffices; the fused silu/sigmoid's `__expf` -> `sycl::native::exp` (port-documented fast
  set) is a second candidate this fixture does not separate.
* **The reference cannot separate the two reads**: host `ref/gr.py` transcription — fused 1.907e-07, `gr_read`
  native 1.960e-07, the two 2.004e-07 apart; numpy `scripts/m5g_headmix.py` on each arm's own window state —
  4.07e-07 vs 5.02e-07, both inside its 1e-6 target.
* **First-divergence layer = 0**, tensor class = the residual R after layer 0 (last-bit only: rel_rms 1.450e-07,
  max|diff| 2.980e-08, 8660/10240 values); the `.bo` (attention-half) ladder first differs at layer 2; **layer 2
  is where it goes material** (9.030e-04, a 6,200x jump = the first router near-tie flip), growing to ~1.6e-02 by
  layer 22.  The layer table had to come from a **one-card** pair: under a layer split the ladder dump site is
  only reached by the last stage (`verify.cpp:1720-1733`), so the two-card ladder covers layers >= lb (23) and the
  arms are already 1.1e-02 apart at its first entry.
* **Verdict: keep the fused read** (21.5% faster); neither path is measurably wrong, and the defect is the
  banner's scope, not the kernel.  Making the fused read bit-equal would need the MMVF reduction tree in the
  three dots plus accurate `expf` in the silu/sigmoid — i.e. the structure that buys the 21.5% — and no
  measurement here says that is a precision win.
* **Tree**: `src/core/verify.cpp` +7 lines (`STRATA_DUMP_LADDER_PERSTAGE=1` names each stage's ladder file;
  default off, unchanged), `src/kernels/gr_parity.cpp` +~120 lines (the informational fused-vs-`gr_read` block;
  adds no failure, `gr_parity` still prints `gr_read/gr_write: 0 failures` and `gr_parity OK`), and one additive
  stderr line in `src/kernels/{sycl/fused_gr.cpp,cuda/fused_gr.cu}` saying which comparison the load-time check
  actually made (the ids control arm `d2b-fused2-4096` on the final binary `cc6a41ad…` is byte-identical to the
  shipped stream).  No ctest run; no engine default moved.  Nothing pushed.
