# D2 — the decode path's dispatch, its precision, and the `--spec-min-p 0.7` win landed

Card `t_0416a0c0` (D2), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`.  One measurement session, both
B70s at a time, `ZE_AFFINITY_MASK` unset, one engine at a time, the config of record verbatim except the one lever
each arm names (`strata-sycl-iq3s.json`: IQ3_S GSQ-RCO snapshot `ed59f920…`, `--kv int8 --kv-resident 32768
--expert-cache auto --expert-profile data/expert-profile.bin --mmap-experts --prefill 512 --spec 4 --spec-min-p 0.5
--max-context 262144 --no-capture --stats`, MTP drafter `~/strata-xpu/mtp/rt`, `gpu [0,1]` → `--layer-split auto`
K=23).  Engine: `build-sycl/strata`; every arm's log carries the md5 it ran.

Rig under `d2/`, raw output under `d2/runs/` (the card asks for both inside the repo):
`d2/d2_run_arm.sh` (D1's arm runner, retargeted at `d2/runs`: one engine, the ask fed from a FIFO after
"everything loaded", md5 + device check + every engine line in the arm's own log), `d2/d2_chain0.sh` (task 0's
re-verification), `d2/d2_chain1.sh` (tasks 1-3's dispatch A/B), `d2/d2_chain2.sh` (the expert-kernel variant),
`d2/d2_chain3.sh` (the multi-column layout variant, after its build), `d2/d2_report.py` (one row per arm out of
that arm's own logs), `d2/d2_census.py` (the window's launch-site census with the attribution read off the
source), `d2/d2_types.py` + `d2/D2-TYPES.txt` (the tensor→type census), `d2/d2_sites.py` (per-site counts).

## 1. Verdict

_(filled in at the end of the session — see §2 for the `--spec-min-p` re-verification and the config diff, §4 for
the dispatch map, §6 for the variant table, §7 for the precision findings, §8 for what was ruled out.)_

## 2. The dispatch map (tensor class → kernel → file:line → predicate)

The decode path is `Verifier::record_window` (`src/core/verify.cpp:708`) — one window, two stages (K=23: CUDA0
layers 0..22, CUDA1 layers 23..47 and the head), the window captured as ONE SYCL `command_graph` and replayed once
(D1's default).  The classes below are every GEMV the window makes; the type of every one of them is in
`d2/D2-TYPES.txt` (the GGUF's own types, read by `tools/gguf_reader.py` — the same types `NativeDense` reads).

The pack's decisive fact: **the 302 non-expert dense tensors that a GEMV touches are all served NATIVELY**
(`--native <shard 1>`; the engine prints `300 native projection matrices, 2018.88 MiB of weights` and
`302 canonical tensors skipped: served natively`), so `gemv_quantized`'s FIRST branch — `w.native_data`
(`src/core/layer.cpp:142-158`) — is the one that fires for every quantized projection, and the canonical
`code_bits == 2 ? s2_gemv_q8 : (wants_q8k ? s_gemv_q8k_split : s_gemv_q8_0_split)` tail
(`src/core/layer.cpp:163-172`) is unreachable.  `native_of()` (`src/core/verify.cpp:898-900`, `:935-937`) would
have failed the window outright if any of them were not native.

| tensor class (count in the window) | pack type(s) | kernel | file:line | predicate |
|---|---|---|---|---|
| `attn_qkv.weight` (36 GDN layers) | Q6_K 22, Q5_K 10, IQ4_XS 2, Q4_K 2 | `native_mmvq` → `native_q{6,5,4}_k_mmvq` / `native_iq4_xs_mmvq` → `launch_multi` → **`launch_multi_n<F,4>`** (`native_mmvq_multi_kernel`) | call `src/core/verify.cpp:909`; dispatch `src/kernels/cuda/native_mmvq.cu:1475` (switch on `ggml_type`) → `:1371`/`:1332`/`:1293` → `:1064` → `:1046` | `w.native_data` set ⇒ `native_mmvq`; `ncols = n = 4` (the window's rows) ⇒ the multi-column launcher |
| `attn_gate.weight` (36 GDN) | Q6_K 14, Q4_K 10, IQ4_XS 8, Q5_K 4 | same | `src/core/verify.cpp:917` | same |
| `ssm_out.weight` (36 GDN) | Q6_K 29, Q5_K 4, Q4_K 3 | same | `src/core/verify.cpp:924` | same |
| `attn_k.weight`, `attn_v.weight` (12 QSA each) | Q6_K 11+10, IQ4_XS 1, Q5_K 2, Q4_K 1 | same | `src/core/verify.cpp:953`, `:954` | same |
| `attn_q.weight` (12 QSA) | Q6_K 5, IQ4_XS 3, Q5_K 3, Q4_K 1 | same | `src/core/verify.cpp:988` | same |
| `attn_output.weight` (12 QSA) | Q6_K 11, Q4_K 1 | same | `src/core/verify.cpp:1044` | same |
| `ffn_{gate,up,down}_shexp.weight` (48 layers each) | IQ4_NL/Q8_0, IQ4_XS/Q4_K/Q5_K/Q6_K | same (`small_mmvq<IQ4NLBlock,4>` for IQ4_NL) | `src/core/verify.cpp:1090` (`shared_expert_multi` → `src/kernels/cuda/shared_expert.cu:251-300`) | `native.gate_data != nullptr`; else the `s2_gemv_q8` / `s_gemv_*` tail |
| **routed experts** `ffn_{gate,up}_exps` (48 layers) | IQ3_XXS 17, IQ2_S 20, IQ3_S 10, IQ4_XS 1 (pack/native_experts.txt) | `native_expert_grouped` → `launch_gu<TG>` → `native_gu_multi_kernel` (GRP_NC=4) | call `src/core/verify.cpp:1135`; dispatch `src/kernels/cuda/iq_kernels.cu:1590`, `:1603-1608` → `:1438` (`:1444` multi) | `expert_layout().native` (the pack's per-layer GGUF types); `g_old_kernels` (env `STRATA_OLD_IQ_MMVQ`) forces `native_gu_kernel` instead |
| **routed experts** `ffn_down_exps` (48 layers) | IQ4_NL 39, Q2_0 9 | `launch_down<TD>` → `native_down_multi_kernel` | `src/kernels/cuda/iq_kernels.cu:1619-1624` → `:1447` (`:1452` multi) | as above |
| expert epilogue (96 calls) | — | `swiglu_entries_kernel`, then `quantize_q8_1_kernel` (or `_sumq` for the `Fmt<103>` types) | `src/kernels/cuda/iq_kernels.cu:1611`, `:1614-1617` | always, inside `native_expert_grouped` |
| router `ffn_gate_inp.weight` (48) | BF16 | `bf16_gemv_fp32_mmvf_multi` (llama.cpp `mmvf` = FMA dot, **not** mma/XMX) | `src/core/verify.cpp:1057`; kernel `src/kernels/cuda/native_bf16.cu:76-119` | `dec_batch && n > 1 && native_router_enabled() && NE == 512 && K == 10`; else per-token `moe_route` → `project_bf16` (`src/core/layer.cpp:94-100`) |
| router top-10 | — | `native_router_top10_multi` | `src/core/verify.cpp:1059` | same predicate (`STRATA_DEC_BATCH=0` → `router_top10`) |
| `indexer.k_proj.weight`, `indexer.q_proj.weight` (12 QSA each) | BF16 | `bf16_gemv_fp32_mmvf_multi` | `src/core/verify.cpp:949`, `:998` | `qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled() && !st.kv_q4`; else `bf16_gemv_fp32_mmvf` per token (`:951`, `:1014`) |
| hyper-connection mixers `hc_{attn,ffn}_{down,up,inject}` (48 layers each) | BF16 | `fused_gr_read_multi` → `launch_multi` (3 kernels: the norm read, the down/up pair, the inject) | call `src/core/verify.cpp:880`; launcher `src/kernels/sycl/fused_gr.cpp:840`, `:1045` | `layer_set_fused_gr(...)`; the variant is `fused_gr_variant()` = **staged** on both cards (on-card bit-for-bit check vs the plain read), forced by `STRATA_HC_SPLIT` (0 plain / 1 split / 2 staged); `STRATA_WINDOW_PLAIN_GR=1` bypasses the fused read for `gr_read` (`src/kernels/cuda/gr.cu:761`) |
| head `output.weight` (1) | Q6_K | `native_mmvq` → `launch_multi_n<Q6KTraits,4>` (the 301st dense launch) | `src/core/verify.cpp:1229` | `head_->loaded()`; else `lm_head_mix`/`lm_head` |
| `ssm_alpha/beta` (36), `ssm_{dt,a,norm,conv1d}`, all norms | BF16 (α,β) / **F32** (the rest) | elementwise / norm kernels — **no GEMV** | `gdn_ab_multi`, `gdn_conv_l2_multi`, `gdn_step_norm_multi`, `native_gr_rms_norm_weighted`, `native_rope_apply` | — |
| `token_embd.weight` | IQ4_XS | `iq_embed_rows` | `src/core/verify.cpp` embed step | PLE path |

Two things this map settles, both of which the card asked for and D1's census had wrong or unstated:

1. **The six shipped GEMV variants are not on this config's decode path.**  `s2_gemv.cu`, `s2_gemv_fast.cu`,
   `s2_gemv_q8.cu`, `s2_gemv_quads.cu`, `s_gemv.cu` and `bf16_gemv.cu` are the CANONICAL (pack-plane) family; the
   native pack bypasses them (above).  What IS on the path is `native_mmvq.cu`'s per-type MMVQ, `iq_kernels.cu`'s
   grouped expert kernels, and `native_bf16.cu`'s `mmvf` — so the alternatives worth forcing are the ones in §6,
   and the canonical family's own parity tests (`s2_gemv_parity`, `s_gemv_parity`, `s_gemv_q8k_parity`,
   `s2_gemv_q8_parity`, `bf16_gemv_parity`) are the only place they are exercised at all on this box.
2. **`launch_multi_n<…>` is the DENSE projection MMVQ, not the experts'.**  D1's `d1_families.py` maps
   `launch_multi_n` to "expert MMVQ — the native expert mat-vec, one launch per expert group", and that is the
   attribution error this card found first: `launch_multi_n` is `native_mmvq.cu:1046` (the host launcher of
   `native_mmvq_multi_kernel`), reached only from `native_mmvq()`'s `ncols > 1` path (`:1064-1078`), one launch per
   (native weight matrix, window).  The engine's own count makes it unambiguous: **301 launches, and the window has
   exactly 300 native dense matrices** (36 GDN × 3 + 12 QSA × 4 + 48 × 3 shexp = 300) **plus the head**.  The
   experts are `launch_gu`/`launch_down` (`iq_kernels.cu:1438/1447`), reached only through
   `native_expert_grouped`.  §3 is the corrected census.

## 3. The corrected census (one T=4 decode window, closure path, device microseconds)

`d2/d2_census.py` — the same instrument D1 built (`STRATA_LAUNCH_HIST=1`, closure path so every launch has a
device timestamp), the attribution read off the source instead of off the symbol's spelling.  Measured on
`d2/runs/d2-hist-4096` (3 810 counted submissions, 130.8 ms of device time, both cards summed).

_(table filled in from `d2-census.txt` — see §3 below once the arm has run.)_
