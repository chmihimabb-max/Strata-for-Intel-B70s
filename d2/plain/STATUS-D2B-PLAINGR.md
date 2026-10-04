# D2b (card t_2b6b6797) — forcing the plain `gr_read` changes the greedy ids: the fused read is NOT bit-equal to it

The card's question: the load-time banner says the shipped (fused, "staged") hyper-connection read is "checked
**bit for bit** against the plain read on this card", yet `STRATA_WINDOW_PLAIN_GR=1` (which runs
`strata::kernels::gr_read`, `src/kernels/cuda/gr.cu`) both costs 21.5% and moves the greedy ids at 4K.  Which of
the two claims is wrong?

**Answer, short.** The banner's claim is true and its scope is what is wrong: the "plain read" it compares against
is `fused_gr.cu`'s own variant 1 (`kHcPlain` — see the `what[]` table at `src/kernels/sycl/fused_gr.cpp:1271`),
i.e. `fused_gr_check`/`fused_gr_selftest` compare the fused family **with itself**.  `gr_read` is never part of
that check — no test in the tree compared the two reads until this card.  Measured at the real geometry (2560/4/320)
on the card: fused-staged == fused-plain bitwise (0/2560 mixed floats), and fused vs `gr_read`'s native branch
differs in **1707 of 2560** `mixed` floats (worst 5.9e-07 rel), 3/4 `inject`, 267/320 `lo`, while the write/fold
path (`R_out`) agrees bitwise.  Against the reference both are the same distance away (fused 1.9e-07, `gr_read`
2.0e-07, mutual 2.0e-07) — **the reference cannot separate them**.  Neither path is measurably wrong; the ids move
because a last-bit difference in the mixer activations reaches a router near-tie at layer 2 and is amplified there.
So: keep the fused read (it is 21.5% faster), and fix the banner's wording rather than the kernel.

## 1. The arms (both cards, one engine at a time, `ZE_AFFINITY_MASK` unset)

The D2 rig at its own defaults (`d2/d2_run_arm.sh`, `--spec-min-p 0.5`, `--prefill auto`, `--prompt-cache 0
--prompt-cache-every 0`), 4K prompt = `m6c/prompts/prompt-needle-ctx4096.txt` (3832 ids), MAXNEW 256:

| arm | tag | read | windows | ms/window | tok/s | generated | ids (D2's join) | ids (rig `md5sum`) |
|---|---|---|---|---|---|---|---|---|
| shipped | `d2b-fused-4096` | fused, staged | 51 | 125.31 | 23.0 | 147 | `7a7638cd9b4fe97f3a19d7af73e9216a` | `ac9f16fceed82e5681e42bd197835c51` |
| forced plain | `d2b-plain-4096` | `gr_read`, native branch | 52 | 159.53 | 18.1 | 150 | `617671996fdc67119a710eb6c20d8cd5` | `ad985c23cad68dcf1c462da621fc66e4` |

Both arms reproduce D2's pair exactly under one hashing method (D2's own `d2-rebase-4096` = `7a7638cd…`,
`d2-plaingr-4096` = `61767199…` — the same two streams, checked with `d2/plain/ids.py`), so this card re-ran D2's
measurement, not a new configuration: **+27.3% window time = -21.5% decode throughput, and the greedy stream
moves** (147 vs 150 tokens, different text).  Engine wall 59 s / 62 s on the rig's own line (load included), ask wall 21.7 / 23.7 s.

The ladder/window-state instrumentation used below did not change either stream (the shipped arm still hashes to
D2's baseline), which is the control for §4.

## 2. The kernel-level A/B (the differential the engine did not have)

`src/kernels/gr_parity.cpp` gained one INFORMATIONAL block (`d2/plain/grparity.txt`; it adds no failure — the
binary still prints `gr_read/gr_write: 0 failures` / `gr_parity OK`).  Same fixture, same real dims, T=4, one card
(`ZE_AFFINITY_MASK=0`), each variant forced through the switch `fused_gr_variant` reads when no on-card check has
run (`STRATA_HC_SPLIT=0` = fused plain, `=2` = fused staged):

    fused plain  vs gr_read native (real dims 2560/4/320, T=4)
        mixed                         1707/2560 floats differ, worst rel 5.884e-07
        inject                           3/4 floats differ, worst rel 3.417e-07
        lo (down+silu)                 267/320 floats differ, worst rel 1.687e-07
        xn (norm, before down)           0/10240 floats differ, worst rel 0.000e+00

    fused staged vs gr_read native — identical to the row above (staged == plain)

    fused staged vs fused plain (the load-time check's own claim, re-run at real dims)
        mixed 0/2560 | inject 0/4 | lo 0/320 | xn 0/10240          <-- the banner's claim, TRUE, and about itself

    fused staged + write vs gr_read native (with the pending write)
        mixed 1945/2560, inject 3/4, lo 264/320
        R_out (write folded)             0/10240 floats differ      <-- the write agrees; the READ differs

    against the host reference (ref/gr.py transcription, the arithmetic scripts/m5g_headmix.py repeats in numpy)
        gr_read BF16            mixed 2.031e-03 inject 2.386e-03
        gr_read FP32            mixed 1.731e-07 inject 2.552e-07
        gr_read native pinned   mixed 1.960e-07 inject 9.612e-08
        fused plain             mixed 1.907e-07 inject 1.955e-07
        fused staged            mixed 1.907e-07 inject 1.955e-07
        fused plain vs gr_read  mixed 2.004e-07 inject 2.784e-07

What this says, in order:

* **The first tensor class that differs is `lo`** (the down projection + its silu), 267 of 320 rows, worst
  1.7e-07; then `inject` (3/4) and `mixed` (1707/2560, worst 5.9e-07).  `xn` — the weighted RMSNorm output, i.e.
  the input to those projections — is **bit-identical** on this fixture, so the norm's scale order/summation is
  not where the two reads part company here.
* The divergence survives **without any activation function**: `inject = w_inject · xn` is a plain bf16 dot and it
  still differs (3 of 4 values, worst 3.4e-07).  So the **reduction order alone is sufficient**: the fused read
  accumulates `dot8` (pairs of bf16 in `fmaf`, lane `lane + 32q` ascending, `fused_gr.cu:96/217`) where the native
  branch runs `bf16_gemv_fp32_mmvf`'s pinned MMVF tree.
* A second, documented candidate rides on top of it: the fused read's silu/sigmoid call `__expf`, which the SYCL
  port maps to `sycl::native::exp` (the low-accuracy set: `include/strata/sycl_compat/intrinsics.hpp:299-300`,
  rewritten at the call sites in `src/kernels/sycl/fused_gr.cpp:131/296/514/763`), while the native post-ops use
  accurate `expf` (`native_gr_postops.cu:36/46`).  This fixture does **not** isolate that term from the ordering
  term; the `inject` row proves the ordering term is enough by itself.
* **The reference cannot separate the reads**: each is ~2e-07 from the host reference (fused 1.907e-07, native
  1.960e-07) and they are 2.004e-07 apart from each other.  The reference's own resolution is that 2e-07 floor and
  the script's stated target is 1e-6 — the same number for both arms.  Nothing here says `gr_read` is the more
  accurate kernel; what is measured is that the two produce the same answer to the reference's precision and
  different last bits.

## 3. The numpy reference on the real window state

`/home/michael/strata-xpu/scripts/m5g_headmix.py --state <arm>/window-state.bin --pack
/run/media/michael/2208B12208B0F63F/strata-iq3s/pack` (`STRATA_DUMP_WINDOW_STATE`, first window, pos 3831, T=1;
`d2/plain/m5g.txt`).  The IQ3_S pack's `index.txt` carries `output_hc_norm/down/up` with kinds (2,4,4), which is
what the script asserts, so it runs unchanged on this pack:

| arm | numpy vs engine, head mixer (mixed) | max abs | corr |
|---|---|---|---|
| shipped (fused) | rel_rms **4.0685e-07** | 2.77e-05 | 1.000000 |
| forced plain | rel_rms **5.0206e-07** | 1.00e-05 | 1.000000 |

Both arms are inside the script's own 1e-6 agreement target, i.e. the numpy reference (the card's "(2)") supports
**both** reads equally and, exactly like the host reference above, cannot separate them.  (Its per-arm residual rms
does differ — 0.65816 vs 0.653212 — because the two arms' residuals have genuinely diverged by then; that is the
amplification of §4, not the mixer read.)

## 4. Where the divergence is born, layer by layer

`STRATA_DUMP_LADDER` records the window's own residual per layer (device buffers written inside the captured
window, no arithmetic touched).  Two facts about the dump had to be established first, and one cost a code change:

* the dump site sits **after** the earlier stage returns (`verify.cpp:1720-1733`), so on a layer split the ladder
  is written **by the last stage only** and covers layers `>= lb` — the two-card arms' ladder is a single
  `…lb23` file with 27 entries, starting at R entering layer 23;
* with `STRATA_DUMP_LADDER_PERSTAGE=1` the stage's first layer goes into the file name (new, default off), which
  also documents that behaviour in the code.

So the layer table below is from a **one-card** pair (`--onecard 0`, tags `d2p-1c-fused-4096` /
`d2p-1c-plain-4096`, 36/37 s wall), where the single stage is the last stage and the ladder holds all 48 layers
(50 entries; `d2/plain/ladder-1c.txt`).  One card is a different split of the experts from the two-card arms, so
its ids are its own (151 vs 150 tokens, `7c874f62…` / `ad985c23…`); the layer numbers are what it is for.

    after layer   fused rms     plain rms    rel_rms(A,B)  max|diff|   diff/10240
    initial R     1.524321e-03  1.524321e-03  0.000e+00    0.000e+00        0     <-- identical start
    layer 0       9.124202e-03  9.124201e-03  1.450e-07    2.980e-08     8660     <-- FIRST DIVERGENCE (last bits)
    layer 1       9.168570e-03  9.168569e-03  1.505e-07    2.980e-08     9180
    layer 2       1.180654e-02  1.180729e-02  9.030e-04    7.738e-05    10239     <-- 6,200x jump: a router flip
    layer 3       1.226084e-02  1.226240e-02  2.435e-03    2.201e-04    10240
    layer 4       1.282211e-02  1.282381e-02  3.863e-03    3.548e-04    10240
    ...
    layer 9       1.535596e-02  1.536460e-02  1.171e-02    1.033e-03    10240
    layer 22      3.322076e-02  3.324787e-02  1.607e-02    2.896e-03    10240
    layer 47      6.515185e-01  6.517592e-01  4.937e-02    3.341e-01    10240

* **First-divergence layer = layer 0**, tensor class = the **residual R after layer 0** (all 10240 values already
  differ, at rel_rms 1.45e-07, max 2.98e-08 against a layer rms of 9.12e-03, i.e. max/rms 3.3e-06).  Layer 1 keeps
  exactly that magnitude (1.51e-07): the read's own last-bit difference, per layer, accumulating.
* The **attention-half ladder** (`.bo`, slot == layer) is **bit-identical at layers 0 and 1** and first differs at
  **layer 2** (rel_rms 1.58e-03, all 2560 values).  That is consistent with the residual being driven by the read's
  `inject` through the folded write (a last-bit change in `2*sigmoid(inj/hc)`) while the attention block's own
  path absorbs 1.5e-07 (quantized activations, discrete routing).
* **Layer 2 is where it becomes material**: 9.0e-04 rel_rms, a 6,200x jump in one layer = the first router top-10
  near-tie flipped by a 1.5e-07 input difference.  After that the divergence grows to ~1e-02 by layer 22 and ~5e-02
  at the last layer — the shape the `strata-xpu-measurement` skill's noise-floor note describes (1-3% rel_rms
  recurring, once MoE selection flips).
* The two-card ladder says the same thing from the other side: at its first entry (R entering layer 23) the arms
  are already 1.1e-02 rel_rms apart in that stream, i.e. the divergence is fully established long before the split
  boundary — which is why the birth of it had to be measured without a split.

## 5. Verdict and the acceptance items

1. **First-divergence layer and tensor class**: layer **0**, the residual **R after layer 0** (last-bit only:
   rel_rms 1.450e-07, max|diff| 2.980e-08); at kernel level the first divergent tensor is `lo` (down+silu,
   267/320 values, worst 1.687e-07), then `inject` and `mixed` (1707/2560, worst 5.884e-07).  The `.bo`
   (attention-half) ladder first differs at layer 2.
2. **The reference**: fused 1.907e-07 / plain 1.960e-07 from the `ref/gr.py` transcription, 2.004e-07 apart from
   each other; numpy (`m5g_headmix.py`) 4.07e-07 / 5.02e-07 on the arms' own window state.  **The reference cannot
   separate them** — the measurement that shows it is those three numbers being the same size as each other
   (its floor), against a 1e-6 target.
3. **Ids outcome**: the forced-plain arm still moves the stream — 147 vs 150 greedy tokens and different text,
   reproducing D2's pair exactly (`ac9f16fc…` vs `ad985c23…` by the rig's method), at +27.3% window time
   (-21.5% decode).
4. **Which path is right**: neither is measurably wrong, and the load-time check's claim does not cover the
   comparison it appears to make.  **Keep the fused read** (21.5% faster).  The honest statement of the defect is
   in the *claim*: `src/core/verify.cpp:852-869` says the fused read is "checked bit for bit against the plain
   read" when `fused_gr_check` only ever compares the fused variants with each other, so a `gr_read`-vs-fused
   divergence is invisible to it — which is exactly what this card measured.  Making the fused read bit-equal
   would require the MMVF reduction tree in the three dots and accurate `expf` in the silu/sigmoid, i.e. giving up
   the structure that buys the 21.5%; nothing measured says that is a precision win.

## 6. What was changed in the tree, and what was not validated

Engine changes (both instrumentation only; no default behaviour moves):

* `src/core/verify.cpp`: +7 lines — `STRATA_DUMP_LADDER_PERSTAGE=1` appends `.lb<lb_>` to the ladder path (and its
  `.bo` companion), with the comment that the dump site only ever runs in the last stage of a split.  Unset =
  byte-for-byte the old behaviour.
* `src/kernels/gr_parity.cpp`: +~120 lines — the informational fused-vs-`gr_read` block of §2.  It adds no
  failure: `gr_read/gr_write: 0 failures`, `gr_parity OK`.
* `src/kernels/sycl/fused_gr.cpp` and `src/kernels/cuda/fused_gr.cu`: +7 lines each — one extra stderr line at
  load saying what the banner's check covered (`that check compared the fused variants with each other; the plain
  gr_read (src/kernels/cuda/gr.cu) is a different kernel and is not part of it`).  Additive, printed on both cards
  (`d2/runs/d2b-fused2-4096/err.txt`), and the ids control arm below is the proof it is inert.

The control: the shipped 4K arm re-run on the final binary (`d2b-fused2-4096`, md5
`cc6a41ad8ff3cf95fd10669d7600bf2e`) gives the same stream as before the change — 147 tokens, `7a7638cd…`/`ac9f16fc…`,
96/132 drafts, 124.98 ms/window (125.31 before, 0.3%), 23.1 tok/s.

Not validated / bounds:

* the kernel-level A/B is one fixture (random bf16 weights, T=4, one shared R, one `bo_prev`/`inj_prev` pair);
  `xn` agreeing there is that fixture's result, not a proof that the two norms are identical arithmetic
  (`(R*w_norm)*rs` in the fused read vs `(scale*R)*w_norm` in `native_gr_rms_norm_weighted`, `gr.cu:157` vs
  `native_gr_norm.cu:74`);
* the fixture cannot separate the reduction order from `expf_fast`; the `inject` row shows the order alone
  suffices, and no measurement here bounds the exp term on its own;
* the one-card arms' token streams are not the two-card arms' (different split → different kernels at the
  boundary), so §4's layer numbers describe the one-card stream while §1's ids are the two-card pair;
* the ladder covers the FIRST window only (T=1 here) and, under a split, only the last stage's layers;
* no ctest run: the only test touched is `gr_parity` and it passes run directly (`d2/plain/grparity.txt`);
* nothing pushed (origin has no `sycl-xpu` branch).

## 7. Machine state left behind

Both cards free at completion: no `strata` process (every arm ran one engine at a time, all five exited 0 and were
waited on), no `serve/server.py`, nothing on 8099; only desktop apps hold the render nodes.  `build-sycl/strata`
holds `cc6a41ad8ff3cf95fd10669d7600bf2e` (HEAD `8b10dcb` + the three instrumentation changes above, the last one
confirmed inert by `d2b-fused2-4096`) and `build-sycl/gr_parity` the new differential.  The config of record was
not touched by this card.  Raw evidence: `d2/runs/d2b-fused-4096/`, `d2/runs/d2b-plain-4096/`,
`d2/runs/d2p-1c-fused-4096/`, `d2/runs/d2p-1c-plain-4096/`, `d2/runs/d2b-fused2-4096/` (engine log/err/out/timeline
plus the ladder and window-state dumps); derived tables and scripts under `d2/plain/`.
