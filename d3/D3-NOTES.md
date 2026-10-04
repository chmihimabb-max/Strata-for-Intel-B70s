D3 (t_87aa2963) settled facts — the source attribution and the handshake protocol, verified in the tree at HEAD
4ad2d12 + this card's commits.  (Working notes; the write-up is d3/STATUS-D3.md.)

=== TASK 1: WHERE THE SELECTION IS =======================================================================

The decode window is Verifier::record_window (src/core/verify.cpp:708) -> Verifier::run (src/core/verify.cpp:1408).
One window = ONE captured graph (D1: graph path default; a closure list with STRATA_SYCL_GRAPH=0), and the
selection is TWO launches per QSA layer per window:

  src/core/verify.cpp:1019-1022
    qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                     s, scores_ + (size_t) tb * max_blocks_, cs);          <- n = the window's rows (tb..te)
    qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                   sel_ + (size_t) tb * cap_, cs);
  both calls pass `n` (the number of rows in this stage's group) and NO active-block count (the last argument
  defaults to -1), i.e. they are WINDOW-LEVEL: one launch covers every row of the window, each row with its own
  step record (step_ + t * kStepCount, filled by qsa_step_fill at verify.cpp:1461: n_kv = pos+1, n_bid = n_kv/4).
  Nothing is launched per token.  (On this config spec_split is false - generate.cpp:373, :1150-1152 - so the
  group count G = (split_ && T >= 2) ? 2 : 1 is 1: verify.cpp:724, :1527.  The census counts 1 topk + 1 scores
  launch per QSA layer per window, 12 per window = the model's 12 QSA layers.)

  The capacity knobs, from the state (verify.cpp:514-517):
    cap_         = qsa_selection_width(kTopkMaxCells = 32768, s) = min(32768, 2048 + 4 - 1) = 2051 cells
    max_blocks_  = ss.qsa_states[primary].max_cells / idx_block + 2   (max_cells = --max-context: the QSA state
                   is sized by session_bytes(g, o.max_context, ...), generate.cpp:2181/2380/2469)

  scores: qsa_block_scores (include/strata/kernels/qsa_select.hpp:26; built kernel = the hand port
    src/kernels/sycl/qsa_select.cpp, generated from src/kernels/cuda/qsa_select.cu by tools/sycl/handport.py).
    Dispatch (qsa_select.cu:649-671): with no active count and nq <= MQ = 8 -> the WINDOW-SHARED kernel
    block_scores_multi_kernel (qsa_select.cu:606-646): a fixed 256-block grid strides over the blocks, and the
    pooled key of block b is read ONCE for ALL of the call's queries (`kp` outside the qi loop, cu:624), with
    "per (block, query) the same arithmetic in the same order as block_scores_kernel" (cu:600-604).  That is
    upstream commit a20f3b5 ("decode: QSA block scores read each key block once for all of a window's queries
    (bit-identical; STRATA_SCORES_MULTI=0 = old grid)"), ported: qsa_select.cpp:654/714/723-724.  The old form
    (STRATA_SCORES_MULTI=0) is the per-(query,block) grid: grid = ((reach + 7)/8, nq), block_scores_kernel
    (cu:28-51), "~24,600 mostly-idle blocks per layer at a decode window, each key re-read per query" (cu:601).

  top-k: qsa_block_topk (header :38; cu:748-786).  ONE block per query (grid = nq), and the dispatch is by
    CAPACITY, not by the depth the window reaches: `counted = active_blocks > 0` on HIP, `counted = false`
    otherwise (cu:756-763, the port keeps it: qsa_select.cpp:886-893), so reach = max_blocks_ (the CAPACITY) and
    fit = TK_T * TK_PER = 1024 * 33 = 33,792 blocks (cu:467-476; on this build TK_PER_MAX == TK_PER because the
    #if defined(__HIPCC__) branch is not taken).  reach > fit -> qsa_block_topk_ref -> block_topk_kernel
    (cu:57-153), which re-reads sc[b] from memory on every radix pass; reach <= fit -> block_topk_reg_kernel
    (cu:506-597), keys in registers, 1024 threads.
    CONSEQUENCE, and it is a real difference between the record and every published arm: the config of record's
    --max-context 262144 is max_blocks_ = 65,538 > 33,792 -> the MEMORY-keyed kernel, while D1's and D2's rigs
    passed the ARM's own CTX as --max-context (D2's write-up states it as a deviation) -> the register kernel for
    every published 4K/32K/128K decode arm.  D3's rig has a separate --maxctx for exactly this reason.

  Measured (arm CTX = register top-k), exact device microseconds from the launch-site histogram, T=4 window,
  closure path, P9 stamps ON:  see d3/D3-CENSUS-ARMCTX.txt.
    4K : scores 0.210 ms, topk 0.684 ms, attention 2.031 ms, window 132.07 ms
    32K: scores 1.466 ms, topk 0.750 ms, attention 2.000 ms, window 131.82 ms
  i.e. the DEPTH-GROWING part of the selection is the SCORES (x7.0 for x8.4 blocks), NOT the top-k (x1.1) and not
  the attention (flat, as P9 found).  The card's "scores+topk is what grows with depth" is confirmed for the
  scores and refuted for the top-k in this range.

  The scores kernel is LATENCY-bound, not read-bound: at 32K it reads 8064 blocks x 512 B = 4.13 MB per QSA layer,
  x 12 layers = 49.5 MB per window in 1.466 ms = 33.7 GB/s (the B70's own class is ~450 GB/s); at 4K the same
  arithmetic gives 28 GB/s.  Its per-(block,query) chain is 4 serially dependent head dot-products, each with a
  5-step shuffle reduction (cu:631-639).

=== TASK 3: THE PER-LAYER HANDSHAKE, AS BUILT ============================================================

Per LAYER per WINDOW on this config (G = 1), counted in the census (d3/D3-CENSUS-ARMCTX.txt) and read off the
source:

  host -> device  3 publications, each a 4-byte H2D cudaMemcpyAsync on the copy stream:
                    flagA (the group's GPU plan)  verify.cpp:1580-1587 (only when the pool published none)
                    flagB (the PCIe share is staged) :1588
                    flag  (the CPU share is in the mapped rows) :1590
                  publish_flag/raise_flag_dev: verify.cpp:282-294.
  device -> host  1 ring: doorbell_publish (verify.cpp:1072; elementwise.hpp:113) increments the MAPPED sequence
                  word; the host spins on it at verify.cpp:1539 while (*seq < want), accumulating ms_wait - the
                  "GPU-reach wait" of the decode line.  Census: 48 launches/window.
  device waits    3 wait_flag_ge spins per layer (verify.cpp:1112/1115 flagA, :1144/1145 flagB, :1156/1160 flag).
                  Census: 144 launches/window, 14.69 ms at 4K and 15.42 ms at 32K = 11.1%/11.7% of the window's
                  counted device time - the largest single non-projection family in the window.
  host's own work per layer: `per-layer host` 1.00 ms/window (plan 0.31 actq 0.22 jobs 0.23 CPU 0.00),
                  decode timing line; the whole GPU-reach wait is 86.82 ms of 128.69 ms = 67%.

E-6 / STRATA_VERIFY_DEVICE_PLAN=1 is the engine's existing coarse-sync answer and the switch that prices the
ceiling: with it the group's plan is computed ON THE DEVICE (resident_plan_kernel, verify_kernels.cu:503, which
sets `*skip = ring` in device memory, :478) and the device's wait becomes wait_flag_ge_or
(verify_kernels.cu:480-486): `if (*skip == value) return;` - i.e. the device does NOT stall for the host's flag
at all for a group it planned itself.  The host still publishes per layer; what E-6 removes is the device-side
stall (the ceiling is therefore the measured wait_flag_ge time, up to ~15.5 ms/window = 12%, plus the host's
1.0 ms per-layer work if the host cadence is batched too).  Upstream's comment calls E-6 "exact, but neutral on
RIBPC 1-2 GPUs" (verify.cpp:667-668) - never measured on this box; D3 measures it.

=== THE SAFETY REASON THE TOP-K DISPATCH CANNOT SIMPLY FOLLOW THE WINDOW'S DEPTH =========================

The choice of top-k kernel is a HOST decision taken while the window is RECORDED, and the window's graph is
captured ONCE per T and replayed for every later window of the same T:
  verify.cpp:1275-1276   bool Verifier::capture(int T, ...) { if (exec_[T] != nullptr) return true; ... }
and the Verifier is a member of the per-process Stage struct (generate.cpp:712), so exec_[T] survives across
requests.  The engine's own design rule for this is stated at verify.cpp:936-938: "`max_blocks` and `cap` are
CAPACITIES from the state, not this token's counts: a grid or a shared-memory size that follows the sequence
length is baked into a captured graph, and the kernels guard for the surplus."  The register top-k's width
(TK_PER keys per thread) IS such a baked size: passing the *current* window's reach would bake too few registers
for a later, deeper replay of the same graph (the kernel's `per = ceil(nb/TK_T)` would exceed PER and the blocks
past the bound would silently drop out of the selection).  So a safe fix is not "pass the reach": it needs an
nb-agnostic top-k (a register kernel that loops over memory for the blocks past TK_T*TK_PER) or a bound that is
an upper bound over every replay (the capacity - i.e. today's rule).  The measured value of such a change is the
register-vs-memory gap, measured in d3/D3-CENSUS-RECORD.txt and the report table.
