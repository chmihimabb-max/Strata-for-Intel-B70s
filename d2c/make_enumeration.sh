#!/usr/bin/env bash
# D2c (card t_c7d8cd86): the named list of the 61 decode-phase programs -- the card's first requirement.
# Produces d2c/ENUMERATION.txt from the cold runs' own caches; the warmable/bound verdict is the table below,
# which is what decode_warmup.cu actually launches (or deliberately does not).
set +e
R=/home/michael/strata-xpu
SRC=$R/strata
BEFORE=${1:-d2b-wu-cold-4096}
AFTER=${2:-d2c-wu-cold-4096}
O=$SRC/d2c/ENUMERATION.txt
{
echo "############ D2c: every program the decode phase still built on a COLD first run, named"
echo "############ $(date -Is)"
echo
echo "METHOD.  A SYCL program here IS one strata::sycl_compat::launch(...) call site (device code split per"
echo "kernel at compile AND link time).  Each cold run's compiler cache entry carries the site's SPIR-V, whose"
echo "NUL-separated name section holds the mangled C++:"
echo "  typeinfo name for strata::sycl_compat::launch<strata::kernels::<SITE>(args)::{lambda...}>::operator()<...>"
echo "  void strata::kernels::(anonymous namespace)::<KERNEL><template args>(unsigned char*, ..., launch_shape, ...)"
echo "c++filt turns the first into the LAUNCH SITE (the helper in src/kernels/cuda/*.cu) and the second into the"
echo "exact KERNEL instantiation when the kernel is a real function template.  d2c/enumerate_c.py does this for"
echo "every 0.src written between the engine's own 'prompt ... read in N ms' line and the ask's end."
echo "D2b's classify.py/name_other.py read neither form for 35 of the 61 and reported them as '(no symbol)'."
echo
echo "===== 1. THE COLD DECODE PHASE BEFORE THIS CARD ($BEFORE: D2b's pass only) ====="
/usr/bin/python3 "$SRC/d2c/enumerate_c.py" "$BEFORE" --phase decode --list
echo
echo "===== 2. THE SAME PHASE WITH THIS CARD'S PASS ON ($AFTER) ====="
/usr/bin/python3 "$SRC/d2c/enumerate_c.py" "$AFTER" --phase decode --list
echo
echo "===== 3. WHERE THE DECODE PHASE'S PROGRAM TIME SITS, BY SITE (cold, D2b's pass only) ====="
/usr/bin/python3 "$SRC/d2c/site_cost.py" "$BEFORE" --top 25
echo
echo "===== 4. WARMABLE OR BOUND: WHAT decode_warmup() DOES WITH EACH SITE, AND WHY ====="
cat <<'TABLE'
WARMED (decode_warmup.cu launches it once, on the caller's stream, over zeroed scratch of its own; every count
it steers is a zeroed DEVICE word, so the kernels return from their first guard):

  site                          launch(es)                     why it warms
  ------------------------------------------------------------------------------------------------------------
  bf16_gemv_fp32_mmvf           3 (n_in 640/2560/4096)         block size = HOST fn of n_in (mmvf_block_size) ->
  bf16_gemv_fp32_mmvf_multi     3 (n_tok 2/4/8)                  <160> and <256>; n_tok<=4 vs >4 -> <256,4>/<256,8>
  native_expert_grouped         7 (one per (gu,down) pair)     one call per TYPE builds native_gu_multi_kernel<TG>
                                                               and native_down_multi_kernel<TD> (pack/native_experts.txt)
  moe_grouped_s2                1                              the drafter's gu_grouped_t/down_grouped_t (H/FF are
                                                               compile-time; alignment picks the fast kernels)
  moe_group_resident            1                              group_resident_kernel
  moe_hit_add                   1                              add_hits_kernel
  gdn_conv_l2_multi             2 (n_tok 1/4)                  gdn_conv_l2_multi_kernel
  gdn_conv_commit               1                              gdn_conv_commit_kernel (n_keep = a zeroed word)
  gdn_ab_multi                  2 (n_tok 1/4)                  gdn_ab_multi_kernel
  gdn_step_norm_multi           2 (n_tok 1/4)                  gdn_step_norm_multi_kernel (n_keep = nullptr = the
                                                               kernels' own read-only state path)
  mtp_select                    1                              mtp_select_kernel
  native_router_top10           1                              route
  native_router_top10_multi     2 (n_tok 1/8)                  route
  sample_tokens                 2 (temperature 0 / >0)         sampler_greedy_kernel and the sampled path
  row_top_prob                  1                              row_top_prob_kernel
  map_ids                       1                              map_ids_kernel
  native_moe_combine            1                              native_moe_combine_multi_kernel's single-column twin
  native_moe_combine_multi      2 (n_tok 1/4)                  the MoE combination kernel
  native_gr_pre_gated           2 (fused_layer 0/1)            pre_gated<true>/<false>
  native_gr_post                1                              post
  native_gr_down_silu           1                              down_silu
  native_gr_rms_norm_weighted   2 (rows 1/4)                   weighted_rms_norm<1024>
  native_qsa_gate_apply         1                              gate_kernel
  ple_history_advance           1                              the PLE history shift
  copy_from_mapped              1                              copy_from_mapped_kernel
  copy_i32_from_mapped          1                              copy_i32_from_mapped_kernel
  copy_rows_from_mapped         1                              copy_rows_from_mapped_kernel
  f32_to_bf16_bulk              1                              f32_to_bf16_bulk_kernel
  broadcast_streams             1                              broadcast_streams_kernel
  copy_indexed                  1                              copy_indexed_kernel
  doorbell_publish              1                              doorbell_publish_kernel
  fetch_blobs                   1                              fetch_blobs_kernel
  rebase_ptrs                   1                              rebase_ptrs_kernel
  wait_flag_ge                  1                              wait_flag_ge_kernel (value 0 -> returns at once)
  quantize_q8_1_rows            1                              quantize_q8_1_kernel
  quantize_q8_0_scaled          1                              quantize_q8_0_scaled_kernel

BOUND (NOT warmed, with the reason -- the card's "bound why it cannot be done for some of them"):

  qsa_decode_attn_batch -> attn_chunk_kernel<1>, attn_merge_kernel
      reads the QSA attention POOLS (QsaAttnPools: k/v pool, q4 codes, scales, a per-page table) through page
      indices a request builds.  A dummy pool is not a smaller pool: the table indexes it, so zeroed table AND
      zeroed pool are only safe if their arithmetic agrees with the real geometry, and QsaShapes/the pools are
      built after the load phase.  ~101 ms of the cold decode phase; left to the request.
  qsa_block_scores -> block_scores_multi_kernel
      same pool/geometry dependency (pooled/dead buffers + steps).  ~41 ms.
  native_qsa_indexer_append -> append<false>
      needs QsaIndexerBuffers (tail/dead/pooled/block_pos) and QsaShapes; same reason.  ~59 ms.
  shared_expert -> native_swiglu_kernel / the shared projections, shared_expert_multi
      dispatches on the pack's OWN shared-expert types (NativeSharedWeights::gate_type/up_type/down_type) and on
      SForm, i.e. on pack metadata rather than on a compile-time table.  Warming with a guessed type would build a
      program the request does not use - strictly worse than not warming (D2b's over-warm measurement).  ~72 ms;
      needs the pack's type fields plumbed into the pass.
  native_ple_postops -> gate_kernel / the PLE post-ops
      same: PleWeights (key/query/conv norms, conv1d taps) and five 10240-float buffers whose layout comes from
      the pack.  ~10 ms.
  The dense (quant type x ncols) MMVQ family is D2b's pass and is not repeated here.
TABLE
echo
echo "NOTE.  The bound set above is ~280 ms of the cold decode phase's ~1959 ms residual; the warmed set is the"
echo "other ~85% of it.  Section 1 names every one of the 61, so the split is auditable line by line."
echo
echo "===== 5. THE PER-SITE COST RANKING OF THE WARMED SET (what the pass moves) ====="
/usr/bin/python3 "$SRC/d2c/site_cost.py" "$BEFORE" --top 40 | head -45
} > "$O" 2>&1
echo "wrote $O ($(wc -l < $O) lines)"
