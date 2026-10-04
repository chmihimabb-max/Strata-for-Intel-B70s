#ifndef STRATA_KERNELS_DECODE_WARMUP_HPP
#define STRATA_KERNELS_DECODE_WARMUP_HPP

namespace strata::kernels {

/// D2c (card t_c7d8cd86): build the REST of the decode path's first-launch SYCL programs in the LOAD phase.
///
/// D2b's `native_mmvq_warmup` moved the dense (quant type x ncols) family out of the first decode windows; the
/// same phenomenon was still there in 55 other decode-path kernel sites (one program per site, one first launch
/// each) plus the 6 routed-expert `native_gu_multi_kernel<T>`/`native_down_multi_kernel<T>` programs.  This pass
/// calls every one of those sites once, on the caller's stream, over scratch buffers it allocates itself and
/// **nothing reads** - so it cannot change a number (the ids guard is the evidence, not the argument).
///
/// It is called from the serve path immediately AFTER `native_mmvq_warmup`, i.e. still before the
/// "everything loaded" line, before the ask can arrive and before any window or graph capture has run.
/// `STRATA_KERNEL_WARMUP=0` is its A/B control arm; `STRATA_MMVQ_WARMUP=0` stays D2b's, so the two passes can be
/// measured separately.
void decode_warmup(void* stream);

}  // namespace strata::kernels

#endif  // STRATA_KERNELS_DECODE_WARMUP_HPP
