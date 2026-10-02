// src/kernels/w4a16_expert_parity.cpp - card t_7e74bb89 (Strata XPU W2): the W4A16 int4-g128 expert path
// against the W1 CPU reference.  See W4A16-PLAN.md §2 (DW2, the Q4_0 repack), §5 (the ladder) and §6 (W2).
//
//     build-sycl/w4a16_expert_parity --pack <pack dir> [--oracle <base>] [--layer 16] [--expert 330]
//
// One expert of the real W4A16 pack (`experts.bin`: the 2,764,800-byte ggml Q4_0 blobs W1 wrote) is run
// through four implementations at the engine's own shapes (H = 2560, FF = 640, one token):
//
//   (a) THE W1 ORACLE       the checkpoint's OWN dequantized weights (`artifacts/w1-oracle/*.f32`) and the fp64
//                           forward in `<base>.xdot.f32` - the reference W4A16-PLAN.md §5 rungs 0/2 are defined
//                           against.  Rung 0 here: the blob's 4,915,200 weights must decode to the oracle's
//                           planes BIT FOR BIT, which is W4A16-PLAN.md risk R1's falsifier.
//   (b) THE CPU MISS PATH   ggml-cpu's Q4_0 vec_dot through `cpu::native_gu_rows`/`native_down_rows` - what the
//                           expert pool computes when the pack's experts are not resident.
//   (c) THE SYCL HIT PATH   `native_expert_grouped` with the new Fmt<2> Q4_0 arm: gate/up rows, SwiGLU, the
//                           intermediate's own quantization, down rows - the whole expert on the card.
//   (d) THE PER-COLUMN ARM  `native_q4_0_mmvq` (parity-green since M2) on the same blob and the same q8_1
//                           activation - the existing kernel the new grouped arm is judged against.
//
// WHY (c) RUNS TWICE.  Q4_0's decode is `d * (q - 8)` per 32 values, so the dot needs a correction term and the
// engine carries two forms of it (`iq_kernels.cu`'s block comment at vec_dot_q4_0_q8_1):
//
//     type 2   the integer-exact form, `sumi - 8 * sumq` - exactly ggml-cpu's `ggml_vec_dot_q4_0_q8_0`, i.e.
//              exactly what the CPU miss computes.  The default, and the only form a pack can name.
//     type 102 llama.cpp-CUDA's form, `sumi * d8 - 8 * sum(x)`, which takes the correction through the q8_1
//              block's stored fp16 `sum(x)`.  This is what `native_mmvq.cu`'s Q4_0 arm - and so (d) - computes.
//
// The two do NOT agree; the deviation is the q8 rounding error of the activation weighted by the affine offset,
// i.e. of the same order as the activation quantization itself.  The cross-comparison is the experiment:
// (c type 102) vs (d) shares the convention and differs only in the reduction, and (c type 2) vs (b) shares the
// convention and differs only in the implementation, so whichever large number appears is the convention and not
// a kernel defect.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cuda_runtime.h>

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

namespace {

constexpr int64_t H = 2560, FF = 640;
constexpr size_t kBlob = 2764800;
constexpr double kRung2Tol = 1e-5;   // W4A16-PLAN.md §5 rung 2, the oracle .json's own number

// ggml's blocks, restated so the test decodes exactly what the kernels read (native_expert_parity.cpp does the
// same for block_q8_1).  The static_asserts are the point: a silent struct drift would make every number below
// compare the wrong bytes.
struct Q40 { uint16_t d; uint8_t qs[16]; };
struct Q81 { uint16_t d, s; int8_t qs[32]; };
static_assert(sizeof(Q40) == 18 && offsetof(Q40, qs) == 2, "block_q4_0 layout");
static_assert(sizeof(Q81) == 36 && offsetof(Q81, qs) == 4, "block_q8_1 layout");

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - (double) b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

double rel_f(const std::vector<float>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs(b[i]); }
    return n / (d + 1e-30);
}

size_t n_diff_bits(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        if (x != y) ++n;
    }
    return n;
}

bool read_file(const std::string& p, std::vector<uint8_t>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize((size_t) n);
    f.read((char*) out.data(), n);
    return (bool) f;
}

// One row of ggml Q4_0 -> the values it decodes to, in fp64: element j is the LOW nibble of byte j&15, element
// j+16 the HIGH nibble of the same byte, and `d` is the block's fp16 scale.
void dequant_row_q4_0(const uint8_t* row, int64_t n_in, std::vector<double>& w) {
    w.resize((size_t) n_in);
    const auto* b = (const Q40*) row;
    for (int64_t k = 0; k < n_in / 32; ++k) {
        const double d = ggml_fp16_to_fp32(b[k].d);
        for (int j = 0; j < 32; ++j)
            w[(size_t) (k * 32 + j)] = (((b[k].qs[j & 15] >> (4 * (j >> 4))) & 0xF) - 8) * d;
    }
}

void dequant_q8_0_bytes(const uint8_t* blocks, int64_t n, std::vector<double>& a) {   // 34 B: fp16 d + 32 int8
    a.resize((size_t) n);
    for (int64_t k = 0; k < n / 32; ++k) {
        const double d = ggml_fp16_to_fp32(((const uint16_t*) blocks)[k * 17]);
        for (int j = 0; j < 32; ++j) a[(size_t) (k * 32 + j)] = d * (double) ((const int8_t*) blocks)[k * 34 + 2 + j];
    }
}

void dequant_q8_1_bytes(const uint8_t* blocks, int64_t n, std::vector<double>& a) {   // 36 B: fp16 d, fp16 sum
    a.resize((size_t) n);
    const auto* b = (const Q81*) blocks;
    for (int64_t k = 0; k < n / 32; ++k) {
        const double d = ggml_fp16_to_fp32(b[k].d);
        for (int j = 0; j < 32; ++j) a[(size_t) (k * 32 + j)] = d * (double) b[k].qs[j];
    }
}

double dot_fp64(const std::vector<double>& w, const std::vector<double>& a) {
    double s = 0;
    for (size_t i = 0; i < w.size(); ++i) s += w[i] * a[i];
    return s;
}

void print_usage() {
    std::printf("usage: w4a16_expert_parity --pack <dir> [--oracle <base>] [--layer N] [--expert N]\n"
                "  --pack    a W4A16 pack (experts.bin + native_experts.txt); required\n"
                "  --oracle  the W1 oracle basename, default artifacts/w1-oracle/expert-L16-E330\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string pack, oracle = "artifacts/w1-oracle/expert-L16-E330";
    int layer = 16, expert = 330;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) oracle = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (a == "--expert" && i + 1 < argc) expert = std::atoi(argv[++i]);
        else { print_usage(); return 2; }
    }
    if (pack.empty()) { print_usage(); return 2; }

    std::printf("w4a16_expert_parity: pack %s\n  oracle %s, expert layer %d / #%d\n", pack.c_str(), oracle.c_str(),
                layer, expert);

    // ------------------------------------------------------------- the pack's own table, and the engine's view
    uint64_t layer_off = 0, blob_bytes = 0;
    int gu_type = -1, d_type = -1, found = 0;
    {
        std::ifstream lines(pack + "/native_experts.txt");
        if (!lines) { std::printf("  FAIL  %s/native_experts.txt is not readable\n", pack.c_str()); return 1; }
        std::string l;
        while (std::getline(lines, l)) {
            if (l.empty() || l[0] == '#') continue;
            long long ly = 0;
            int gt = 0, dt = 0;
            unsigned long long off = 0, bb = 0;
            if (std::sscanf(l.c_str(), "%lld %d %d %llu %llu", &ly, &gt, &dt, &off, &bb) != 5) continue;
            if (ly == layer) { layer_off = off; blob_bytes = bb; gu_type = gt; d_type = dt; ++found; }
        }
    }
    std::printf("  table   layer %d: gu_type %d d_type %d offset %llu blob_bytes %llu\n", layer, gu_type, d_type,
                (unsigned long long) layer_off, (unsigned long long) blob_bytes);
    check(found == 1, "the layer appears exactly once in native_experts.txt");
    check(gu_type == 2 && d_type == 2, "the pack names ggml type 2 (Q4_0) for both roles");
    check(blob_bytes == kBlob, "the pack's blob_bytes is 2,764,800");

    check(K::native_expert_supported(2, 2, H, FF), "native_expert_supported(2, 2, 2560, 640)");
    check(K::native_expert_supported(K::kQ4_0PinnedForm, K::kQ4_0PinnedForm, H, FF),
          "native_expert_supported(102, 102, 2560, 640)");
    const K::NativeExpertLayout L = K::native_expert_layout(2, 2, H, FF);
    const K::NativeExpertLayout Lp = K::native_expert_layout(K::kQ4_0PinnedForm, K::kQ4_0PinnedForm, H, FF);
    std::printf("  layout  gu_row %zu  d_row %zu  up_off %zu  down_off %zu  bytes %zu\n", L.gu_row, L.d_row,
                L.up_off, L.down_off, L.bytes);
    check(L.gu_row == 1440 && L.d_row == 360 && L.up_off == 921600 && L.down_off == 1843200 && L.bytes == kBlob,
          "the Q4_0 layout is 1440 / 360 / 921600 / 1843200 / 2764800 B");
    check(Lp.bytes == L.bytes && Lp.gu_row == L.gu_row && Lp.d_row == L.d_row,
          "the pinned pseudo-type has the same geometry");

    // ------------------------------------------------------------- the oracle, and the blob
    std::vector<uint8_t> xdot;
    if (!read_file(oracle + ".xdot.f32", xdot) || xdot.size() != (size_t) (2 * H) * 4) {
        std::printf("  FAIL  %s.xdot.f32 unreadable / not 2 x 2560 floats\n", oracle.c_str());
        return 1;
    }
    std::vector<float> x((size_t) H), ref_dot((size_t) H);
    std::memcpy(x.data(), xdot.data(), (size_t) H * 4);
    std::memcpy(ref_dot.data(), xdot.data() + (size_t) H * 4, (size_t) H * 4);

    std::vector<uint8_t> planes;
    const size_t plane_elems = (size_t) FF * H;
    if (!read_file(oracle + ".f32", planes) || planes.size() != 3 * plane_elems * 4) {
        std::printf("  FAIL  %s.f32 unreadable / not 3 planes\n", oracle.c_str());
        return 1;
    }
    const float* or_gate = (const float*) planes.data();
    const float* or_up = or_gate + plane_elems;
    const float* or_down = or_up + plane_elems;

    std::vector<uint8_t> blob(kBlob);
    {
        std::ifstream f(pack + "/experts.bin", std::ios::binary);
        if (!f) { std::printf("  FAIL  %s/experts.bin unreadable\n", pack.c_str()); return 1; }
        f.seekg((std::streamoff) (layer_off + (uint64_t) expert * blob_bytes), std::ios::beg);
        f.read((char*) blob.data(), (std::streamoff) kBlob);
        if (!f || f.gcount() != (std::streamoff) kBlob) {
            std::printf("  FAIL  experts.bin is short at layer %d expert %d\n", layer, expert);
            return 1;
        }
    }
    std::printf("  blob    layer %d expert %d, %zu B (gate 640x1440 | up 640x1440 | down 2560x360)\n", layer,
                expert, kBlob);

    // ------------------------------------------------------------- rung 0 / risk R1: every weight, bit for bit
    {
        size_t bad = 0, elems = 0;
        double worst = 0;
        std::vector<double> w;
        const float* planes3[3] = {or_gate, or_up, or_down};
        const size_t offs3[3] = {0, L.up_off, L.down_off};
        const int64_t rows3[3] = {FF, FF, H};
        const int64_t inn3[3] = {H, H, FF};
        const int64_t rb3[3] = {(int64_t) L.gu_row, (int64_t) L.gu_row, (int64_t) L.d_row};
        for (int role = 0; role < 3; ++role) {
            for (int64_t r = 0; r < rows3[role]; ++r) {
                dequant_row_q4_0(blob.data() + offs3[role] + (size_t) r * rb3[role], inn3[role], w);
                for (int64_t i = 0; i < inn3[role]; ++i, ++elems)
                    if (w[(size_t) i] != (double) planes3[role][r * inn3[role] + i]) {
                        ++bad;
                        worst = (std::max)(worst, std::fabs(w[(size_t) i] - (double) planes3[role][r * inn3[role] + i]));
                    }
            }
        }
        std::printf("  rung 0  gate 640x2560 + up 640x2560 + down 2560x640 = %zu weights decoded from the blob vs\n"
                    "          the checkpoint's own values: %zu mismatched, max abs error %.3e (tolerance 0)\n",
                    elems, bad, worst);
        check(bad == 0, "rung 0: every packed weight equals the checkpoint's own dequantized value");
    }

    // ------------------------------------------------------------- (b) the CPU miss path
    cpu::NativeFmt fmt;
    std::string err;
    if (!cpu::native_fmt(2, 2, H, FF, fmt, err)) { std::printf("  FAIL  native_fmt: %s\n", err.c_str()); return 1; }
    std::printf("  cpu     gu %d (vec_dot %d) / down %d (vec_dot %d); act %zu B, h %zu B\n", fmt.gu_type,
                fmt.gu_act, fmt.d_type, fmt.d_act, fmt.act_bytes, fmt.h_bytes);
    check(fmt.gu_act == 8 && fmt.d_act == 8, "ggml-cpu quantizes Q4_0's activation and h as Q8_0");

    std::vector<uint8_t> cpu_act(fmt.act_bytes), cpu_h(fmt.h_bytes);
    std::vector<float> cpu_gate((size_t) FF), cpu_up((size_t) FF), cpu_ff((size_t) FF), cpu_out((size_t) H);
    {
        cpu::native_quant_act(fmt, x.data(), cpu_act.data());
        const void* act[1] = {cpu_act.data()};
        float* ff[1] = {cpu_ff.data()};
        cpu::native_gu_rows(fmt, blob.data(), act, 1, ff, 0, (int) FF);
        cpu::native_quant_h(fmt, cpu_ff.data(), cpu_h.data());
        const void* hq[1] = {cpu_h.data()};
        float* out[1] = {cpu_out.data()};
        cpu::native_down_rows(fmt, blob.data(), hq, 1, out, 0, (int) H);
        // the gate/up rows on their own (native_gu_rows fuses the SwiGLU), for the row-level comparison
        const auto* tg = ggml_get_type_traits_cpu((ggml_type) 2);
        for (int64_t r = 0; r < FF; ++r) {
            tg->vec_dot((int) H, &cpu_gate[(size_t) r], 0, blob.data() + (size_t) r * L.gu_row, 0, cpu_act.data(), 0, 1);
            tg->vec_dot((int) H, &cpu_up[(size_t) r], 0, blob.data() + L.up_off + (size_t) r * L.gu_row, 0,
                        cpu_act.data(), 0, 1);
        }
    }

    // The fp64 reference the CPU path is judged against: the row dots over the CPU's OWN quantized activation
    // (so this separates "the implementation" from "the activation rounding"), and the same over the raw fp32 x.
    std::vector<double> a_hat, h_hat;
    dequant_q8_0_bytes(cpu_act.data(), H, a_hat);
    dequant_q8_0_bytes(cpu_h.data(), FF, h_hat);
    std::vector<double> ref_gate((size_t) FF), ref_up((size_t) FF), ref_down((size_t) H);
    for (int64_t r = 0; r < FF; ++r) {
        std::vector<double> w;
        dequant_row_q4_0(blob.data() + (size_t) r * L.gu_row, H, w);
        ref_gate[(size_t) r] = dot_fp64(w, a_hat);
        dequant_row_q4_0(blob.data() + L.up_off + (size_t) r * L.gu_row, H, w);
        ref_up[(size_t) r] = dot_fp64(w, a_hat);
    }
    for (int64_t r = 0; r < H; ++r) {
        std::vector<double> w;
        dequant_row_q4_0(blob.data() + L.down_off + (size_t) r * L.d_row, FF, w);
        ref_down[(size_t) r] = dot_fp64(w, h_hat);
    }
    const double cpu_gate_vs_ref = rel_f(cpu_gate, ref_gate);
    const double cpu_up_vs_ref = rel_f(cpu_up, ref_up);
    const double cpu_down_vs_ref = rel_f(cpu_out, ref_down);
    std::printf("  cpu     gate rows %lld / down rows %lld vs the fp64 dot on the CPU's own q8_0 activation:\n"
                "          gate rel %.3e, up rel %.3e, down rel %.3e  (the activation rounding is excluded)\n",
                (long long) FF, (long long) H, cpu_gate_vs_ref, cpu_up_vs_ref, cpu_down_vs_ref);
    std::printf("  cpu     down rows vs the oracle's fp64 dot over the RAW fp32 x: rel %.3e\n", rel(cpu_out, ref_dot));
    check(cpu_gate_vs_ref < kRung2Tol && cpu_down_vs_ref < kRung2Tol,
          "the CPU Q4_0 path is exact to < 1e-5 on its own activation");

    // ------------------------------------------------------------- the device arms
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { std::printf("  FAIL  no CUDA/SYCL stream\n"); return 1; }

    void* d_blob = nullptr;
    void* d_x = nullptr;
    void* d_q8 = nullptr;
    void* d_q8b = nullptr;
    void* d_scratch = nullptr;
    void* d_hq = nullptr;
    float* d_out = nullptr;
    unsigned long long* d_grp = nullptr;
    int32_t *d_start = nullptr, *d_ngroups = nullptr, *d_dst = nullptr, *d_tok = nullptr;
    const size_t q8_bytes = (size_t) H / 32 * 36;
    cudaMalloc(&d_blob, kBlob);
    cudaMalloc(&d_x, (size_t) H * 4);
    cudaMalloc(&d_q8, q8_bytes);
    cudaMalloc(&d_q8b, q8_bytes);
    cudaMalloc(&d_scratch, K::native_expert_scratch_bytes(1, FF));
    cudaMalloc(&d_hq, (size_t) FF / 32 * 36);
    cudaMalloc((void**) &d_out, (size_t) H * 4);
    cudaMalloc((void**) &d_grp, sizeof(unsigned long long));
    cudaMalloc((void**) &d_start, 2 * sizeof(int32_t));
    cudaMalloc((void**) &d_ngroups, sizeof(int32_t));
    cudaMalloc((void**) &d_dst, sizeof(int32_t));
    cudaMalloc((void**) &d_tok, sizeof(int32_t));
    if (!d_blob || !d_x || !d_q8 || !d_q8b || !d_scratch || !d_hq || !d_out || !d_grp || !d_start || !d_ngroups ||
        !d_dst || !d_tok) {
        std::printf("  FAIL  device allocation\n");
        return 1;
    }
    cudaMemcpy(d_blob, blob.data(), kBlob, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), (size_t) H * 4, cudaMemcpyHostToDevice);
    {
        void* p = d_blob;
        const int32_t start[2] = {0, 1}, one = 1, zero = 0;
        cudaMemcpy(d_grp, &p, sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_start, start, sizeof(start), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ngroups, &one, sizeof(one), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dst, &zero, sizeof(zero), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tok, &zero, sizeof(zero), cudaMemcpyHostToDevice);
    }
    K::quantize_q8_1_rows((const float*) d_x, 1, H, d_q8, s);
    K::native_quantize_q8_1((const float*) d_x, d_q8b, (int) H, 1, s);
    cudaStreamSynchronize(s);
    {
        std::vector<uint8_t> a(q8_bytes), b(q8_bytes);
        cudaMemcpy(a.data(), d_q8, q8_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_q8b, q8_bytes, cudaMemcpyDeviceToHost);
        const bool same = std::memcmp(a.data(), b.data(), q8_bytes) == 0;
        std::printf("  q8_1    the iq path's quantizer and native_mmvq's: %s (%zu B)\n",
                    same ? "byte-identical" : "DIFFER", q8_bytes);
        check(same, "both q8_1 quantizers produce the same activation blocks");
    }

    const size_t fa = ((size_t) FF * 4 + 255) & ~(size_t) 255;   // native_expert_grouped's per-entry stride
    const int types[2] = {2, K::kQ4_0PinnedForm};
    const K::NativeExpertLayout* lays[2] = {&L, &Lp};
    const char* names[2] = {"type 2  ", "type 102"};
    std::vector<float> g_gate[2], g_up[2], g_out[2];

    for (int arm = 0; arm < 2; ++arm) {
        K::native_expert_grouped(*lays[arm], d_grp, d_start, d_ngroups, d_dst, d_tok, 1, 1, d_q8, d_scratch, d_out, s);
        cudaStreamSynchronize(s);
        g_gate[arm].resize((size_t) FF);
        g_up[arm].resize((size_t) FF);
        g_out[arm].resize((size_t) H);
        cudaMemcpy(g_gate[arm].data(), (const uint8_t*) d_scratch, (size_t) FF * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(g_up[arm].data(), (const uint8_t*) d_scratch + fa, (size_t) FF * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(g_out[arm].data(), d_out, (size_t) H * 4, cudaMemcpyDeviceToHost);

        // (d) the per-column Q4_0 kernel on the SAME blob and the SAME activation, projection by projection.
        // The down arm uses the grouped kernel's OWN h, re-quantized exactly as the kernel quantizes it.
        std::vector<float> y_g((size_t) FF), y_u((size_t) FF), y_d((size_t) H);
        K::native_q4_0_mmvq(d_blob, d_q8b, y_g.data(), (int) H, (int) FF, 1, s);
        K::native_q4_0_mmvq((const uint8_t*) d_blob + L.up_off, d_q8b, y_u.data(), (int) H, (int) FF, 1, s);
        K::native_quantize_q8_1((const float*) ((const uint8_t*) d_scratch + 2 * fa), d_hq, (int) FF, 1, s);
        K::native_q4_0_mmvq((const uint8_t*) d_blob + L.down_off, d_hq, y_d.data(), (int) FF, (int) H, 1, s);
        cudaStreamSynchronize(s);

        std::printf("  %s grouped vs native_q4_0_mmvq (one column, same activation)\n", names[arm]);
        std::printf("          gate rel %.3e (%zu/640 rows differ in a bit),  up rel %.3e,  down rel %.3e\n",
                    rel(g_gate[arm], y_g), n_diff_bits(g_gate[arm], y_g), rel(g_up[arm], y_u),
                    rel(g_out[arm], y_d));
        std::printf("          gate/up vs the fp64 dot on the q8_1 activation: gate rel %.3e, up rel %.3e\n",
                    rel_f(g_gate[arm], ref_gate), rel_f(g_up[arm], ref_up));
        std::printf("          the whole expert (down rows) vs the oracle's fp64 dot over the raw fp32 x: "
                    "rel %.3e\n", rel(g_out[arm], ref_dot));
    }

    // ------------------------------------------------------------- the comparison that decides it
    const double gpu_vs_cpu[2] = {rel(g_out[0], cpu_out), rel(g_out[1], cpu_out)};
    std::printf("\n  == %lld down rows of one expert, one token, four implementations (rel L1) ==\n", (long long) H);
    std::printf("  cpu miss          vs the oracle's fp64 dot over the raw fp32 x : rel %.3e\n", rel(cpu_out, ref_dot));
    std::printf("  gpu hit  type 2   vs the oracle's fp64 dot over the raw fp32 x : rel %.3e\n", rel(g_out[0], ref_dot));
    std::printf("  gpu hit  type 102 vs the oracle's fp64 dot over the raw fp32 x : rel %.3e\n", rel(g_out[1], ref_dot));
    std::printf("  gpu hit  type 2   vs the cpu miss                             : rel %.3e (%zu of %lld rows differ "
                "in a bit)\n", gpu_vs_cpu[0], n_diff_bits(g_out[0], cpu_out), (long long) H);
    std::printf("  gpu hit  type 102 vs the cpu miss                             : rel %.3e\n", gpu_vs_cpu[1]);
    std::printf("  gpu hit  type 2   vs gpu hit type 102                         : rel %.3e\n",
                rel(g_out[0], g_out[1]));
    std::printf("\n");
    check(gpu_vs_cpu[0] < kRung2Tol, "the type-2 GPU hit reproduces the CPU miss (< 1e-5): the hit/miss contract");
    check(gpu_vs_cpu[1] > 10 * gpu_vs_cpu[0],
          "cross-check: the pinned type-102 convention is measurably NOT the CPU miss");

    cudaFree(d_blob); cudaFree(d_x); cudaFree(d_q8); cudaFree(d_q8b); cudaFree(d_scratch); cudaFree(d_hq);
    cudaFree(d_out); cudaFree(d_grp); cudaFree(d_start); cudaFree(d_ngroups); cudaFree(d_dst); cudaFree(d_tok);
    cudaStreamDestroy(s);

    std::printf("\nw4a16_expert_parity: %s (%d failed check%s)\n", g_fail ? "FAIL" : "PASS", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
