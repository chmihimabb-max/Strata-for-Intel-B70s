// bench/micro/w4a16_pack_check.cpp - check the W4A16 experts artifact with the ENGINE'S OWN code, no model.
//
// The card's fourth deliverable: does the pack load, and what does the engine's own reader make of it?  Every
// step below is engine code, not a re-implementation:
//
//   1. `strata::kernels::cpu::expert_layout_load` - the real `native_experts.txt` reader: the header version
//      gate, the per-layer line parse, the blob size against the format's own arithmetic, and the contiguity
//      rule the arena depends on (expert_layout.cpp:132-238).
//   2. `strata::kernels::cpu::native_fmt(2, 2, H, FF)` - THE FORMAT GATE a native pack must pass
//      (native_expert.cpp:32-68): ggml-cpu must have a `vec_dot` for the type and the geometry must be whole
//      blocks.  Its `bytes` is what the engine requires of every layer's blob.
//   3. `strata::core::FileExpertSource::open` + `blob` - the engine's own expert accessor: mmaps
//      `<pack>/experts.bin`, refuses a size that is not exactly the layout's, hands back an expert's bytes
//      (expert_source.cpp:291-...).
//   4. ggml's own dequantizer, `ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float`, over the artifact's bytes -
//      compared BIT FOR BIT with the reference the Python oracle wrote (`tools/w4a16_reference.py
//      --oracle-out`, the SOURCE dequant of the same expert).
//   5. That expert's forward on one token in fp32 against the oracle's double-precision dot (W4A16-PLAN.md §5
//      rung 2: relative 1e-5).
//
//    usage: w4a16_pack_check <pack dir> [--layer L] [--expert E] [--ref F.f32] [--xdot F.f32]
//                            [--layers-dump N] [--copy N]
//
// `--ref` is `<oracle>.f32` and `--xdot` is `<oracle>.xdot.f32`, both written by the Python oracle.
#include "strata/core/expert_source.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %-96s %s\n", what, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

std::vector<float> read_f32(const std::string& path, size_t expect, bool& ok) {
    std::vector<float> v;
    std::ifstream f(path, std::ios::binary);
    ok = false;
    if (!f) { std::printf("  cannot open %s\n", path.c_str()); return v; }
    f.seekg(0, std::ios::end);
    const size_t bytes = (size_t) f.tellg();
    f.seekg(0);
    v.resize(bytes / 4);
    f.read((char*) v.data(), (std::streamsize) (v.size() * 4));
    ok = (bool) f && v.size() == expect;
    if (!ok) std::printf("  %s holds %zu floats, expected %zu\n", path.c_str(), v.size(), expect);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <pack dir> [--layer L] [--expert E] [--ref F.f32] [--xdot F.f32]\n",
                     argv[0]);
        return 2;
    }
    const std::string pack = argv[1];
    int64_t layer = 0, expert = 0, dump = 4, copy_n = 8;
    std::string ref, xdot;
    for (int i = 2; i + 1 < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layer") layer = std::atoll(argv[++i]);
        else if (a == "--expert") expert = std::atoll(argv[++i]);
        else if (a == "--ref") ref = argv[++i];
        else if (a == "--xdot") xdot = argv[++i];
        else if (a == "--layers-dump") dump = std::atoll(argv[++i]);
        else if (a == "--copy") copy_n = std::atoll(argv[++i]);
    }
    const int64_t kLayers = 48, kExpert = cpu::NE;

    std::printf("== 1. expert_layout_load(%s, %lld, %lld)  [expert_layout.cpp]\n", pack.c_str(),
                (long long) kLayers, (long long) kExpert);
    std::string err;
    const bool loaded = cpu::expert_layout_load(pack, kLayers, kExpert, err);
    if (!loaded) {
        std::printf("  REFUSED: %s\n", err.c_str());
        return 1;
    }
    const cpu::ExpertLayout& L = cpu::expert_layout();
    std::printf("  header version %d, native=%d, n_layers=%lld, n_expert=%lld, total=%llu B (%.3f GiB), "
                "max_blob=%llu\n", L.version, (int) L.native, (long long) L.n_layers, (long long) L.n_expert,
                (unsigned long long) L.total, L.total / 1073741824.0, (unsigned long long) L.max_blob);
    for (int64_t l = 0; l < dump && l < L.n_layers; ++l) {
        std::printf("  layer %2lld  fmt gu=%d d=%d  blob %llu B  offset %llu\n", (long long) l,
                    L.fmt[(size_t) l].gu_type, L.fmt[(size_t) l].d_type, (unsigned long long) L.bytes[(size_t) l],
                    (unsigned long long) L.offset[(size_t) l]);
    }
    check(L.version == 3, "native_experts.txt is version 3 (the engine reads up to 4)");
    check(L.native, "the layout is a NATIVE pack (no canonical Q2_0 blob assumed)");
    check(L.n_layers == kLayers && L.n_expert == kExpert, "48 layers x 512 experts, as the model geometry says");

    std::printf("\n== 2. native_fmt(2, 2, %d, %d)  [native_expert.cpp: the format gate]\n", cpu::H, cpu::FF);
    cpu::NativeFmt f;
    err.clear();
    const bool fmt_ok = cpu::native_fmt(2, 2, cpu::H, cpu::FF, f, err);
    if (!fmt_ok) {
        std::printf("  REFUSED: %s\n", err.c_str());
        return 1;
    }
    std::printf("  gu_type %s (id %d), d_type %s (id %d); activations %s / %s\n",
                ggml_type_name((ggml_type) f.gu_type), f.gu_type, ggml_type_name((ggml_type) f.d_type), f.d_type,
                ggml_type_name((ggml_type) f.gu_act), ggml_type_name((ggml_type) f.d_act));
    std::printf("  gu_row %zu B, d_row %zu B, up_off %zu, down_off %zu, blob %zu B, act %zu B, h %zu B\n",
                f.gu_row, f.d_row, f.up_off, f.down_off, f.bytes, f.act_bytes, f.h_bytes);
    std::printf("  ggml_row_size(Q4_0, %d) = %zu; ggml_row_size(Q4_0, %d) = %zu; ggml_blck_size = %d\n",
                cpu::H, ggml_row_size(GGML_TYPE_Q4_0, cpu::H), cpu::FF,
                ggml_row_size(GGML_TYPE_Q4_0, cpu::FF), (int) ggml_blck_size(GGML_TYPE_Q4_0));
    check(f.bytes == 2764800, "the format's blob is 2,764,800 B, as the packer wrote");
    bool all_match = true;
    for (int64_t l = 0; l < L.n_layers; ++l) {
        if (L.bytes[(size_t) l] != f.bytes) { all_match = false; std::printf("  layer %lld blob differs\n", (long long) l); }
    }
    check(all_match, "every layer's declared blob equals the format's own size (expert_layout.cpp:184-188)");
    check(f.up_off == 640 * 1440 && f.down_off == 2 * 640 * 1440, "gate|up|down laid out back to back");
    check(f.gu_act == GGML_TYPE_Q8_0 && f.d_act == GGML_TYPE_Q8_0,
          "the activation both roles quantize is Q8_0 (what ggml_vec_dot_q4_0_q8_0 takes)");

    std::printf("\n== 3. FileExpertSource::open + blob  [expert_source.cpp: the engine's own accessor]\n");
    strata::core::FileExpertSource src;
    err.clear();
    if (!src.open(pack, kLayers, kExpert, err)) {
        std::printf("  REFUSED: %s\n", err.c_str());
        return 1;
    }
    std::printf("  blobs %lld, mapped=%d, gguf_mode=%d (experts.bin is mmap'd, not assembled from GGUF slices)\n",
                (long long) src.blobs(), (int) src.mapped(), (int) src.gguf_mode());
    check(src.blobs() == kLayers * kExpert, "24,576 blobs");
    check(src.mapped() && !src.gguf_mode(), "the pack loads from experts.bin (no --native shard needed)");

    const uint8_t* blob = src.blob(layer, expert);
    check(blob != nullptr, "the engine hands back the expert's bytes");
    if (blob == nullptr) return 1;
    std::printf("  expert (layer %lld, expert %lld) first 18 B of the gate row 0: ", (long long) layer,
                (long long) expert);
    for (int i = 0; i < 18; ++i) std::printf("%02x%s", blob[i], i == 17 ? "\n" : " ");

    std::printf("\n== 4. ggml's own dequantizer over the artifact's bytes\n");
    const ggml_type_traits* tg = ggml_get_type_traits((ggml_type) f.gu_type);
    const ggml_type_traits* td = ggml_get_type_traits((ggml_type) f.d_type);
    check(tg != nullptr && tg->to_float != nullptr && td != nullptr && td->to_float != nullptr,
          "ggml has a to_float for Q4_0");
    std::vector<float> G((size_t) cpu::FF * cpu::H), U((size_t) cpu::FF * cpu::H), D((size_t) cpu::H * cpu::FF);
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t r = 0; r < cpu::FF; ++r) {
        tg->to_float(blob + (size_t) r * f.gu_row, G.data() + (size_t) r * cpu::H, cpu::H);
        tg->to_float(blob + f.up_off + (size_t) r * f.gu_row, U.data() + (size_t) r * cpu::H, cpu::H);
    }
    for (int64_t r = 0; r < cpu::H; ++r)
        td->to_float(blob + f.down_off + (size_t) r * f.d_row, D.data() + (size_t) r * cpu::FF, cpu::FF);
    const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  decoded %zu x 3 weights (gate/up/down) in %.1f ms (one thread)\n",
                (size_t) cpu::FF * cpu::H, decode_ms);
    if (ref.empty()) {
        std::printf("  (no --ref: skipping the comparison against the Python oracle)\n");
    } else {
        const size_t per = (size_t) cpu::FF * cpu::H;
        bool ok = false;
        const std::vector<float> want = read_f32(ref, per * 3, ok);
        if (!ok) {
            g_fail += 1;
        } else {
            const float* planes[3] = {G.data(), U.data(), D.data()};
            const char* names[3] = {"gate_proj", "up_proj", "down_proj"};
            size_t total = 0, differ = 0;
            double worst = 0.0;
            for (int p = 0; p < 3; ++p) {
                size_t d = 0;
                double w = 0.0;
                for (size_t i = 0; i < per; ++i) {
                    if (std::memcmp(&planes[p][i], &want[p * per + i], 4) != 0) ++d;
                    w = (std::max)(w, std::fabs((double) planes[p][i] - (double) want[p * per + i]));
                }
                std::printf("  %-10s %zu elements, %zu differ in any bit, max abs %.3e\n", names[p], per, d, w);
                differ += d;
                total += per;
                worst = (std::max)(worst, w);
            }
            char buf[160];
            std::snprintf(buf, sizeof buf,
                          "the engine's ggml Q4_0 decode of the artifact equals the Python oracle's source "
                          "dequant: %zu weights, %zu differ, max abs %.3e", total, differ, worst);
            check(differ == 0 && worst == 0.0, buf);
        }
    }

    std::printf("\n== 5. the expert's forward, fp32, against the oracle's fp64 dot (rung 2)\n");
    if (xdot.empty()) {
        std::printf("  (no --xdot: skipping)\n");
    } else {
        bool ok = false;
        const std::vector<float> xd = read_f32(xdot, (size_t) cpu::H * 2, ok);
        if (!ok) {
            g_fail += 1;
        } else {
            const float* x = xd.data();
            const float* want = xd.data() + cpu::H;
            std::vector<float> h((size_t) cpu::FF), got((size_t) cpu::H);
            for (int64_t r = 0; r < cpu::FF; ++r) {
                float g = 0.f, u = 0.f;
                for (int64_t i = 0; i < cpu::H; ++i) { g += G[(size_t) r * cpu::H + i] * x[i]; u += U[(size_t) r * cpu::H + i] * x[i]; }
                h[(size_t) r] = (g / (1.f + std::exp(-g))) * u;
            }
            for (int64_t r = 0; r < cpu::H; ++r) {
                float s = 0.f;
                for (int64_t i = 0; i < cpu::FF; ++i) s += D[(size_t) r * cpu::FF + i] * h[(size_t) i];
                got[(size_t) r] = s;
            }
            double n = 0, d = 0;
            for (int64_t r = 0; r < cpu::H; ++r) { n += std::fabs((double) got[(size_t) r] - want[r]); d += std::fabs((double) want[r]); }
            const double rel = n / (d + 1e-30);
            std::printf("  engine decode + fp32 forward vs the oracle's fp64 dot: relative L1 %.3e "
                        "(tolerance 1e-5)\n", rel);
            check(rel < 1e-5, "rung 2: the engine-side forward reproduces the reference");
        }
    }

    std::printf("\n== 6. read cost of the artifact through the engine's accessor\n");
    {
        const auto c0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> buf((size_t) f.bytes);
        for (int64_t e = 0; e < copy_n; ++e) {
            if (!src.copy_blob(layer, (expert + e) % kExpert, buf.data())) {
                std::printf("  copy_blob(layer %lld, expert %lld) failed\n", (long long) layer, (long long) (expert + e));
                g_fail += 1;
                break;
            }
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
        std::printf("  %lld blobs copied in %.3f s = %.2f GB/s (%.2f ms/blob)\n", (long long) copy_n, s,
                    (double) copy_n * (double) f.bytes / s / 1e9, s / copy_n * 1e3);
    }

    src.close();
    std::printf("\nRESULT: %s\n", g_fail ? "PROBLEMS" : "the W4A16 experts artifact loads and decodes with the "
                                                       "engine's own code");
    return g_fail ? 1 : 0;
}
