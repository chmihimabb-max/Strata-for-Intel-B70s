// bench/micro/pack_census.cpp - load a pack with the ENGINE's own loader and validate it, without the model.
//
// The engine (`strata generate`) refuses to start on a host without AVX512-VNNI/VBMI (generate.cpp:1738, the
// canonical pack's CPU expert kernels), so on this box its pack census cannot be reached through the main
// program.  Everything that census consists of is in `strata::core::WeightTable` + `strata::core::check_all`,
// which is what this runs:
//
//   1. pool_bytes()          the arena the index asks for
//   2. load()                every row of index.txt into the arena, with LoadReport
//   3. check_all()           layout.cpp's per-layer + cross-layer shape contract, all 48 layers
//   4. two value-level checks against the PACK's own bytes (WeightRef::src_off/file_id exist for this):
//        * a kind-4 BF16 tensor: the arena's bytes must equal dense.bin's bytes at src_off
//        * token_embd.weight: the arena's codes/scales planes, decoded as (code + code_bias) * scale, must
//          equal a decode of the pack's own planes - which is what makes the S4 form's convention checkable
//
//     pack_census <pack dir> [--tensor <name>] [--rows N]
#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace core = strata::core;

static bool read_span(const std::string& path, uint64_t off, uint64_t n, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg((std::streamoff) off);
    out.resize((size_t) n);
    f.read((char*) out.data(), (std::streamsize) n);
    return (bool) f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <pack dir> [--tensor <name>] [--embed-rows N]\n", argv[0]);
        return 2;
    }
    const std::string pack = argv[1];
    std::string probe_tensor = "blk.3.attn_q.weight", emb = "token_embd.weight";
    for (int i = 2; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--tensor")) probe_tensor = argv[i + 1];
        if (!std::strcmp(argv[i], "--embed")) emb = argv[i + 1];
    }

    std::string err;
    uint64_t pool = 0;
    if (!core::WeightTable::pool_bytes(pack, pool, err)) {
        std::fprintf(stderr, "pool_bytes: %s\n", err.c_str());
        return 1;
    }
    std::printf("pool_bytes: %llu B (%.3f GiB) for %s\n", (unsigned long long) pool, pool / 1073741824.0,
                pack.c_str());
    void* arena = nullptr;
    if (cudaMalloc(&arena, pool) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(%llu) failed\n", (unsigned long long) pool);
        return 1;
    }
    core::WeightTable wt;
    if (!wt.load(pack, arena, pool, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    const core::LoadReport& rep = wt.report();
    std::printf("load: %zu tensors, arena %llu B (%.3f GiB), %zu re-rounded to 16 bits saving %llu B, "
                "read %.0f ms, upload %.0f ms\n",
                rep.tensors, (unsigned long long) rep.arena_bytes, rep.arena_bytes / 1073741824.0, rep.re_rounded,
                (unsigned long long) rep.bytes_saved, rep.read_ms, rep.upload_ms);

    std::map<std::string, int> kinds;
    int quant = 0, bf16 = 0, f32 = 0, f16 = 0;
    for (const auto& kv : wt.all()) {
        const core::WeightRef& r = kv.second;
        if (r.quantized())
            ++quant, kinds["S" + std::to_string(r.code_bits) + "-g" + std::to_string(r.group_elems) + "-b" +
                           std::to_string(r.code_bias)]++;
        else if (r.kind == core::WeightKind::F32) ++f32;
        else if (r.kind == core::WeightKind::F16InF32) ++f16;
        else ++bf16;
    }
    std::printf("forms: %d quantized, %d bf16, %d f32, %d f16\n", quant, bf16, f32, f16);
    for (const auto& k : kinds) std::printf("   %-14s %4d tensors\n", k.first.c_str(), k.second);

    core::ModelGeometry g;
    err.clear();
    const bool ok = core::check_all(wt, g, err);
    std::printf("check_all (layout.cpp: %d layers, QSA/GDN split, every 2-D and 1-D shape): %s%s\n",
                (int) g.n_layers, ok ? "PASS" : "FAIL: ", err.c_str());

    int bad = ok ? 0 : 1;
    // ---- 1. a kind-4 tensor: the arena's bytes must equal the pack's own bytes at src_off
    if (const core::WeightRef* r = wt.find(probe_tensor)) {
        std::vector<uint8_t> src;
        const char* files[3] = {"dense.bin", "embd.bin", "experts.bin"};
        const std::string p = pack + "/" + files[r->file_id];
        bool same = false;
        if (read_span(p, r->src_off, r->src_bytes, src) && r->data != nullptr && r->src_bytes == r->bytes) {
            // the arena is device memory: copy it back before comparing
            std::vector<uint8_t> got((size_t) r->bytes);
            cudaMemcpy(got.data(), r->data, (size_t) r->bytes, cudaMemcpyDeviceToHost);
            same = (src == got);
        }
        std::printf("%s: %s %llu B at %s+%llu -> arena %s\n", probe_tensor.c_str(),
                    r->quantized() ? "quantized" : "unquantized", (unsigned long long) r->src_bytes,
                    files[r->file_id], (unsigned long long) r->src_off, same ? "BIT-IDENTICAL" : "DIFFERS");
        bad += !same;
    } else {
        std::printf("%s: NOT IN THE PACK\n", probe_tensor.c_str());
        bad += 1;
    }
    // ---- 2. the S-form embedding: the loader's planes must be the pack's planes, and (code+bias)*scale is
    //         the decode the arena's bytes define
    if (const core::WeightRef* r = wt.find(emb)) {
        if (!r->quantized()) {
            std::printf("%s: NOT an S-form tensor\n", emb.c_str());
            bad += 1;
        } else {
            const uint64_t per = 8u / (unsigned) r->code_bits;
            const uint64_t row_codes = (uint64_t) r->ne0 / per;
            const uint64_t n_groups = (uint64_t) r->ne0 / (uint64_t) r->group_elems;
            const uint64_t row_scale_bytes = n_groups * 4;             // f32 scales in the arena
            const size_t row = (size_t) (12345 % (size_t) r->ne1);
            const char* files[3] = {"dense.bin", "embd.bin", "experts.bin"};
            const std::string p = pack + "/" + files[r->file_id];
            std::vector<uint8_t> p_codes((size_t) row_codes), p_scales((size_t) row_scale_bytes);
            std::vector<uint8_t> a_codes((size_t) row_codes), a_scales((size_t) row_scale_bytes);
            const bool have_src = read_span(p, r->src_off + row * row_codes, row_codes, p_codes) &&
                                  read_span(p, r->src_off + r->codes_bytes + row * row_scale_bytes,
                                            row_scale_bytes, p_scales);
            if (have_src) {
                cudaMemcpy(a_codes.data(), (const uint8_t*) r->data + row * row_codes, row_codes,
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(a_scales.data(), (const uint8_t*) r->data + r->codes_bytes + row * row_scale_bytes,
                           row_scale_bytes, cudaMemcpyDeviceToHost);
                const bool same = (p_codes == a_codes) && (p_scales == a_scales);
                const float* sc = (const float*) a_scales.data();
                double first4[4];
                for (int j = 0; j < 4; ++j) {
                    const unsigned q = (a_codes[j / per] >> ((unsigned) (j % per) * (unsigned) r->code_bits)) &
                                       ((1u << (unsigned) r->code_bits) - 1u);
                    first4[j] = (double) ((int) q + r->code_bias) * (double) sc[0];
                }
                std::printf("%s: S%d g%d bias %d, row %zu of %llu: %llu codes + %llu f32 scales/row, arena planes "
                            "%s the pack's; decode (q+bias)*scale of the first 4 = %.6g %.6g %.6g %.6g\n",
                            emb.c_str(), r->code_bits, r->group_elems, r->code_bias, row,
                            (unsigned long long) r->ne1, (unsigned long long) row_codes,
                            (unsigned long long) n_groups, same ? "MATCH" : "DIFFER", first4[0], first4[1],
                            first4[2], first4[3]);
                bad += !same;
            } else {
                std::printf("%s: cannot read the pack's planes at src_off %llu\n", emb.c_str(),
                            (unsigned long long) r->src_off);
                bad += 1;
            }
        }
    } else {
        std::printf("%s: NOT IN THE PACK\n", emb.c_str());
        bad += 1;
    }
    std::printf("RESULT: %s\n", bad ? "PROBLEMS" : "the pack loads and validates");
    cudaFree(arena);
    return bad ? 1 : 0;
}
