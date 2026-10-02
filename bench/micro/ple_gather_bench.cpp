// bench/micro/ple_gather_bench.cpp - the PLE row gather's wall clock, for whichever table is passed in.
//
// W4A16 W1b asks for one number the plan cannot supply: the PLE table's row gather cost for the two candidate
// forms (FP8 E4M3, 160 B/row, 51.20 GB; IQ4_NL, 90 B/row, 28.80 GB) through the ENGINE's own reader
// (`strata::kernels::PleTable`, include/strata/kernels/ngram.hpp), in both of its I/O modes:
//
//   Direct  the plan's default: 4 KiB unbuffered reads from the SSD, one 4 KiB page per row, issued by an
//           I/O worker (`--ple-io direct`)
//   Mmap    the earlier path, kept as the A/B arm (`--ple-io mmap`)
//
// `gather()` is the decode hot path: 16 rows -> 2560 floats.  `gather_batch()` is the prompt path: all of a
// batch's rows as one request (page dedupe + sort across the batch).
//
//     ple_gather_bench <ple.gguf> <n_tokens> [--batch 256] [--repeat 3] [--cache 1048576] [--inflight 64]
//
// The row indices are a deterministic LCG over [0, rows): with 320,001,536 rows and a 1,048,576-row cache the
// hit rate is ~0.3%, so this measures the cold SSD path, which is the one a real token takes.  (A row index
// that repeats is still a legitimate draw - the real hash produces repeats too.)
#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace k = strata::kernels;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <ple.gguf> <n_tokens> [--batch N] [--repeat N] [--cache N] "
                             "[--inflight N] [--ram]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const int n_tok = std::atoi(argv[2]);
    int batch = 256, repeat = 3;
    uint64_t cache = 1u << 20;
    uint32_t inflight = 64;
    bool ram = false;
    for (int i = 3; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--batch")) batch = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--cache")) cache = std::strtoull(argv[i + 1], nullptr, 0);
        else if (!std::strcmp(argv[i], "--inflight")) inflight = (uint32_t) std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--ram")) ram = true;
    }

    std::vector<uint32_t> rows((size_t) n_tok * k::PLE_N_HEADS);
    uint64_t x = 88172645463325252ull;
    for (size_t i = 0; i < rows.size(); ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;                       // xorshift64: deterministic, in range
        rows[i] = (uint32_t) (x % k::PLE_TABLE_ROWS);
    }
    std::vector<float> out((size_t) n_tok * k::NG_N_EMBD);
    std::vector<float> bat((size_t) batch * k::NG_N_EMBD);
    std::vector<uint32_t> brows(rows.begin(), rows.begin() + (size_t) batch * k::PLE_N_HEADS);

    for (const k::PleIo mode : {k::PleIo::Direct, k::PleIo::Mmap}) {
        k::PleTable t;
        std::string err;
        k::PleIoOptions io;
        io.mode = mode;
        io.max_inflight = inflight;
        io.cache_rows = cache;
        io.lock = ram;
        const auto t0 = clk::now();
        if (!t.open(path, err, io)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        const double open_s = std::chrono::duration<double>(clk::now() - t0).count();
        std::vector<double> us;
        double worst = 0;
        for (int r = 0; r < repeat; ++r) {
            for (int i = 0; i < n_tok; ++i) {
                const auto a = clk::now();
                t.gather(rows.data() + (size_t) i * k::PLE_N_HEADS, out.data() + (size_t) i * k::NG_N_EMBD);
                const double d = std::chrono::duration<double, std::micro>(clk::now() - a).count();
                if (r) us.push_back(d);        // the first pass warms the reader's own page cache
                worst = std::max(worst, d);
            }
        }
        std::sort(us.begin(), us.end());
        double sum = 0;
        for (double v : us) sum += v;
        const int nb = std::max(1, n_tok / batch);
        const auto ab = clk::now();
        bool ok = true;
        for (int i = 0; i < nb; ++i) ok = ok && t.gather_batch(brows.data(), (size_t) batch, bat.data(), err);
        const double btot = std::chrono::duration<double, std::micro>(clk::now() - ab).count();
        std::printf("%-6s %-8s rows %llu  open %.2f s  gather16 rows  n=%zu  mean %8.1f us  median %8.1f  "
                    "p99 %8.1f  worst %8.1f  |  gather_batch %d x %d tokens: %.1f us/token%s%s\n",
                    mode == k::PleIo::Direct ? "Direct" : "Mmap", t.format(), (unsigned long long) t.rows(),
                    open_s, us.size(), sum / us.size(), us[us.size() / 2], us[us.size() * 99 / 100], worst,
                    ok ? nb : -nb, batch, btot / (nb * batch),
                    t.mode() == k::PleIo::Direct ? "  (direct)" : "  (mmap)",
                    t.locked() ? " locked" : "");
        if (t.mode() == k::PleIo::Direct) std::printf("        %s", t.io_report().c_str());
        t.close();
    }
    return 0;
}
