#!/usr/bin/env python3
"""D2y (card t_ba006576) step 1: the launch geometry of every layout the shipped code can take.

Host arithmetic from source constants only -- no measurement here.  Every number is computed from
`src/kernels/cuda/native_mmvq.cu` at HEAD:

  * the traits' (DIV, T, BPI, block_bytes)                        native_mmvq.cu:793-994
  * the exact layout's shape rule    launch_multi_n<Traits, 4>   native_mmvq.cu:1059-1064
      ROWS = WARPS (4) and blocks = ceil(n_out / WARPS)  when n_in / DIV < BPI
      ROWS = 1         and blocks = n_out                otherwise
  * the generic layout (multi_exact = false)                      native_mmvq.cu:1052-1056
      NW = 4 for ncols <= 4 else 2, ROWS = 2, blocks = ceil(n_out / 2)
  * D2y's knob (`native_mmvq_set_exact_rows`, default 0 = the rule) forces ROWS in the exact layout

and from d2/D2-TYPES.txt / the bench's own case table for the 25 geometries and their launch counts.

What it is for: the card's step 1 asks what limits each launch shape.  The columns that answer it are
  thr/wB       threads per weight byte the call moves -- "not enough parallelism" would show up here
  wB/act.thr   weight bytes each *working* thread reads in the whole call -- "too little work per
               thread/block" shows up here
  iters        iterations of the kbx loop each working thread runs
  act/128      how many of the 128 threads in the block do any work at all (the rest only store
               partials and hit the barrier)
  part B       the __shared__ partial array the block allocates (3 x NCOLS x ROWS x 32 floats): it is
               what limits blocks-in-flight, and it also scales with the reduction work per block

    /usr/bin/python3 d2y/attribute.py > d2y/D2Y-ATTRIB.txt
"""
import sys

WARP = 32
WARPS = 4
NW_EXACT = 4          # the exact layout is always WARPS warps
NCOLS = 4             # the census's window is T = 4

# type -> (DIV, T, BPI, block_bytes, block_elems, x bytes one apply reads, name)
TRAITS = {
    14: (256, 32, 4, 210, 256, 12, "Q6_K"),
    13: (256, 16, 8, 176, 256, 12, "Q5_K"),
    12: (256, 16, 8, 144, 256, 12, "Q4_K"),
    23: (256, 8, 16, 136, 256, 34, "IQ4_XS"),
    20: (32, 2, 64, 18, 32, 10, "IQ4_NL"),
    8: (32, 4, 32, 34, 32, 10, "Q8_0"),
}

# the 25 geometries and their per-window launch counts (bench/micro/mmvq_xmx_price.cpp CASES)
CASES = [
    ("Q6_K 2560x10240", 14, 2560, 10240, 22),
    ("Q6_K 2560x12288", 14, 2560, 12288, 5),
    ("Q6_K 2560x6144", 14, 2560, 6144, 14),
    ("Q6_K 2560x640", 14, 2560, 640, 26),
    ("Q6_K 2560x512", 14, 2560, 512, 21),
    ("Q6_K 6144x2560", 14, 6144, 2560, 40),
    ("Q6_K 2560x248320-head", 14, 2560, 248320, 1),
    ("Q5_K 2560x10240", 13, 2560, 10240, 10),
    ("Q5_K 2560x12288", 13, 2560, 12288, 3),
    ("Q5_K 2560x6144", 13, 2560, 6144, 4),
    ("Q5_K 2560x640", 13, 2560, 640, 12),
    ("Q5_K 2560x512", 13, 2560, 512, 2),
    ("Q5_K 6144x2560", 13, 6144, 2560, 4),
    ("Q4_K 2560x10240", 12, 2560, 10240, 2),
    ("Q4_K 2560x12288", 12, 2560, 12288, 1),
    ("Q4_K 2560x6144", 12, 2560, 6144, 10),
    ("Q4_K 2560x640", 12, 2560, 640, 30),
    ("Q4_K 6144x2560", 12, 6144, 2560, 4),
    ("IQ4_XS 2560x10240", 23, 2560, 10240, 2),
    ("IQ4_XS 2560x12288", 23, 2560, 12288, 3),
    ("IQ4_XS 2560x6144", 23, 2560, 6144, 8),
    ("IQ4_XS 2560x640", 23, 2560, 640, 28),
    ("IQ4_XS 2560x512", 23, 2560, 512, 1),
    ("IQ4_NL 640x2560", 20, 640, 2560, 47),
    ("Q8_0 640x2560", 8, 640, 2560, 1),
]

BW_GBS = 571.6          # D2x's measured incompressible read rate, for the floor column


def launch(t, n_in, n_out, rows):
    div, tt, bpi, blk_bytes, blk_elems, x_apply, _ = TRAITS[t]
    bpr = n_in // div
    bpi_eff = bpi * NW_EXACT // WARPS
    blocks = (n_out + rows - 1) // rows
    threads = WARPS * WARP
    # a thread starts at kbx = tid / T and steps by bpi_eff; only kbx < bpr does any work
    total_iters = 0
    active = 0
    for tid in range(threads):
        kbx = tid // tt
        n = 0
        while kbx < bpr:
            n += 1
            kbx += bpi_eff
        total_iters += n
        if n:
            active += 1
    wrow = bpr * blk_bytes                       # weight bytes in one row
    wblock = wrow * rows                         # weight bytes one work-group reads
    return {
        "rows": rows, "blocks": blocks, "threads": threads, "active": active, "iters": total_iters,
        "iters_per_active": (total_iters / active) if active else 0.0,
        "wrow": wrow, "wblock": wblock,
        "wb_per_active": (wblock / active) if active else 0.0,
        "thr_per_wb": (threads / wblock) if wblock else 0.0,
        "active_frac": active / threads,
        "part_b": (WARPS - 1) * NCOLS * rows * WARP * 4,
        "out_per_block": NCOLS * rows,
        "applies_blk": total_iters * NCOLS,
        "applies_call": total_iters * NCOLS * blocks,
        "x_apply": x_apply,
        "applies_per_b": (total_iters * NCOLS / wblock) if wblock else 0.0,
        "x_bytes": total_iters * NCOLS * x_apply,
        "x_per_wb": (total_iters * NCOLS * x_apply / wblock) if wblock else 0.0,
        "bpr": bpr, "bpi_eff": bpi_eff, "el_per_thread_iter": blk_elems // tt,
    }


def rule_rows(t, n_in):
    div, tt, bpi, blk_bytes, blk_elems, x_apply, _ = TRAITS[t]
    return WARPS if (n_in // div) < bpi else 1


def main():
    print("D2y -- the launch geometry of every layout the shipped dense MMVQ can take")
    print("host arithmetic from src/kernels/cuda/native_mmvq.cu at HEAD + d2/D2-TYPES.txt; no measurement here")
    print("ncols = %d, WARPS = %d, WARP = %d; the exact layout is NW = WARPS warps in every case" % (NCOLS, WARPS, WARP))
    print()
    print("the traits the shapes are made of")
    print("%-8s %5s %4s %5s %7s %9s %6s %9s %11s" %
          ("type", "DIV", "T", "BPI", "blk B", "el/blk", "el/thr", "kqs span", "x B/apply"))
    for t in sorted(TRAITS, key=lambda x: TRAITS[x][6]):
        div, tt, bpi, blk_bytes, blk_elems, x_apply, name = TRAITS[t]
        print("%-8s %5d %4d %5d %7d %9d %6d %9d %11d" %
              (name, div, tt, bpi, blk_bytes, blk_elems, blk_elems // tt, bpi * div, x_apply))
    print()
    print("   el/thr = elements one thread covers per kbx iteration; kqs span = elements one work-group's")
    print("   128 threads reach in one iteration (one paged pass over the row's blocks)")
    print()

    print("=" * 150)
    print("the shipped rule, per geometry (ROWS = WARPS when n_in/DIV < BPI, else 1)")
    print("=" * 150)
    hdr = ("%-22s %4s %5s %6s %6s %7s %7s %9s %11s %8s %9s %9s %9s" %
           ("case", "cnt", "DIV", "ROWS", "blocks", "act/128", "it/blk", "wB/block", "wB/act.thr", "thr/wB",
            "part B", "x/wB", "app/blk"))
    print(hdr)
    tot = {"blocks": 0, "bytes": 0, "applies": 0, "floors": 0.0, "part_bytes": 0.0}
    for name, t, n_in, n_out, cnt in CASES:
        rows = rule_rows(t, n_in)
        g = launch(t, n_in, n_out, rows)
        div = TRAITS[t][0]
        print("%-22s %4d %5d %6d %6d %5d/128 %7d %9d %11.1f %8.3f %9d %9.2f %9d" %
              (name, cnt, div, rows, g["blocks"], g["active"], g["iters"], g["wblock"], g["wb_per_active"],
               g["thr_per_wb"], g["part_b"], g["x_per_wb"], g["applies_blk"]))
        wbytes = g["wrow"] * n_out
        tot["blocks"] += cnt * g["blocks"]
        tot["bytes"] += cnt * wbytes
        tot["applies"] += cnt * g["applies_call"]
        tot["floors"] += cnt * wbytes / (BW_GBS * 1e9) * 1e6
        tot["part_bytes"] += cnt * g["part_b"] * g["blocks"]
    print("-" * 150)
    print("the window: 301 calls, %d work-groups, %.0f B of weights, %.1fM applies (one apply = one thread's dot "
          "over one block slice, x %d columns), %.0f B of __shared__ partial arrays (%.2fx the weights)" %
          (tot["blocks"], tot["bytes"], tot["applies"] / 1e6, NCOLS, tot["part_bytes"],
           tot["part_bytes"] / tot["bytes"]))
    print()
    print("   the seven geometries the rule sends down the ROWS = WARPS path (n_in/DIV < BPI):")
    for name, t, n_in, n_out, cnt in CASES:
        if rule_rows(t, n_in) == WARPS:
            g = launch(t, n_in, n_out, WARPS)
            print("     %-22s %2d launches, %5d blocks, only %3d/128 threads work (%4.1f%%), %.1f iteration(s) each, "
                  "%5.0f B/block" %
                  (name, cnt, g["blocks"], g["active"], 100 * g["active_frac"], g["iters_per_active"], g["wblock"]))
    print()
    r1 = [(name, t, n_in, n_out, cnt) for name, t, n_in, n_out, cnt in CASES if rule_rows(t, n_in) == 1]
    it = sum(launch(t, n_in, n_out, 1)["iters_per_active"] * cnt for _, t, n_in, n_out, cnt in r1)
    n1 = sum(cnt for _, _, _, _, cnt in r1)
    print("   the %d geometries on the ROWS = 1 path (%d of the 301 launches): one work-group per row, %d threads, "
          "%.1f kbx iterations per thread on average, %.0f B of weights per work-group" %
          (len(r1), n1, WARPS * WARP, it / n1,
           sum(launch(t, n_in, n_out, 1)["wblock"] * cnt for _, t, n_in, n_out, cnt in r1) / n1))
    print()

    print("=" * 150)
    print("what D2y's knob changes: the exact layout's ROWS forced (native_mmvq_set_exact_rows)")
    print("=" * 150)
    print("%-22s %8s %8s %8s %8s %8s   %s" % ("case", "r2", "r4", "r8", "r16", "shipped", "note"))
    for name, t, n_in, n_out, cnt in CASES:
        line = []
        shipped = rule_rows(t, n_in)
        for rows in (2, 4, 8, 16):
            g = launch(t, n_in, n_out, rows)
            line.append("%8d" % g["blocks"])
        note = ""
        if shipped == 2:
            note = "shipped already 2"
        elif n_in // TRAITS[t][0] < TRAITS[t][2]:
            note = "shipped path is r4 (same kernel)"
        if n_out % 16 != 0:
            note = (note + "; " if note else "") + "n_out % 16 != 0 (tail guard)"
        print("%-22s %s   %s" % (name, " ".join(line), note))
    print()
    print("   blocks only: the knob changes which rows a work-group covers.  A thread's kbx sequence, its")
    print("   accumulation order and the warp/cross-warp reduction tree are identical for every ROWS, so every")
    print("   rN column is bitwise equal to the shipped kernel -- the bench's section 6 checks that on the GPU.")
    print()

    print("=" * 150)
    print("the same three layouts against the floor, per work-group (the card's step 1 in one table)")
    print("=" * 150)
    print("%-22s %-28s %7s %6s %6s %7s %9s %8s %9s" %
          ("case", "layout", "blocks", "act", "iters", "wB/blk", "wB/act", "thr/wB", "out/blk"))
    for name, t, n_in, n_out, cnt in CASES:
        for label, rows, blocks_override in (("exact, ROWS = the rule", rule_rows(t, n_in), None),
                                             ("generic, ROWS = 2, NW = 4", 2, (n_out + 1) // 2)):
            g = launch(t, n_in, n_out, rows)
            blocks = blocks_override if blocks_override is not None else g["blocks"]
            wblock = g["wrow"] * rows
            print("%-22s %-28s %7d %6d %6.1f %7d %9.1f %8.3f %9d" %
                  (name, label, blocks, g["active"], g["iters_per_active"], wblock, wblock / max(g["active"], 1),
                   g["thr_per_wb"], NCOLS * rows))
    print()
    print("   act = working threads per 128-thread block (the rest only store partials); iters = the kbx loop,")
    print("   per working thread; wB/act = the weight bytes one working thread reads in the whole call -- for the")
    print("   small-n_in shapes this is a few hundred bytes, which is why a call costs a fixed per-block price")
    print("   rather than a per-byte one.  out/blk = outputs one work-group owns, i.e. the work the block-level")
    print("   reduction (partial store + cross-warp add + warp butterfly) is spread over.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
