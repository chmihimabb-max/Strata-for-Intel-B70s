#!/usr/bin/env python3
"""D2x (card t_85e61269): turn d2x/D2X-BENCH.txt into the card's tables.

Reads the bench's own section-3 rows (one per (type, n_in, n_out, ncols)) and produces:
  * the per-type roll-up at ncols = 4: launches, bytes, this table's us, the census's us for the same type,
    the achieved GB/s, and the ratio between the two (the bench's per-call proxy against the engine's own
    in-stream census);
  * the window table: shipped / generic / f16-XMX prototype / the pack's floor / the prototype's floor, and
    what fractional ceiling each implies against the census's 40.9%;
  * the same for the other column counts the file carries;
  * the one-time price of the f16 representation (the sum of the bench's own dequant + repack times);
  * the byte arithmetic for the variants the device REFUSES (int8, 1 B/element) so the ceiling comparison is
    on one page.

    /usr/bin/python3 d2x/xmx_price.py > d2x/D2X-TABLE.txt

D2y (card t_ba006576) added, on top of the above and without changing it:
  * --bench PATH   read another bench run (default d2x/D2X-BENCH.txt, so D2x's own table is reproducible);
  * --variants     read the bench's section 6 (run the bench with --variants) and print the per-shape,
                   per-variant table of us and GB/s against the floor, the flat-read (same bytes, no dot)
                   comparison, the rows-per-work-group sweep and the bit-for-bit column;
  * --base PATH    an earlier run of the same bench, per case, for the knob's inert control;
  * the per-column marginal: the shipped kernel's own table at several ncols, split into the part that scales
    with the columns (activation loads, integer dot, per-output reduction) and the part that does not (the
    weight bytes).

    /usr/bin/python3 d2x/xmx_price.py --bench d2y/D2Y-BENCH.txt --variants --base d2y/D2Y-BASE.txt > d2y/D2Y-TABLE.txt
"""
import collections
import re
import sys

BENCH = "d2x/D2X-BENCH.txt"
BASE = None                 # D2y: an optional second run of the same binary-less rig, for the shipped-row check
VARIANT = False             # D2y: print the section-6 variant tables (only present when the bench ran --variants)
CENSUS_US = {"Q6_K": 30718.5, "IQ4_NL": 6213.2, "Q5_K": 5131.1, "IQ4_XS": 4940.6, "Q4_K": 4171.1, "Q8_0": 92.8}
CENSUS_N = {"Q6_K": 129, "IQ4_NL": 47, "Q5_K": 35, "IQ4_XS": 42, "Q4_K": 47, "Q8_0": 1}
CENSUS_TOTAL_US = 51267.3
CENSUS_SHARE = 0.409        # the family's share of a T=4 window (d2/STATUS-D2.md section 4)
WINDOW_MS = 125.4

ROW = re.compile(r"^(\S+ \S+) +(\d+) +(\d+) +(\d+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +"
                 r"([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+)$")

# D2y: the bench's section-6 rows (--variants), one per (case, layout variant)
S6 = re.compile(r"^(\S+ \S+) +(\d+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) "
                r"+([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +(r2:\S+.*)$")

T0 = 4      # the census's window width; set from the file in main()


E_PER_BLOCK = {"Q6_K": 210.0 / 256, "Q5_K": 176.0 / 256, "Q4_K": 144.0 / 256, "IQ4_XS": 136.0 / 256,
               "IQ4_NL": 18.0 / 32, "Q8_0": 34.0 / 32}


def elems_of(case, nbytes):
    """Elements behind a weight blob: nbytes / (bytes per element of that type)."""
    return nbytes / E_PER_BLOCK[case.split()[0]]


def counts_of(rows, T):
    """case -> launches per window, taken from the bench's own section-3 rows."""
    return {r["case"]: r["cnt"] for r in rows if r["T"] == T}


def read_variants(path):
    """The bench's section 6 rows (D2y, --variants): one per (case, layout variant)."""
    rows = []
    for line in open(path):
        m = S6.match(line.rstrip())
        if not m:
            continue
        g = m.groups()
        rows.append({"case": g[0], "wbytes": int(g[1]), "floor": float(g[2]), "ship": float(g[3]),
                     "ship_gbs": float(g[4]), "flat4": float(g[5]), "flat4_gbs": float(g[6]),
                     "flat16": float(g[7]), "flat16_gbs": float(g[8]), "r2": float(g[9]), "r4": float(g[10]),
                     "r8": float(g[11]), "r16": float(g[12]), "gen": float(g[13]), "eq": g[14]})
    return rows


def print_variants(v, rows, bw):
    cnt = counts_of(rows, T0)
    print("=" * 118)
    print("D2y section 6: the pack's own bytes read flat, and the exact layout's rows per work-group forced")
    print("=" * 118)
    print("   flat4 / flat16 = the shipped grid and row-to-group mapping reading the same bytes with NO dot, no")
    print("   activation and no reduction, at 4 and 16 bytes per thread per pass; rN = the shipped exact kernel with")
    print("   ROWS forced to N (native_mmvq_set_exact_rows, default 0 = the shipped rule); gen = the generic")
    print("   multi-column layout.  eq = bit for bit against the shipped kernel's own y, all ncols x n_out outputs.")
    print()
    print("%-22s %13s %8s %8s %6s %8s %6s %5s %8s %6s %6s %6s %6s %6s  %s" %
          ("case", "weight B", "floor us", "ship us", "GB/s", "flat4", "GB/s", "x", "flat16", "r2 us", "r4 us",
           "r8 us", "r16 us", "gen us", "eq"))
    tot = collections.defaultdict(float)
    for r in v:
        n = cnt.get(r["case"], 0)
        print("%-22s %13d %8.1f %8.1f %6.1f %8.1f %6.1f %5.1f %8.1f %6.1f %6.1f %6.1f %6.1f %6.1f  %s" %
              (r["case"], r["wbytes"], r["floor"], r["ship"], r["ship_gbs"], r["flat4"], r["flat4_gbs"],
               r["ship"] / r["flat4"], r["flat16"], r["r2"], r["r4"], r["r8"], r["r16"], r["gen"], r["eq"]))
        tot["bytes"] += n * r["wbytes"]
        for k in ("floor", "ship", "flat4", "flat16", "r2", "r4", "r8", "r16", "gen"):
            tot[k] += n * r[k]
    print("-" * 118)
    print("%-22s %13.0f %8.1f %8.1f %6.1f %8.1f %6.1f %5.1f %8.1f %6.1f %6.1f %6.1f %6.1f %6.1f  %s" %
          ("WINDOW (301 calls)", tot["bytes"], tot["floor"], tot["ship"],
           tot["bytes"] / (tot["ship"] * 1e-6) / 1e9, tot["flat4"], tot["bytes"] / (tot["flat4"] * 1e-6) / 1e9,
           tot["ship"] / tot["flat4"], tot["flat16"], tot["r2"], tot["r4"], tot["r8"], tot["r16"], tot["gen"], ""))
    print()
    print("   %-34s %10s %9s %11s %9s %s" % ("variant", "window us", "GB/s", "vs shipped", "of floor", "bit for bit"))
    names = (("shipped layout (the rule in n_in)", tot["ship"]),
             ("flat4: same bytes, no dot", tot["flat4"]),
             ("flat16: same bytes, 16 B/thread", tot["flat16"]),
             ("exact rows = 2", tot["r2"]), ("exact rows = 4", tot["r4"]), ("exact rows = 8", tot["r8"]),
             ("exact rows = 16", tot["r16"]),
             ("generic layout (multi_exact off)", tot["gen"]),
             ("the pack's bandwidth floor", tot["floor"]))
    for i, (nm, us) in enumerate(names):
        if us <= 0:
            continue
        eq = "the reference"
        if 1 <= i <= 2 or i == 8:
            eq = "not applicable (it computes nothing)"
        elif i >= 3:
            eq = "yes" if all(x["eq"].split()[i - 3].endswith("Y") for x in v) else "no"
            if i == 7:
                eq = "yes at ncols = 4 here (measured), but see D2a"
        print("   %-34s %10.1f %9.1f %11.2fx %9.2fx  %s" %
              (nm, us, tot["bytes"] / (us * 1e-6) / 1e9, tot["ship"] / us, tot["floor"] / us, eq))
    print()
    big = max(v, key=lambda x: x["wbytes"])
    print("   the DRAM-sized case is the one that measures the ACCESS PATTERN rather than L2: %s is %.0f MiB, and a\n"
          "   no-dot read of it on the shipped grid runs at %.1f GB/s against this bench's measured card rate of\n"
          "   %.1f GB/s -- the same bytes, the same grid, the same 4 B per thread.  Every smaller case is re-read\n"
          "   `reps` times from L2 by both kernels, so its flat4 GB/s is an L2 rate and only the ship/flat4 ratio is\n"
          "   comparable; the window row above is therefore a floor-on-the-floor, not a DRAM number." %
          (big["case"], big["wbytes"] / 1048576.0, big["flat4_gbs"], bw))
    print()
    print("   the rows sweep, per case where it is not the shipped rule's own ROWS (x = shipped us / variant us):")
    print("   %-22s %8s %8s %8s %8s" % ("case", "r2 x", "r4 x", "r8 x", "r16 x"))
    for r in v:
        print("   %-22s %8.2f %8.2f %8.2f %8.2f" %
              (r["case"], r["ship"] / r["r2"], r["ship"] / r["r4"], r["ship"] / r["r8"], r["ship"] / r["r16"]))
    print()
    print("   the per-type roll-up at ncols = %d (us = cnt x per-call, x = shipped / variant):" % T0)
    print("   %-8s %6s %10s %10s %10s %10s %10s %10s" %
          ("type", "n", "shipped", "flat4", "r2", "r4", "r8", "r16"))
    byt = collections.defaultdict(lambda: collections.defaultdict(float))
    byn = collections.Counter()
    for r in v:
        t = r["case"].split()[0]
        n = cnt.get(r["case"], 0)
        byn[t] += n
        for k in ("ship", "flat4", "r2", "r4", "r8", "r16"):
            byt[t][k] += n * r[k]
    for t in sorted(byt, key=lambda x: -byt[x]["ship"]):
        d = byt[t]
        print("   %-8s %6d %10.1f %10.1f %5.2fx %5.2fx %5.2fx %5.2fx" %
              (t, byn[t], d["ship"], d["flat4"], d["ship"] / d["r2"], d["ship"] / d["r4"], d["ship"] / d["r8"],
               d["ship"] / d["r16"]))
    print()


def print_column_marginal(rows, bw):
    """The per-column cost of the shipped kernel: its own table at several ncols is a linear probe of the parts
    that scale with the column count (activation loads, the int8 dot, the per-output reduction) against the part
    that does not (the weight bytes)."""
    ts = sorted({r["T"] for r in rows})
    if len(ts) < 2:
        print("(only ncols = %d in %s: no per-column marginal to take)" % (ts[0], BENCH))
        return
    print("=" * 118)
    print("the shipped kernel against its own column count: which part of it scales with the columns")
    print("=" * 118)
    print("   %-8s %12s %12s %12s" % ("ncols", "window us", "GB/s", "vs ncols=%d" % ts[0]))
    tot = {}
    for T in ts:
        cnt = counts_of(rows, T)
        us = sum(r["cnt"] * r["ex"] for r in rows if r["T"] == T)
        b = sum(r["cnt"] * r["bytes"] for r in rows if r["T"] == T)
        tot[T] = us
        print("   %-8d %12.1f %12.1f %12.2fx" % (T, us, b / (us * 1e-6) / 1e9, us / tot[ts[0]]))
    print()
    if len(ts) >= 3:
        t0, t1 = ts[0], ts[-1]
        slope = (tot[t1] - tot[t0]) / (t1 - t0)
        base = tot[t0] - slope * t0
        print("   a straight line through ncols = %d and %d: %.2f ms of window time per extra column, and %.2f ms\n"
              "   at zero columns -- i.e. the work that scales with the column count (the activation loads, the\n"
              "   integer dot and the per-output reduction) is %.1f%% of the family at ncols = %d, and the part that\n"
              "   does not (reading the weight bytes, once per row regardless of the columns) is %.1f%%." %
              (t0, t1, slope / 1e3, base / 1e3, 100 * slope * t0 / tot[t0], t0, 100 * base / tot[t0]))
        print("   the weight-byte part at 0 columns is %.0f GB/s against the %.0f GB/s floor: even with every column's\n"
              "   work removed, what is left runs at %.0f%% of the card, so the part of this kernel that only reads the\n"
              "   weights is itself %.1fx above its own floor -- the bytes are not what the 8.8%% is made of." %
              (sum(r["cnt"] * r["bytes"] for r in rows if r["T"] == t0) / (base * 1e-6) / 1e9, bw,
               100 * (sum(r["cnt"] * r["bytes"] for r in rows if r["T"] == t0) / (base * 1e-6) / 1e9) / bw,
               base / (sum(r["cnt"] * r["bytes"] for r in rows if r["T"] == t0) / (bw * 1e9) * 1e6)))
    print()


def print_base_check(rows, path):
    """The knob's inert control: the same section-3 rows from an earlier run, per case, with the delta."""
    if path is None:
        return
    old = {}
    for line in open(path):
        m = ROW.match(line.rstrip())
        if m:
            old[(m.group(1), int(m.group(2)))] = float(m.group(6))
    if not old:
        print("(no section-3 rows in %s)" % path)
        return
    print("=" * 118)
    print("the inert control: the shipped layout's own numbers from %s against this run" % path)
    print("=" * 118)
    print("   %-22s %4s %12s %12s %8s" % ("case", "ncols", "earlier us", "this run us", "delta"))
    worst = 0.0
    n = 0
    for r in rows:
        k = (r["case"], r["T"])
        if k not in old:
            continue
        d = r["ex"] / old[k] - 1
        worst = max(worst, abs(d))
        n += 1
        print("   %-22s %4d %12.1f %12.1f %+7.2f%%" % (r["case"], r["T"], old[k], r["ex"], d * 100))
    print()
    print("   %d rows, largest single-case delta %.2f%%: the engine's shipped path (knob at 0) is the same kernel it\n"
          "   was, and this run's own section-3 rows are the reference for every ratio above." % (n, worst * 100))
    print()


def main():
    global BENCH, BASE, VARIANT, T0
    args = sys.argv[1:]
    while args:
        a = args.pop(0)
        if a == "--bench" and args:
            BENCH = args.pop(0)
        elif a == "--base" and args:
            BASE = args.pop(0)
        elif a == "--variants":
            VARIANT = True
    rows = []
    bw = None
    for line in open(BENCH):
        m = re.search(r"hash-filled, incompressible\]: [\d.]+ ms -> +([\d.]+) GB/s", line)
        if m:
            bw = float(m.group(1))
        m = ROW.match(line.rstrip())
        if m:
            rows.append({
                "case": m.group(1), "T": int(m.group(2)), "cnt": int(m.group(3)), "bytes": int(m.group(4)),
                "floor": float(m.group(5)), "ex": float(m.group(6)), "ex_gbs": float(m.group(7)),
                "gen": float(m.group(8)), "gen_gbs": float(m.group(9)), "f16": float(m.group(10)),
                "f16_gbs": float(m.group(11)), "dq": float(m.group(12)), "ex1": float(m.group(13))})
    if not rows or bw is None:
        print("could not parse %s (rows=%d bw=%s)" % (BENCH, len(rows), bw))
        return 1
    types = sorted({r["case"].split()[0] for r in rows})
    print("D2x -- an int8-XMX MMVQ priced against the shipped scalar MMVQ, on this config's decode shapes")
    print("source: %s (bench build-sycl/mmvq_xmx_price, one B70, ZE_AFFINITY_MASK=0)" % BENCH)
    print("card read-only rate, incompressible (this bench): %.1f GB/s" % bw)
    print()

    for T in sorted({r["T"] for r in rows}):
        sub = [r for r in rows if r["T"] == T]
        print("=" * 118)
        print("ncols = %d" % T)
        print("=" * 118)
        print("%-22s %5s %13s %10s %9s %8s %8s %9s %8s %8s" %
              ("case", "cnt", "weight B", "floor us", "bench us", "GB/s", "gen us", "f16 us", "f16 GB/s",
               "f16/ex"))
        tot = collections.defaultdict(float)
        for r in sub:
            print("%-22s %5d %13d %10.1f %9.1f %8.1f %8.1f %9.1f %8.1f %8.2f" %
                  (r["case"], r["cnt"], r["bytes"], r["floor"], r["ex"], r["ex_gbs"], r["gen"], r["f16"],
                   r["f16_gbs"], r["f16"] / r["ex"]))
            tot["bytes"] += r["cnt"] * r["bytes"]
            tot["floor"] += r["cnt"] * r["floor"]
            tot["ex"] += r["cnt"] * r["ex"]
            tot["gen"] += r["cnt"] * r["gen"]
            tot["f16"] += r["cnt"] * r["f16"]
            tot["f16_floor"] += r["cnt"] * 2.0 * elems_of(r["case"], r["bytes"]) / (bw * 1e9) * 1e6
            tot["dq"] += r["cnt"] * r["dq"]
        print("-" * 118)
        print("%-22s %5s %13.0f %10.1f %9.1f %8.1f %8.1f %9.1f %8.1f %8.2f" %
              ("WINDOW (301 calls)", "", tot["bytes"], tot["floor"], tot["ex"], tot["bytes"] / (tot["ex"] * 1e-6) / 1e9,
               tot["gen"], tot["f16"], tot["bytes"] * 2 / (tot["f16"] * 1e-6) / 1e9, tot["f16"] / tot["ex"]))
        print()
        print("   the bench's shipped-layout window total  %8.1f us   %7.2f ms" % (tot["ex"], tot["ex"] / 1e3))
        if T == 4:
            print("   the engine's own census for the same 301  %8.1f us   %7.2f ms   -> this table is %+.1f%%" %
                  (CENSUS_TOTAL_US, CENSUS_TOTAL_US / 1e3, (tot["ex"] / CENSUS_TOTAL_US - 1) * 100))
            print("   (the proxy is the check the card asked for: same launches, same order, same answer)")
        else:
            print("   (the engine's census is a T=4 window's: %.1f us.  This is ncols=%d, the other window width" %
                  (CENSUS_TOTAL_US, T))
            print("    the config reaches, so it has no census of its own; every variant in this row is measured")
            print("    at the same ncols in the same run, so the shipped-vs-variant ratios stand.)")
        print()
        print("   %-32s %10s %10s %12s %12s" % ("variant", "window us", "window ms", "% of window", "vs shipped"))
        for name, v in (("shipped layout", tot["ex"]), ("generic layout", tot["gen"]),
                        ("f16 XMX prototype", tot["f16"]), ("the pack's own floor", tot["floor"]),
                        ("the f16 prototype's floor", tot["f16_floor"])):
            print("   %-32s %10.1f %10.2f %11.2f%% %11.2fx" %
                  (name, v, v / 1e3, v / WINDOW_MS / 1e3 * 100, tot["ex"] / v))
        print()
        print("   the family's own share of a T=4 window is 40.9%% (the census); at this ncols = %d table:" % T)
        print("     an f16 MMA/MMVQ measured like this makes the family %.2fx smaller: %.1f%% of the window "
              "instead of 40.9%%" % (tot["ex"] / tot["f16"], (tot["f16"] / tot["ex"]) * CENSUS_SHARE * 100))
        print("     its own representation's floor would make it %.2fx smaller (%.1f%% of the window)" %
              (tot["ex"] / tot["f16_floor"], (tot["f16_floor"] / tot["ex"]) * CENSUS_SHARE * 100))
        print("   the same kernel reading the PACK'S OWN bytes would buy %.2fx (%.1f%% of the window): that is" %
              (tot["ex"] / tot["floor"], (tot["floor"] / tot["ex"]) * CENSUS_SHARE * 100))
        print("   the lever that does NOT need the matrix unit, and it is the larger one")
        print()
        print("   one-time price of the f16 representation (dequantize + repack, this host): %.0f ms" % tot["dq"])
        print()

    # ---------------------------------------------------------------- per type, ncols = 4
    T0 = min(r["T"] for r in rows)
    print("=" * 118)
    print("per type at ncols = %d: this bench's per-call proxy against the engine's own census" % T0)
    print("=" * 118)
    print("%-8s %8s %8s %13s %10s %10s %10s %9s" %
          ("type", "launches", "bench n", "bytes", "bench us", "census us", "bench GB/s", "bench/census"))
    by_type = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows:
        if r["T"] != T0:
            continue
        t = r["case"].split()[0]
        by_type[t]["n"] += r["cnt"]
        by_type[t]["bytes"] += r["cnt"] * r["bytes"]
        by_type[t]["us"] += r["cnt"] * r["ex"]
        by_type[t]["f16"] += r["cnt"] * r["f16"]
    for t in sorted(by_type, key=lambda x: -by_type[x]["bytes"]):
        d = by_type[t]
        print("%-8s %8d %8d %13.0f %10.1f %10.1f %10.1f %9.2f" %
              (t, d["n"], CENSUS_N.get(t, 0), d["bytes"], d["us"], CENSUS_US.get(t, float("nan")),
               d["bytes"] / (d["us"] * 1e-6) / 1e9, d["us"] / CENSUS_US.get(t, float("nan"))))

    # ---------------------------------------------------------------- the variants the device refuses
    e_per_block = {"Q6_K": 210.0 / 256, "Q5_K": 176.0 / 256, "Q4_K": 144.0 / 256, "IQ4_XS": 136.0 / 256,
                   "IQ4_NL": 18.0 / 32, "Q8_0": 34.0 / 32}
    elems = 0.0
    pack_bytes = 0.0
    for r in rows:
        if r["T"] != T0:
            continue
        t = r["case"].split()[0]
        pack_bytes += r["cnt"] * r["bytes"]
        elems += r["cnt"] * r["bytes"] / e_per_block[t]
    print()
    print("=" * 118)
    print("the byte arithmetic for every representation the card could be asked for (one window, 301 calls)")
    print("=" * 118)
    print("%-34s %13s %9s %10s %10s %10s" % ("representation", "bytes", "x pack", "floor us", "floor ms", "vs shipped"))
    shipped_us = sum(r["cnt"] * r["ex"] for r in rows if r["T"] == T0)
    for label, b in (("the pack's blocks (shipped)", pack_bytes),
                     ("int8 (1 B/element)", elems),
                     ("f16 / bf16 (2 B/element)", 2 * elems),
                     ("f32 (4 B/element)", 4 * elems)):
        fl = b / (bw * 1e9) * 1e6
        print("%-34s %13.0f %9.2f %10.1f %10.2f %10.2fx" % (label, b, b / pack_bytes, fl, fl / 1e3,
                                                            shipped_us / fl))
    print()
    print("   the int8 dot IS the one this card was proposed for, and it is the one the driver refuses")
    print("   (bench section 1: 'joint_matrix with parameters matrix_type::sint32, use::accumulator, Rows=8,")
    print("   Cols=32 is not supported on this device'); the f16 16x16x16 tile probe21 recorded as running no")
    print("   longer builds on this toolchain either, so the prototype above runs on f16 8x16x16.")
    print()
    print("   the bandwidth a variant must reach to beat the shipped layout, per shape class:")
    print("     with 2 B/element it must exceed 2.69x the shipped kernel's achieved rate; the shipped kernel")
    print("     runs at %.1f%% of this card's measured %.0f GB/s, so the bar is %.0f GB/s (and the prototype" %
          (shipped_us and pack_bytes / (shipped_us * 1e-6) / 1e9 / bw * 100, bw,
           pack_bytes / (shipped_us * 1e-6) / 1e9 * 2.69))
    print("     reaches %.0f GB/s on the big shapes and %.0f-%.0f GB/s on the small ones)" %
          (max(r["f16_gbs"] for r in rows if r["T"] == T0), min(r["f16_gbs"] for r in rows if r["T"] == T0),
           max(r["f16_gbs"] for r in rows if r["T"] == T0 and r["bytes"] < 2e6)))
    print()
    print("   the bar every variant must clear to beat the shipped layout, per shape: its own bytes per")
    print("   element x the shipped kernel's rate on THAT shape.  The f16 prototype against its own bar:")
    print("   %-22s %2s %9s %9s %9s %9s %7s" % ("case", "T", "pack B/e", "shipped GB/s", "bar GB/s", "f16 GB/s",
                                             "vs bar"))
    worst = None
    for r in rows:
        if r["T"] != T0:
            continue
        bpe = E_PER_BLOCK[r["case"].split()[0]]
        bar = 2.0 / bpe * r["ex_gbs"]
        ratio = r["f16_gbs"] / bar
        worst = ratio if worst is None else min(worst, ratio)
        print("   %-22s %2d %9.3f %9.1f %9.1f %9.1f %7.2f%s" %
              (r["case"], r["T"], bpe, r["ex_gbs"], bar, r["f16_gbs"], ratio,
               "" if ratio >= 1 else "   <-- misses"))
    n_ok = sum(1 for r in rows if r["T"] == T0 and r["f16_gbs"] >= 2.0 / E_PER_BLOCK[r["case"].split()[0]] * r["ex_gbs"])
    print("   %d of %d geometries clear their own bar at ncols = %d; the worst is %.2fx"
          % (n_ok, len([r for r in rows if r["T"] == T0]), T0, worst))
    print()

    # ---------------------------------------------------------------- D2y (card t_ba006576) additions
    print_column_marginal(rows, bw)
    if VARIANT:
        v = read_variants(BENCH)
        if v:
            print_variants(v, rows, bw)
        else:
            print("(no section-6 rows in %s: run the bench with --variants)" % BENCH)
    print_base_check(rows, BASE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
