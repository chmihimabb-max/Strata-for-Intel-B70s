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
"""
import collections
import re
import sys

BENCH = "d2x/D2X-BENCH.txt"
CENSUS_US = {"Q6_K": 30718.5, "IQ4_NL": 6213.2, "Q5_K": 5131.1, "IQ4_XS": 4940.6, "Q4_K": 4171.1, "Q8_0": 92.8}
CENSUS_N = {"Q6_K": 129, "IQ4_NL": 47, "Q5_K": 35, "IQ4_XS": 42, "Q4_K": 47, "Q8_0": 1}
CENSUS_TOTAL_US = 51267.3
CENSUS_SHARE = 0.409        # the family's share of a T=4 window (d2/STATUS-D2.md section 4)
WINDOW_MS = 125.4

ROW = re.compile(r"^(\S+ \S+) +(\d+) +(\d+) +(\d+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+) +"
                 r"([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+)$")


E_PER_BLOCK = {"Q6_K": 210.0 / 256, "Q5_K": 176.0 / 256, "Q4_K": 144.0 / 256, "IQ4_XS": 136.0 / 256,
               "IQ4_NL": 18.0 / 32, "Q8_0": 34.0 / 32}


def elems_of(case, nbytes):
    """Elements behind a weight blob: nbytes / (bytes per element of that type)."""
    return nbytes / E_PER_BLOCK[case.split()[0]]


def main():
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
