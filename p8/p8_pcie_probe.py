#!/usr/bin/env python3
"""P8 probe: read the PCIe config space of the two Battlemage cards and decode the PCI Express capability
(LnkCap / LnkSta / LnkCtl) the way lspci -vv does - lspci itself prints "<access denied>" for a non-root user."""
import struct
import sys

SPEED = {1: "2.5 GT/s", 2: "5 GT/s", 3: "8 GT/s", 4: "16 GT/s", 5: "32 GT/s", 6: "64 GT/s"}
WIDTH = {1: "x1", 2: "x2", 4: "x4", 8: "x8", 12: "x12", 16: "x16", 32: "x32"}


def cfg(bdf, n=256):
    with open(f"/sys/bus/pci/devices/{bdf}/config", "rb") as f:
        return f.read(n)


def cap_list(d):
    if not (d[6] & 0x10):
        return []
    ptr = d[0x34] & 0xFC
    seen = set()
    out = []
    while ptr and ptr not in seen and ptr + 1 < len(d):
        seen.add(ptr)
        cap = d[ptr]
        nxt = d[ptr + 1] & 0xFC
        out.append((ptr, cap))
        ptr = nxt
    return out


def show(bdf):
    d = cfg(bdf)
    print(f"=== {bdf}  (vendor {d[0]:#04x} device {d[2]:#04x} class {d[0x0b]:#04x}{d[0x0a]:02x}{d[0x09]:02x})")
    for off, cap in cap_list(d):
        if cap == 0x10:  # PCI Express
            lnkcap = struct.unpack_from("<I", d, off + 0x0C)[0]
            lnksta = struct.unpack_from("<H", d, off + 0x12)[0]
            lnkctl = struct.unpack_from("<H", d, off + 0x10)[0]
            slcap = struct.unpack_from("<I", d, off + 0x14)[0] if len(d) >= off + 0x18 else 0
            print(f"  PCIe cap at {off:#04x}: LnkCap={lnkcap:#010x} LnkSta={lnksta:#06x} LnkCtl={lnkctl:#06x}")
            print(f"    LnkCap.speed={SPEED.get(lnkcap & 0xF, lnkcap & 0xF)} LnkCap.width={WIDTH.get((lnkcap >> 4) & 0x3F, (lnkcap >> 4) & 0x3F)}")
            print(f"    LnkSta.speed={SPEED.get(lnksta & 0xF, lnksta & 0xF)} LnkSta.width={WIDTH.get((lnksta >> 4) & 0x3F, (lnksta >> 4) & 0x3F)}")
            print(f"    LnkSta.training={bool(lnksta & 0x800)} LnkCtl.retrain={bool(lnkctl & 0x20)}")
            print(f"    SlotCap2/power-budget-ish word @+0x14 = {slcap:#010x}")
        elif cap == 0x01:
            print(f"  PCI Power Management at {off:#04x}")
    for name in ("current_link_speed", "current_link_width", "max_link_speed", "max_link_width"):
        try:
            with open(f"/sys/bus/pci/devices/{bdf}/{name}") as f:
                print(f"  sysfs {name} = {f.read().strip()}")
        except OSError as e:
            print(f"  sysfs {name} = <{e.__class__.__name__}>")


for bdf in sys.argv[1:] or ["0000:03:00.0", "0000:08:00.0"]:
    show(bdf)
