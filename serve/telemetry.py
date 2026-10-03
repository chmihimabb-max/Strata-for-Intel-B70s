"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power.  With the Intel backend (the SYCL/XPU port):
  the xe driver's Linux sysfs files and its own per-process /proc/<pid>/fdinfo counters - load, VRAM, host-memory
  footprint, temperature, fan, clock and power derived from the energy counter, with anything this platform cannot
  read reported as unavailable rather than as 0.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
import platform
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


# ------------------------------------------------------------------------------------------------ AMD (Linux sysfs)
SYSFS = "/sys"


def amd_device_dir(index, sysfs=None):
    """The amdgpu sysfs folder (/sys/class/drm/renderD<N>/device) of the AMD GPU that HIP numbers `index`: the KFD
    topology's GPU nodes in order, the CPU nodes skipped, linked to their render node by drm_render_minor - the
    numbering setup's amd_gpus() and HIP_VISIBLE_DEVICES use.  None when there is no such card (or no amdgpu)."""
    base = os.path.join(sysfs or SYSFS, "class", "kfd", "kfd", "topology", "nodes")
    try:
        nodes = sorted((n for n in os.listdir(base) if n.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    for n in nodes:
        try:
            with open(os.path.join(base, n, "properties"), encoding="utf-8") as f:
                props = dict(line.strip().partition(" ")[::2] for line in f if line.strip())
            if int(props.get("gfx_target_version") or 0) == 0 or int(props.get("simd_count") or 0) == 0:
                continue
            gpus.append(props)
        except (OSError, ValueError):
            continue
    if not 0 <= index < len(gpus) or not gpus[index].get("drm_render_minor"):
        return None
    dev = os.path.join(sysfs or SYSFS, "class", "drm", "renderD" + gpus[index]["drm_render_minor"].strip(), "device")
    return dev if os.path.isdir(dev) else None


class _Amd:
    """#301: an AMD card's readings from the amdgpu driver's sysfs files (Linux; no ROCm library needed), with _Nvml's
    interface: load (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature
    (temp1_input, the edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap)."""

    def __init__(self, index=0, sysfs=None):
        self.dev = amd_device_dir(index, sysfs)
        self.hwmon = None
        if self.dev:
            try:
                hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
                self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
            except OSError:
                pass

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _int(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        try:
            with open(os.path.join(self.dev, "product_name"), encoding="utf-8") as f:
                return f.read().strip() or "AMD Radeon"
        except (OSError, TypeError):
            return "AMD Radeon"

    def read(self):
        out = {"util": self._int(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": self._int(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": self._int(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = self._int(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            p = self._int(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = self._int(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = self._int(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        return out


# ------------------------------------------------------------------------------------------------ Intel (Linux sysfs)
PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids", "/usr/share/pci.ids")


def _text(path):
    """A sysfs file's stripped content, or None when it does not exist / cannot be read."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except (OSError, ValueError, TypeError):
        return None


def _number(path, base=10):
    t = _text(path)
    try:
        return int(t, base) if t else None
    except (TypeError, ValueError):
        return None


def _pci_name(vendor, device):
    """The card's name from the pci.ids database lspci reads (no Intel GPU names itself anywhere in sysfs); None when
    the database is not installed - the caller then shows the ids instead."""
    want = f"{device:04x}"
    for path in PCI_IDS:
        try:
            f = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with f:
            ours = False
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                if line[0] not in "\t ":
                    if ours:                     # past the vendor's block: the device is not in this database
                        break
                    ours = line.startswith(f"{vendor:04x} ")
                elif ours and line.startswith("\t") and not line.startswith("\t\t"):
                    tok = line.strip().split(None, 1)
                    if tok and tok[0] == want:
                        return tok[1].strip() if len(tok) > 1 else None
    return None


def intel_device_dirs(sysfs=None):
    """Every Intel display device's PCI folder (/sys/bus/pci/devices/<bdf>), in PCI order."""
    base = os.path.join(sysfs or SYSFS, "bus", "pci", "devices")
    out = []
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for n in names:
        v = _number(os.path.join(base, n, "vendor"), 16)
        c = _number(os.path.join(base, n, "class"), 16)
        if v == 0x8086 and c is not None and (c >> 16) == 0x03:      # VGA / XGA / 3D controller
            out.append(os.path.join(base, n))
    return out


def intel_device_dir(index=0, sysfs=None):
    """The Intel GPU the SYCL/Level Zero runtime numbers `index`: the display devices in PCI order.  Measured on the
    two B70s, 0 -> 0000:03:00.0 (= /dev/dri/card0, renderD128) and 1 -> 0000:08:00.0 (card1, renderD129) - the same
    order the driver's fdinfo files and the engine's ZE_AFFINITY_MASK use.  None when there is no such card."""
    dirs = intel_device_dirs(sysfs)
    return dirs[index] if 0 <= index < len(dirs) else None


def _fdinfo(path):
    """One /proc/<pid>/fdinfo file as {key: value}: the lines are "key:<tab>value", and the driver's own keys
    (drm-pdev, drm-total-vram0, drm-cycles-<class>, ...) are the ones kept."""
    out = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                key, sep, val = line.partition("\t")
                if sep and key.startswith("drm-"):
                    out[key.rstrip(":").strip()] = val.strip()
    except (OSError, ValueError, TypeError):
        return {}
    return out


def _kib(v):
    """A fdinfo size ("27456368 KiB") in bytes, or None."""
    if not v:
        return None
    try:
        return int(str(v).split()[0]) << 10
    except (IndexError, ValueError):
        return None


def render_clients(proc=None):
    """{pdev: [fdinfo, ...]}: the driver's per-process numbers for every process holding a /dev/dri/renderD* fd,
    keyed by the card that fd belongs to.  This is the only per-card VRAM and GPU-busy source this platform has (the
    same numbers intel_gpu_top builds its per-process view from); one readlink per open fd finds the processes."""
    root = proc or "/proc"
    out: dict = {}
    try:
        pids = [p for p in os.listdir(root) if p.isdigit()]
    except OSError:
        return None                                   # cannot look: the caller must not read that as "nothing there"
    for pid in pids:
        fds = os.path.join(root, pid, "fd")
        try:
            names = os.listdir(fds)
        except OSError:
            continue                                  # another user's process, or it just exited
        for fd in names:
            try:
                if "renderD" not in os.readlink(os.path.join(fds, fd)):
                    continue
            except OSError:
                continue
            kv = _fdinfo(os.path.join(root, pid, "fdinfo", fd))
            if kv.get("drm-pdev"):
                out.setdefault(kv["drm-pdev"], []).append(kv)
    # one process can hold several render-node fds of the same card (Xorg held four here) and each fd repeats the
    # same client's numbers, so a client is counted once - summing the fds would multiply its memory and its cycles
    for pdev, rows in out.items():
        one = {}
        for kv in rows:
            one.setdefault(kv.get("drm-client-id") or f"?{id(kv)}", kv)
        out[pdev] = list(one.values())
    return out


class _Intel:
    """The Intel backend (the SYCL/XPU port): an Intel card's readings with _Nvml's interface, from the xe/i915
    driver's own Linux files and no extra tooling (no xpu-smi, no intel_gpu_top, no perf - none are on this host):
      util        the compute engines' busy share of the card's own cycle clock, from the clients' drm-cycles-ccs and
                  drm-total-cycles-ccs across the sampler's interval (intel_gpu_top's derivation).  Validated against
                  a known workload: a 3.03 s copy burst in a 147.7 s window reads 2.06% of the copy engine's cycles;
      mem_used    what the card's render-node clients hold in VRAM (drm-total-vram0, summed);
      mem_total   the VRAM aperture the driver maps (PCI BAR 2 of the resource file; this driver exposes no VRAM size);
      temp        the hwmon sensor **chosen by its label**, not by index: this card has no temp1 at all and its labels
                  are temp2="pkg", temp3="vram", temp4="mctrl", temp5="pcie", temp6..temp21="vram_ch_0..15", so the
                  package sensor is the GPU temperature (as NVML's GPU temperature is);
      power       derived from energy1_input (a microjoule counter, label "card") across the sampler's interval: this
                  driver has no power1_average and no power1_input, so an interval is what makes it meaningful and the
                  first sample has none (it is None, never 0);
      power_limit power1_cap.
    Every field this backend has no source for comes back None, never 0 - a zeroed load, power or PCIe reading reads
    as "idle" and that would be a lie; `unsupported()` carries the reason for each one and `sources()` its origin."""

    def __init__(self, index=0, sysfs=None, proc=None):
        self.dev = intel_device_dir(index, sysfs)
        self.proc = proc
        self.pdev = os.path.basename(self.dev) if self.dev else None
        self.hwmon = None
        self.labels: dict = {}
        if self.dev:
            hw = os.path.join(self.dev, "hwmon")
            try:
                hs = sorted(h for h in os.listdir(hw) if h.startswith("hwmon"))
            except OSError:
                hs = []
            if hs:
                self.hwmon = os.path.join(hw, hs[0])
                try:
                    names = os.listdir(self.hwmon)
                except OSError:
                    names = []
                for n in names:
                    if n.startswith("temp") and n.endswith("_label"):
                        self.labels[n[:-6]] = _text(os.path.join(self.hwmon, n)) or ""
        self._energy = None      # (t, µJ) of the previous read: power is the delta across the interval
        self._energy_pkg = None
        self._cycles = None      # (t, {client: (cycles, total)}) of the previous read: util is the delta

    def ok(self):
        return self.dev is not None

    def name(self):
        v = _number(os.path.join(self.dev, "vendor"), 16)
        d = _number(os.path.join(self.dev, "device"), 16)
        named = _pci_name(v, d) if (v is not None and d is not None) else None
        if named:
            return named
        return f"Intel Graphics [{v:04x}:{d:04x}]" if (v is not None and d is not None) else "Intel Graphics"

    # -- the parts everything else is built from ---------------------------------------------------------------
    def _temp(self, label):
        """The temperature of the sensor whose label is `label` (m°C -> °C); None when there is no such sensor."""
        for name, lab in self.labels.items():
            if lab == label:
                v = _number(os.path.join(self.hwmon, name + "_input"))
                return v / 1000.0 if v is not None else None
        return None

    def _vram_temps(self):
        return [v / 1000.0 for n, lab in self.labels.items() if lab.startswith("vram_ch")
                for v in [_number(os.path.join(self.hwmon, n + "_input"))] if v is not None]

    def _busy(self, clients, now):
        """The clients' busy share per engine class across the interval since the previous read, from the driver's own
        counters: 100 * delta(cycles<class>) / delta(total-cycles-<class>).  {} when there was no previous read or the
        card is not running anything (there is no counter to derive from, and that is not a zero)."""
        if not clients:
            self._cycles = None
            return {}
        cur = {}
        for c in clients:
            cyc = {k[len("drm-cycles-"):]: int(v) for k, v in c.items()
                   if k.startswith("drm-cycles-") and v.isdigit()}
            tot = {k[len("drm-total-cycles-"):]: int(v) for k, v in c.items()
                   if k.startswith("drm-total-cycles-") and v.isdigit()}
            cur[c.get("drm-client-id") or "?"] = (cyc, tot)
        prev, self._cycles = self._cycles, (now, cur)
        if not prev:
            return {}
        out = {}
        for cls in sorted({c for cyc, _ in cur.values() for c in cyc}):
            d, total = 0, 0
            for cid, (cyc, tot) in cur.items():
                if cid not in prev[1]:
                    continue
                d += cyc.get(cls, 0) - prev[1][cid][0].get(cls, 0)
                total = max(total, tot.get(cls, 0) - prev[1][cid][1].get(cls, 0))
            if total > 0:
                out[cls] = max(0.0, min(100.0, 100.0 * d / total))
        return out

    def _power(self, uj, prev_attr, now):
        """µJ counter -> W over the interval since the previous read; None on the first read (no interval yet) and
        None when the counter went backwards (a reset, e.g. after a suspend)."""
        state = getattr(self, prev_attr)
        if uj is None:
            return None
        setattr(self, prev_attr, (now, uj))
        if not state or now <= state[0] or uj < state[1]:
            return None
        return round((uj - state[1]) / 1e6 / (now - state[0]), 2)

    def _vram_total(self):
        """The VRAM aperture: PCI BAR 2 of the driver's `resource` file (the card's Region 2; checked here against
        lspci: line 0 = the 16 MiB MMIO window, line 2 = the 32 GiB aperture).  This driver exposes no VRAM size
        anywhere, so the aperture is the closest honest reading - and if a client holds more than it, it is not the
        VRAM size after all and the answer is None."""
        try:
            with open(os.path.join(self.dev, "resource"), encoding="utf-8") as f:
                lines = f.read().splitlines()
            lo, hi = (int(x, 16) for x in lines[2].split()[:2])
            return hi - lo + 1 if hi >= lo > 0 else None
        except (OSError, IndexError, ValueError):
            return None

    def _clock(self):
        """The first tile/gt's current and maximum frequency in MHz, and the GT's own idle state, when the driver
        exposes them (tile0/gt0/freq0 on this card; a card with no such files answers None)."""
        for tile in sorted(os.listdir(self.dev) if self.dev else []):
            if not tile.startswith("tile"):
                continue
            gt = os.path.join(self.dev, tile)
            for g in sorted(os.listdir(gt)):
                f = os.path.join(gt, g, "freq0")
                if os.path.isdir(f):
                    return (_number(os.path.join(f, "act_freq")), _number(os.path.join(f, "max_freq")),
                            _text(os.path.join(gt, g, "gtidle", "idle_status")))
        return None, None, None

    # -- the payload -------------------------------------------------------------------------------------------
    def read(self):
        now = time.time()
        out = {}
        clients = render_clients(self.proc)
        if clients is not None:
            mine = clients.get(self.pdev) or []
            for key, field in (("drm-total-vram0", "mem_used"), ("drm-total-gtt", "mem_gtt"),
                               ("drm-total-system", "mem_system")):
                vals = [v for v in (_kib(c.get(key)) for c in mine) if v is not None]
                out[field] = sum(vals) if vals else 0
            out["clients"] = len(mine)
            busy = self._busy(mine, now)
            out["util"] = busy.get("ccs")            # the compute engines: NVML's "GPU utilization" on this driver
            out["copy_busy"] = busy.get("bcs")       # the copy engines, which is what host<->device traffic uses
            out["render_busy"] = busy.get("rcs")
            out["video_busy"] = busy.get("vcs")
        total = self._vram_total()
        if total is not None and out.get("mem_used") is not None and out["mem_used"] > total:
            total = None                             # the aperture is smaller than what is held: not the VRAM size
        out["mem_total"] = total
        if self.hwmon:
            out["temp"] = self._temp("pkg")
            out["temp_vram"] = self._temp("vram")
            vt = self._vram_temps()
            out["temp_vram_max"] = max(vt) if vt else None
            out["temp_pcie"] = self._temp("pcie")
            fan = _number(os.path.join(self.hwmon, "fan1_input"))
            out["fan_rpm"] = fan
            cap = _number(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
            crit = _number(os.path.join(self.hwmon, "power1_crit"))
            out["power_crit"] = crit / 1e6 if crit is not None else None
            out["power"] = self._power(_number(os.path.join(self.hwmon, "energy1_input")), "_energy", now)
            out["power_pkg"] = self._power(_number(os.path.join(self.hwmon, "energy2_input")), "_energy_pkg", now)
        freq, freq_max, idle = self._clock()
        out["freq_mhz"], out["freq_max_mhz"], out["gt_idle"] = freq, freq_max, idle
        # no PCIe figures here on purpose, see unsupported()
        out["pcie_gen"] = out["pcie_gen_max"] = out["pcie_width"] = None
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None
        return out

    def sources(self):
        """field -> where its number comes from, for the payload and the Monitor."""
        s = {"util": "fdinfo drm-cycles-ccs / drm-total-cycles-ccs over the card's clients, across the interval",
             "copy_busy": "fdinfo drm-cycles-bcs / drm-total-cycles-bcs (the copy engines)",
             "mem_used": "fdinfo drm-total-vram0, summed over the card's render-node clients",
             "mem_gtt": "fdinfo drm-total-gtt (host memory mapped for the card)",
             "mem_system": "fdinfo drm-total-system",
             "mem_total": "PCI BAR 2 of the driver's resource file (the VRAM aperture)",
             "power": "hwmon energy1_input (label 'card') delta across the interval (this driver has no power1_input)",
             "power_limit": "hwmon power1_cap",
             "fan_rpm": "hwmon fan1_input"}
        for field, label in (("temp", "pkg"), ("temp_vram", "vram"), ("temp_pcie", "pcie")):
            for name, lab in self.labels.items():
                if lab == label:
                    s[field] = f"hwmon {name}_input, the sensor whose label is '{label}'"
                    break
        s["temp_vram_max"] = "the hottest of the vram_ch_* sensors"
        s["freq_mhz"] = "tile0/gt0/freq0/act_freq"
        return s

    def unsupported(self):
        """field -> why this platform has no reading for it (the payload says so instead of showing a 0, and the
        Monitor shows this line where the number would be)."""
        link = "the driver's link registers report a link state that the measured host->device bandwidth refutes"
        return {
            "pcie_gen": link + ", so no link figure is shown",
            "pcie_gen_max": link + ", so no link figure is shown",
            "pcie_width": link + ", so no link figure is shown",
            "pcie_rx_mb": "no PCIe byte counter in sysfs on this driver",
            "pcie_tx_mb": "no PCIe byte counter in sysfs on this driver",
        }


def gpu_reader(index=0, amd=False, intel=False):
    """The card's readings: NVML (NVIDIA), the amdgpu sysfs files with the AMD backend (#301), or the xe/i915 sysfs
    files and the driver's fdinfo counters with the Intel one (the SYCL/XPU port).  Without a flag, NVML first and the
    Intel files when there is no NVIDIA driver to ask - a machine with only the Intel card still fills the Monitor."""
    if amd:
        return _Amd(index)
    if intel:
        return _Intel(index)
    g = _Nvml(index)
    return g if g.ok() else _Intel(index)


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read."""
    g = gpu_reader(index, amd)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False, intel=False):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own.  `amd`: the AMD backend's
        cards, numbered as HIP numbers them, read from sysfs (#301).  `intel`: the Intel backend's cards, numbered as
        Level Zero/SYCL numbers them, read from the xe driver's sysfs files and fdinfo counters (the SYCL/XPU port)."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.gpus = [(i, gpu_reader(i, amd, intel)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.static = {
            "gpu_name": " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None,
            "gpu_count": len(self.gpus),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        # what each reading comes from, and - for the metrics the platform cannot provide - why there is none: the
        # Monitor shows the reason instead of a zero, because a zeroed load/power/PCIe figure reads as "idle" (P8)
        for key, fn in (("gpu_sources", "sources"), ("gpu_unsupported", "unsupported")):
            if hasattr(self.gpu, fn):
                self.static[key] = getattr(self.gpu, fn)()
        self._disk_prev = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self):
        s = {}
        if self.gpu.ok():
            reads = [(i, g.read()) for i, g in self.gpus]
            g = dict(reads[0][1])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "mem_gtt", "mem_system", "power", "power_limit", "pcie_rx_mb",
                          "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                for k in ("util", "copy_busy", "render_busy", "video_busy"):     # shares of a card: their mean
                    v = vals(k)
                    g[k] = sum(v) / len(v) if v else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                s["gpus"] = [dict(r, index=i) for i, r in reads]     # each card's own readings, unaggregated
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def _loop(self):
        while True:
            s = self.sample()
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "gpu_copy_busy", "cpu",
                          "ram_used", "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            time.sleep(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
