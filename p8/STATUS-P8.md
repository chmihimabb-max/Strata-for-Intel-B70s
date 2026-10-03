# P8 — the Intel/Arc backend for the Monitor's GPU telemetry, and "unavailable" instead of 0

Card `t_bf6af1fb` (P8), repo `/home/michael/strata-xpu/strata`, branch `sycl-xpu`, base `81ecdb2`; the change is
`7c3ffa1` (the backend, its test and the docs).  Harness and evidence under `p8/` (scripts) and `p8/evidence/`
(raw output).  Both B70s, config of record untouched (`strata-sycl-iq3s.json`), the server restarted and left
running and populated: **pid 1040401, `127.0.0.1:8099`** (engine pid 1040410).

## 0. Verdict

1. **The dashboard's GPU cards were empty because there was no Intel backend**, not because the wiring was broken:
   `serve/telemetry.py` had NVML (NVIDIA) and amdgpu-sysfs (AMD) and nothing for Intel, so every `gpu_*` field the
   UI reads was `null` (`/metrics` before the change: `hardware_static.gpu_name = null`, `gpu_count = 1`,
   `hardware` with no `gpu_*` keys at all - `p8/evidence/metrics-before-restart.json`).
2. **It is fixed and populated on both cards.**  At rest after the restart: `gpu_util 0.0`, `gpu_temp 58.0 °C`
   (67 °C once the request had run), `gpu_power 92.4 W`, `gpu_mem_used 62.56 GB / 68.72 GB`, `gpu_count 2`,
   `gpu_name "Battlemage G31 [Intel Graphics] + Battlemage G31 [Intel Graphics]"`.  Mid-request: `gpu_util 54-60%`,
   `gpu_power 273-276 W`, `gpu_copy_busy 45-51%`, `gpu_temp 64→68 °C` (`p8/evidence/live-proof.json`).
3. **The PCIe contradiction resolves against the register reading.**  `current_link_speed` x `current_link_width` says
   `2.5 GT/s` x1 for both cards; 26.6 GB/s host→device was measured with the destination buffer *in VRAM*, which a
   Gen1 x1 link (0.5 GB/s raw) cannot carry.  The registers are wrong on this platform, so no PCIe link figure is
   wired; the Monitor shows the reason instead.  The two cards are **not** on equal links (card1 measures exactly
   half of card0, four independent measurements), which the register reading does not show at all.
4. Every metric this platform has no source for is `null` in the payload plus a reason in
   `hardware_static.gpu_unsupported`, never `0` - a zeroed load/power/PCIe field reads as "idle" and that is a lie.

## 1. Field by field: metric -> source -> value observed

Sources are the shipped code (`_Intel` in `serve/telemetry.py`); values are from the live server
(`p8/evidence/metrics-after-restart-idle.json`, `p8/evidence/live-proof.json`), per card.

| `/metrics` field | Source (file / derivation) | card0 (0000:03:00.0) | card1 (0000:08:00.0) |
| --- | --- | --- | --- |
| `gpu_util` | fdinfo `drm-cycles-ccs` / `drm-total-cycles-ccs` over the card's clients, across the 1 s interval | 0.0 % idle, 50.1 % mid-request | 0.0 % idle, 58.0 % mid-request |
| `gpu_copy_busy` (new) | fdinfo `drm-cycles-bcs` / `drm-total-cycles-bcs` | 0.0 % idle, 44.9 % mid-request | 0.0 % idle, 46.3 % mid-request |
| `gpu_mem_used` | fdinfo `drm-total-vram0`, summed over the card's clients (one entry per client id) | 28,625,264,640 B (26.7 GiB) | 33,949,368,320 B (31.6 GiB) |
| `gpu_mem_total` | PCI BAR 2 of the driver's `resource` file (the VRAM aperture) | 34,359,738,368 B (32 GiB) | 34,359,738,368 B (32 GiB) |
| `gpu_mem_gtt` (new) | fdinfo `drm-total-gtt` (host memory mapped for the card) | 5,157,922,816 B | 5,126,373,376 B |
| `gpu_temp` | hwmon `temp2_input`, the sensor whose **label** is `pkg` | 58-68 °C (59→68 °C under load) | 55-67 °C (65→67 °C) |
| `gpu_temp_vram`, `gpu_temp_vram_max`, `gpu_temp_pcie` (new) | hwmon `temp3_input` (label `vram`), the hottest `vram_ch_*`, `temp5_input` (label `pcie`) | 60-66 °C, 60-66 °C, 61-67 °C | 54-64 °C, 54-64 °C, 57-67 °C |
| `gpu_power` | hwmon `energy1_input` (label `card`, µJ) delta across the interval / 1e6 | 47.4 W idle, 125.3 W mid-request | 44.7 W idle, 147.6 W mid-request |
| `gpu_power_limit` | hwmon `power1_cap` (230 W per card; the aggregate is the 460 W the UI shows) | 230.0 W | 230.0 W |
| `gpu_power_pkg`, `gpu_power_crit` (new) | hwmon `energy2_input` (label `pkg`), `power1_crit` | 27.2 W idle, 460.0 W crit | 26.4 W idle, 460.0 W crit |
| `gpu_fan_rpm` (new) | hwmon `fan1_input` | 0 rpm when idle, 1112→1305 rpm under load | 0 rpm when idle, 1330 rpm under load |
| `gpu_freq_mhz`, `gpu_freq_max_mhz`, `gpu_gt_idle` (new) | `tile0/gt0/freq0/act_freq`, `max_freq`, `gtidle/idle_status` | 0-400/2800 MHz, `gt-c0` idle → `gt-c0` with 2800 MHz under load | 0/2800 MHz, `gt-c6` |
| `gpu_pcie_gen`, `gpu_pcie_gen_max`, `gpu_pcie_width`, `gpu_pcie_rx_mb`, `gpu_pcie_tx_mb` | **no source** - see §3 and §4 | `null` | `null` |

Aggregation over several cards is unchanged (sum for memory/power, mean for the busy shares, hottest for
temperature); `hardware.gpus[]` now carries each card's whole reading set instead of six fields.

The `util` derivation is validated against a known workload rather than assumed: a 3.03 s copy burst inside a 147.7 s
window reads **2.06 %** of the copy engines' cycles, and the same burst on the other card 4.08 % of a 147.7 s window
(`p8/p8_util_bench.py`; the copy phase really lasted 3.03 s / 6.01 s).

## 2. Raw `/metrics`: before, in flight, after

Before the change (the running server, old code) - every gpu_ field absent, `gpu_name` null, `gpu_count` 1:
`p8/evidence/metrics-before-restart.json`.

After the restart, at rest: `p8/evidence/metrics-after-restart-idle.json` -
`"gpu_util": 0.0, "gpu_temp": 58.0, "gpu_power": 92.44, "gpu_mem_used": 62562050048, "gpu_mem_total": 68719476736`.

While a real `POST /v1/chat/completions` was in flight (300 completion tokens, 16.3 s wall; prompt 83 tokens),
`GET /metrics` polled once a second - `p8/evidence/live-proof.json`, printed in `p8/evidence/live-proof.txt`:

```
    t      state  tok_s  util  temp  power  copy     mem_used pcie_gen
  0.0       idle   None   0.0  65.0  93.26   0.0  62595833856 None
  1.0 generating   30.6 19.48  64.0 144.81 19.58  62593736704 None
  2.0 generating   26.4 54.04  66.0 272.97 45.63  62597931008 None
  3.0 generating   18.8 56.10  65.0 274.84 47.61  62597931008 None
  5.0 generating   18.6 58.33  66.0 272.70 48.91  62602125312 None
  8.0 generating   15.8 58.98  66.0 272.88 49.14  62610448384 None
 12.0 generating   18.7 57.17  67.0 272.98 49.33  62618836992 None
 16.0 generating   15.9 58.74  68.0 274.26 49.11  62593671168 None
```

and the raw sysfs behind those same samples (card0): `pkg` 59 → 68 °C, `vram` 62 → 66 °C, fan 1112 → 1305 rpm,
`energy1_input` 9260968027282 → 9262907970703 µJ, `act_freq` 0 → 2800 MHz, `gtidle` `gt-c6` → `gt-c0`, and
`current_link_speed` = `2.5 GT/s` x1 at every single sample (10 ms sampling during a transfer: `p8/p8_link_watch.py`).

## 3. The PCIe contradiction, resolved

The claim was: sysfs says `2.5 GT/s` x1 for both cards, but 26.5 GB/s host→device was measured.  What each side
actually is, measured here:

| Reading | Value | Source |
| --- | --- | --- |
| `current_link_speed` / `_width`, `max_link_speed` / `_width` | `2.5 GT/s`, `1` - for **both** GPUs, and for the switch's downstream ports to the GPUs, the GPU audio functions *and* (wrongly) the NVMe's downstream port | `/sys/bus/pci/devices/<bdf>/current_link_*`, `max_link_*` (`p8/evidence/pcie-registers.txt`) |
| `lspci -vv -s 03:00.0 \| grep LnkCap\|LnkSta` | **cannot be run here**: the card's own request - `Capabilities: <access denied>` as a user, and the config space past `0x40` reads as zeros, so the raw `LnkCap`/`LnkSta` registers cannot be decoded without root | `p8/p8_pcie_probe.py` output in `p8/evidence/pcie-registers.txt` |
| the switch's uplink and both root ports | `32.0 GT/s` x8 (`0000:00:06.0`, `0000:00:06.3`, `0000:01:00.0`, `0000:06:00.0`) | same |
| the NVMe endpoint behind the same switch | `16.0 GT/s` x4 - a sane per-device reading, next to the bogus `2.5 GT/s` x1 of its own downstream port | `0000:05:00.0` |
| **measured** host→device, destination in VRAM (fdinfo proves it) | **26.63 GB/s** (card0) | `p8/p8_h2d_bench.cpp`, `p8/p8_pcie_account.py` |
| **measured** host→device, destination in VRAM | **13.29 GB/s** (card1) | same |
| the engine's own startup probe | 26.5 GB/s primary / 13.4 GB/s second card | `/home/michael/strata-xpu/logs/i1-iq3s-engine.log:3,9` |
| P4's copy-kernel probe (shipping `copy_kernel`, one B70) | 25.5-25.8 GB/s at the shipped shape, 26.7 GB/s for 512 B..1 MiB chunks | `p4/STATUS-P4.md` §0.2, §0.5 |

**Verdict: the measurement is true and the register reading is false.**  0.5 GB/s is the entire raw capacity of a
Gen1 x1 link; the slowest thing measured here is 13.29 GB/s with the destination buffer demonstrably in VRAM
(fdinfo `drm-total-vram0` = 278,312 KiB on 0000:08:00.0 during exactly that transfer), i.e. 26x more than that link
can carry.  The upstream end of the same switch tree reads `32 GT/s` x8, which is what an 83 % - efficient
26.6 GB/s payload needs.  So the driver's per-card link registers cannot be used, and the Monitor does not show them.

**And the two cards are not equal on this box**: every measurement that can see both gives card1 exactly half of
card0 - 26.5/13.4 (engine), 26.52/13.38 (P8 SYCL memcpy, engine running), 26.14/13.29 (P8, engine stopped).  A 2:1
ratio is half the lanes (Gen5 x4) or one speed step down (Gen4 x8); the register reading shows neither difference
(it claims the identical Gen1 x1 for both).  The precise per-card gen/width cannot be defended from here (that would
need `lspci -vv` as root), so the payload leaves it `null` with the reason instead of guessing.

## 4. Metrics marked unavailable on this backend, and why

| Field | Why there is no source |
| --- | --- |
| `gpu_pcie_gen`, `gpu_pcie_gen_max`, `gpu_pcie_width` | the xe driver's link-speed registers report a link state the measured host→device bandwidth refutes (§3), so no link figure is shown |
| `gpu_pcie_rx_mb`, `gpu_pcie_tx_mb` | no PCIe byte counter in sysfs on this driver (NVML has `nvmlDeviceGetPcieThroughput`; there is no equivalent here) |
| (not a field, but the reason the PCIe card is a `-`) | `gpu_unsupported` in the payload carries these lines, and `serve/web/app.js` prints the reason where the number would be |

Nothing else is missing: load, VRAM, GTT, temperature (3 sensors + the hottest VRAM channel), fan, power, power cap,
power crit, clock and GT idle state are all real readings from the driver's own files.  What this platform cannot
give - `xpu-smi`, `intel_gpu_top` and `perf` are all absent or unusable (`perf_event_paranoid=4`) - is not needed:
the backend is sysfs + fdinfo only, and no package was installed (as the card required).

## 5. The server and the dashboard

- `bash p6/p6_stop.sh` was run only in an idle window (polled `live.state` until it left `generating`; the operator's
  generation had completed) and `bash p6/p6_start.sh` restarted it: **server pid 1040401, engine pid 1040410,
  127.0.0.1:8099, READY at 30 s** (VERSION 0.1.34, `expert_slots 24576`, `vram_free_mib 5010` - the same running
  state P6 left).
- `/` (the app's Monitor tab) read from the live page over the DevTools protocol, after one real request:
  `GPU load 0 % (GPU 0 0% · GPU 1 0%)`, `VRAM 58.3 / 64 GB (GPU 0 26.7 GB · GPU 1 31.6 GB)`, `GPU temp 67 °C
  (GPU 0 58° · GPU 1 67°)`, `Power 93 W of 460 W limit`, `PCIe - ` with the reason underneath, `CPU 1 %`,
  `Disk read 0.0 MB/s`, `Speed 18.7 t/s (last request)`; the GPU load and Power sparklines carry the request's bump.
  About > hardware: `GPU Battlemage G31 [Intel Graphics] + Battlemage G31 [Intel Graphics], 64 GB`,
  `CPU Intel(R) Core(TM) Ultra 7 265KF, 20 threads`, `RAM 123 GB`.
- `/api-monitor` loads (`Strata API Monitor`) and lists the requests made here (`11:32:07 POST /v1/chat/completions
  ... 16.25 s · HTTP 200`).
- Tests: `python3 serve/test_server.py` = **92 tests OK** (including the new `IntelTelemetry`), `python3 -m
  serve.test_monitor` = 7 OK.  The new tests run against a fake xe sysfs tree and a fake `/proc` (no GPU needed) and
  pin the label-chosen sensor, the energy-derived power, the deduplicated VRAM sum, the cycle-counter load and the
  None-not-0 rule.

## 6. Repro

```
bash p8/p8_account_run.sh 256 20          # H2D on both cards + where the buffers landed + link state during it
/usr/bin/python3 p8/p8_util_bench.py 256 300   # the busy derivation against a known workload
/usr/bin/python3 p8/p8_backend_probe.py 3 1.2  # every field the backend produces, per card, with its sources
/usr/bin/python3 p8/p8_live_proof.py 300       # one real request with /metrics sampled while it runs
/usr/bin/python3 p8/p8_pcie_probe.py           # the link registers (and what a user process can read of them)
```

Residual unknown, for the operator: a root-side `lspci -vv -s 03:00.0` (or `08:00.0`) would say which register is
lying and give the true per-card gen/width.  It is not needed for the dashboard - and it cannot be run from this
session (no sudo), which is why §3 rests on the measurements instead.
