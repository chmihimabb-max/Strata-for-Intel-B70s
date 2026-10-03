# P6 — the resident Strata server + dashboard on 8099 (card t_5dfc11a3)

**State: RUNNING.** The model of record is served resident on both B70s, with the dashboard, at
`http://127.0.0.1:8099/` (app + Monitor tab) and `http://127.0.0.1:8099/api-monitor` (the standalone dash).
Server pid **1021507**, its engine child pid **1021510**, `--max-context 262144` with KV streaming.

**While it runs it holds BOTH GPUs. Any other GPU card must stop it first (`strata/p6/p6_stop.sh`).**

Everything below is raw output from this run; the lane's files are in `strata/p6/` (scripts, requests,
responses, resource CSV, screenshots, the pre-P6 config).

---

## 0. What was started, and the one thing that had to be put back

```
cd /home/michael/strata-xpu/strata && ZE_AFFINITY_MASK=<unset> \
  /usr/bin/python3 -m serve.server --engine strata --config strata-sycl-iq3s.json --port 8099 --api-monitor
```

(bash wrapper: `strata/p6/p6_start.sh`, which unsets `ZE_AFFINITY_MASK`, sources oneAPI's setvars and exports
`SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/home/michael/strata-xpu/sycl-cache/m6c` — the warm program cache every
measurement used.)

Two honest deviations from the card's written recipe, both stated here rather than shipped silently:

1. **`--max-context`.** The card says the config of record carries `--max-context 262144` with KV streaming; the
   file on disk carried **`--max-context 4096` and no `--kv-resident`** — the value P7's 4K serve arm left behind
   (its arm scripts pass their own `--max-context`/`--kv-resident`, so the file was never the arm's source).
   4096 is not the 256K deliverable, so the config of record was **restored to the M6c 256K configuration**:
   `--max-context 262144` and `--kv-resident 32768` (`m6c/m6c_serve.sh:22,114-116` and every row of
   `m6c/TABLE.md`). No engine change of any kind: `git diff` on the engine is empty, the binary is P7's
   (`build-sycl/strata`, 2026-10-03 08:14:31, 35,091,568 B). The pre-P6 file is kept byte-identical at
   `strata/p6/config-before-p6.json` (md5 `2fe9a4b711fbd704abc95958166f74fc`); both files' notes record the swap.
   The engine's own line proves what is running: `KV streaming: 32768 of 262144 cells per QSA layer in VRAM,
   the K/V in 1.29 GiB of pinned RAM`.
2. **`--api-monitor`** (a server flag, no engine effect) is **required for the dash**: `monitor.js:75` polls
   `/api/requests`, which the server only serves when `api_monitor` is on (`serve/server.py:1879`); without it the
   page shows "Disconnected · retrying" and its speed field stays `—`. It keeps the last 100 requests' prompts and
   answers **in memory only** (`retention 100, persistent false`), and the server warns about that when no API key
   is set — which is the case here, on loopback only.

**The dash is not at `/monitor.html`.** The card says both pages are at `/` and `/monitor.html`; the file is
`serve/web/monitor.html` but the HTTP route is **`/api-monitor`** (`serve/server.py:1897-1898`: `path == ""` ->
`index.html`, `path == "/api-monitor"` -> `monitor.html`). `GET /monitor.html` answers
`{"error": {"message": "not found"}}` (measured). The in-app route is the header's **Monitor** tab.

### Startup lines (server log `~/strata-xpu/logs/p6-server.log`)

```
P6 server start 2026-10-03T10:24:47-06:00   pid 1021214   port 8099
engine argv: build-sycl/strata --pack /run/media/.../strata-iq3s/pack --native .../IQ3_S-00001-of-00002.gguf
  --ple-gguf .../IQ3_S-00002-of-00002.gguf --mtp /home/michael/strata-xpu/mtp/rt --kv int8 --expert-cache auto
  --expert-profile /home/michael/strata-xpu/strata/data/expert-profile.bin --mmap-experts --prefill 512 --spec 4
  --spec-min-p 0.5 --max-context 262144 --kv-resident 32768 --no-capture --stats --layer-split auto
loading the model (the first start takes a minute or two) ...
[strata] layer split across GPUs [0, 1] (auto)
[strata] filling the GPU's expert cache (11776 experts, 21.20 GiB of VRAM) ...
[strata] still starting (26 s) - please wait ...
[strata] almost ready ...
[strata] API request monitor on (/api-monitor): the last 100 requests' prompts and answers are kept in memory;
         anyone who can reach this server can read them (no API key)
ready: http://127.0.0.1:8099/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, context 262144 tokens)
       open http://127.0.0.1:8099/ in a browser to chat; close this window to stop the model
```

**LAN line.** It is only printed when the server is *bound* beyond loopback
(`serve/server.py:2589-2595`), and AGENTS.md forbids exposing the server without an API key. The resident server
is on `127.0.0.1` (the config has no `"host"`), so it printed the loopback lines only. What the LAN line looks
like was captured with a throwaway `--lazy --host 0.0.0.0 --api-key` probe on port 8098 (no engine spawned, no
GPU touched, stopped straight after — `strata/p6/lan-probe.log`):

```
ready: http://127.0.0.1:8098/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, context 262144 tokens, API key required)
       open http://127.0.0.1:8098/ in a browser to chat; close this window to stop the model
       from other devices: http://10.0.0.181:8098/   (API: http://10.0.0.181:8098/v1)
```

To offer the resident model on the LAN, add the two flags and restart (and give the apps the key):

```
python -m serve.server --engine strata --config strata-sycl-iq3s.json --port 8099 --api-monitor \
       --host 0.0.0.0 --api-key "<a key you choose>"     # then: from other devices: http://10.0.0.181:8099/
```

### Engine's own load-time lines (`~/strata-xpu/logs/i1-iq3s-engine.log`, appended from line 67)

```
strata generate: layer split across 2 GPUs: CUDA0, then CUDA1 (split auto)
strata generate: 300 native projection matrices, 2018.88 MiB of weights
strata generate: layer split auto: K=23 - predicted 7.6 ms per decode window; the caches hold 24576 of 24576 profiled pairs (~100.0% of the routed mass)
strata generate: KV streaming: 32768 of 262144 cells per QSA layer in VRAM, the K/V in 1.29 GiB of pinned RAM
strata generate: PLE on, table 320001536 rows of ...IQ3_S-00002-of-00002.gguf
strata mtp: draft layer loaded, 835 MiB of VRAM (experts 675, dense 111), files read in 0.41 s (1900 MiB/s)
strata generate: expert cache auto: 27.19 GiB free, 700 MiB reserved (+0 MiB for the draft head) -> 10689 slots
strata generate: expert cache 11776 slots, 21.20 GiB of VRAM; policy is PROFILE, ranked by routing frequency, no eviction.
```

## 1. A served conversation, not a listening port

### `GET /v1/models` (and `/health`, `/slots`)

```
{"object": "list", "data": [{"id": "qwen3.8-flash-next-iq3s", "object": "model", "status": {"value": "loaded"},
 "meta": {"n_ctx": 262144}, "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}}]}

{"status": "ok", "max_context": 262144, "model": "qwen3.8-flash-next-iq3s", "images": false, "api_key": false,
 "loaded": true, "service": "strata"}

[{"id": 0, "n_ctx": 262144, "is_processing": false}]
```

### Request A — thinking OFF (`reasoning_effort: "none"`, non-stream; `strata/p6/req-a-thinking-off.json`)

```
POST /v1/chat/completions   HTTP 200   1.930624 s   690 bytes
{"id": "chatcmpl-72c126d72a2c465e8d251c5a", "object": "chat.completion", "created": 1791044742,
 "model": "qwen3.8-flash-next-iq3s",
 "choices": [{"index": 0, "message": {"role": "assistant",
   "content": "The capital of France is Paris, and the Seine River runs through it."}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48, "prompt_tokens_details": {"cached_tokens": 0}},
 "timings": {"cache_n": 0, "prompt_n": 32, "prompt_ms": 1166.4, "prompt_per_second": 27.4,
             "predicted_n": 16, "predicted_ms": 759.9, "predicted_per_second": 21.1,
             "draft_n": 15, "draft_n_accepted": 12}}
```

The engine's own stats lines for that request:

```
strata decode timing: 6 windows, avg T 3.50, 2.67 tokens/window, 126.66 ms/window = verify 114.03 (GPU-reach wait 43.21 + per-layer host 0.50 [plan 0.30 actq 0.23 jobs 0.25 CPU 0.00] + stage 1.67) + commit/emit 2.04 + draft 10.59; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 35.00, PCIe 0.00
strata serve: prompt 32 tokens = 0 reused + 32 read in 1166 ms (27.4 tok/s), 16 generated in 760 ms (21.1 tok/s), drafts accepted 12 of 15, 1 checkpoints
strata serve: decode expert cache hit rate: 100.0% (10080 hits / 10080 lookups)
strata serve: expert tiers: GPU 10080 hits this request; since the start RAM 0 blobs, files 0 blobs 50296.6 MB read (the GGUF in place)
strata serve: KV streaming: 96.37% of 1790 block reads hit VRAM, 0.3 MiB read from RAM
```

### Request B — thinking ON (`reasoning_effort: "medium"`, `stream: true` — exactly the web app's shape, `req-b-thinking-on.json`)

```
POST /v1/chat/completions   HTTP 200   10.116195 s   38017 bytes   (SSE)
data: {... "delta": {"role": "assistant", "content": ""} ...}
: keep-alive
data: {... "delta": {"reasoning_content": "The"} ...}
data: {... "delta": {"reasoning_content": " user"} ...}
data: {... "delta": {"reasoning_content": " asks"} ...}
...
data: {... "delta": {"content": " the"}, "finish_reason": "length"}, "usage": {"prompt_tokens": 32, "completion_tokens": 160, "total_tokens": 192},
      "timings": {"prompt_per_second": 30.6, "predicted_per_second": 17.6, "draft_n": 172, "draft_n_accepted": 85}}
data: [DONE]
```

```
strata decode timing: 75 windows, avg T 3.29, 2.13 tokens/window, 120.91 ms/window = verify 107.58 (GPU-reach wait 40.74 + per-layer host 0.47 [plan 0.29 actq 0.22 jobs 0.23 CPU 0.00] + stage 1.81) + commit/emit 1.90 + draft 11.43; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 32.93, PCIe 0.00
strata serve: prompt 32 tokens = 0 reused + 32 read in 1044 ms (30.6 tok/s), 160 generated in 9068 ms (17.6 tok/s), drafts accepted 85 of 172, 1 checkpoints
strata serve: decode expert cache hit rate: 100.0% (118560 hits / 118560 lookups)
strata serve: expert tiers: GPU 118560 hits this request; since the start RAM 0 blobs, files 0 blobs 50296.6 MB read (the GGUF in place)
strata serve: KV streaming: 99.35% of 37545 block reads hit VRAM, 1.0 MiB read from RAM
```

So the chat template renders, thinking is off when asked off (no `reasoning_content` frames at all) and on when
asked on (1,189-char reasoning block on request D), and the drafter is live: 12/15, 85/172 and 201/295 drafts
accepted.

### Request C — the third request that makes the dash's speed field move

```
HTTP 200  1.489532 s   "2, 3, 5, 7, 11" (finish_reason stop, 26 prompt + 15 generated)
timings: prompt_per_second 29.8, predicted_per_second 24.4, draft_n 12, draft_n_accepted 12
strata serve: prompt 26 tokens = 0 reused + 26 read in 872 ms (29.8 tok/s), 15 generated in 614 ms (24.4 tok/s), drafts accepted 12 of 12, 1 checkpoints
strata serve: decode expert cache hit rate: 100.0% (8160 hits / 8160 lookups)
strata serve: KV streaming: 95.45% of 1210 block reads hit VRAM, 0.2 MiB read from RAM
```

### Request D — 320 tokens, thinking on (the load the resource sampler ran under)

```
HTTP 200  16.478097 s   320 generated at 21.3 tok/s (finish_reason length), reasoning 1189 chars
content: "In a standard transformer layer, every token passes through the same feed-forward network (FFN)-a pair of large matrix multiplications with a nonlinearity ..."
usage: {"prompt_tokens": 45, "completion_tokens": 320, "total_tokens": 365}
timings: {"prompt_per_second": 31.3, "predicted_per_second": 21.3, "draft_n": 295, "draft_n_accepted": 201}
strata serve: prompt 45 tokens = 0 reused + 45 read in 1436 ms (31.3 tok/s), 320 generated in 15038 ms (21.3 tok/s), drafts accepted 201 of 295, 1 checkpoints
strata serve: decode expert cache hit rate: 100.0% (199680 hits / 199680 lookups)
strata serve: KV streaming: 99.57% of 106980 block reads hit VRAM, 1.9 MiB read from RAM
strata serve: suffix drafts: 1 windows, 1 of 3 drafts accepted
```

Decode sits at **17.6–24.4 tok/s** at 32-45-token prompts on this pair — the M6c reference for the 262K
streaming configuration (18.45 tok/s at 259,943 prompt tokens). Expert cache hit rate is **100.0%** on every
request and the expert tier line reports **0 CPU entries**; the KV tier reads 0.2-1.9 MiB per request from RAM.

## 2. The dash renders — and its speed field moves

| what | where | value |
|---|---|---|
| `serve/web/index.html` (app UI) | `/` | title **Strata**, tabs Chat / **Monitor** / About |
| `serve/web/monitor.html` (standalone dash) | **`/api-monitor`** | title **Strata API Monitor** |

Standalone dash (`/api-monitor`), before and after a new request — same page, speed field moved
**17.6 -> 24.4 tok/s**, requests retained 2 -> 3:

```
10:26:13 AM   MODEL Loaded qwen3.8-flash-next-iq3s | REQUESTS RETAINED 2 | LAST WALL-CLOCK 10.12 s | LAST DECODE SPEED 17.6
10:26:32 AM   MODEL Loaded qwen3.8-flash-next-iq3s | REQUESTS RETAINED 3 | LAST WALL-CLOCK  1.49 s | LAST DECODE SPEED 24.4
```

Screenshots: `strata/p6/monitor-1-speed-17.6.png`, `strata/p6/monitor-2-speed-24.4.png`.

The app's own Monitor tab (same data, plus the per-request table) — `strata/p6/monitor-3-app-monitor-tab.png`:

```
Model state: Idle | last: 15 tokens at 24.4 tok/s
Speed: 24.4 t/s decode last request, 30 t/s prefill last request
VRAM: 24,576 experts cached      Context fill: 0% (41 / 256K)
Experts in VRAM: 24,576 · 46.8 GB      System RAM: 13.0 / 123 GB
Recent requests: 26/0/15 @24.4 100.0% 1.5s | 32/0/160 @17.6 100.0% 10.1s | 32/0/16 @21.1 100.0% 1.9s
Since Sat 10:25 AM: 3 requests · 90 prompt tokens read at 29 tok/s (0 reused) · 191 written at 18.3 tok/s
```

(GPU load / VRAM / temp / power read `–` in the app tab: `xpu-smi` and `intel_gpu_top` are not installed on this
host, the known BRIEF §3b limitation — the expert count and RAM figures come from the engine and /proc.)

## 3. What it costs while it runs

`m6c/m6c_sample.py` on the server's own process tree (server + engine child), 99 samples over 99 s spanning
requests C and D (`strata/p6/p6-resource.csv`; `p6_peaks.py`):

```
peak tree RSS      : 48.57 GiB  (engine alone 48.38 GiB)
peak VRAM card0    : 26808.6 MiB  (driver fdinfo, process tree only)
peak VRAM card1    : 32399.2 MiB  of 32,655 MiB = 99.2% full
last sample        : RSS 48.57 GiB, VRAM 26808.6 + 32399.2 MiB, mlocked 0.00 GiB, read_bytes 0.05 GiB
```

VRAM was **flat on all 99 samples** (the caches are allocated at load: 11,776 expert slots = 21.20 GiB on
card0, card1 holds its layers, the head and the MTP draft layer), so those are also the peaks. Adding the
desktop's clients (802 MiB on card0) the driver totals **card0 27,605 MiB, card1 32,399 MiB** of 32,655 MiB.
The 48.6 GiB tree RSS is almost all *file-backed*: `/proc/1021510/status` reads `VmRSS 50,726,984 kB` with
`RssAnon 1,236,648 kB` (1.18 GiB) and `RssFile 49,490,336 kB` (47.20 GiB — the two mmap'd GGUF shards and the
pack on the external volume), which is why the machine reports `used 12 GiB` with `buff/cache 90 GiB` and
`available 110 GiB` of 123 GiB while the server is up. The engine's own line for the KV pool:
`32768 of 262144 cells per QSA layer in VRAM, the K/V in 1.29 GiB of pinned RAM` — that figure is the engine's
(stage 0); `/proc/meminfo` `Mlocked` reads 16 kB, so it is not a kernel `mlock`.

**Context served: 262,144 tokens** (the 256K deliverable, KV streaming with 32,768 VRAM-resident cells per QSA
layer). The cost of that choice is prefill on long prompts: M6c measured a *fresh* 259,943-token prompt at
**316.9 tok/s, TTFT 820.5 s (~14 min)** in exactly this configuration (M6c also measured 390.7 tok/s / 665.5 s
with streaming off at the same length). Short prompts are unaffected — request D's 45-token prompt prefilled in
1.4 s. For interactive use with no 256K depth needed, the measured 32K arm (`m6c-32k-auto`: 344.8 tok/s prefill,
21.71 tok/s decode, peak VRAM 29,438+31,970 MiB, peak RSS 48.31 GiB) is written out as an optional,
clearly-labelled variant, **not** as what is running:

```
strata/p6/config-32768-snappy.json      # --max-context 32768 --kv-resident 0 --prefill auto
python -m serve.server --engine strata --config p6/config-32768-snappy.json --port 8099 --api-monitor
```

Both files live on this machine only (`strata-sycl-iq3s.json` is gitignored, `.gitignore:37`); the variant is
generated from the config of record by `p6/p6_make_snappy.py`.

## 4. Leaving it usable

```
# status
curl -s http://127.0.0.1:8099/health ; curl -s http://127.0.0.1:8099/v1/status
ss -ltnp | grep 8099 ; pgrep -a -x strata

# relaunch (after it has been stopped: takes ~60 s to load and fill the expert caches)
bash /home/michael/strata-xpu/strata/p6/p6_start.sh
#   == cd /home/michael/strata-xpu/strata && ZE_AFFINITY_MASK=<unset> bash -c \
#      'source /opt/intel/oneapi/setvars.sh; SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=.../sycl-cache/m6c \
#       /usr/bin/python3 -m serve.server --engine strata --config strata-sycl-iq3s.json --port 8099 --api-monitor'

# STOP (it holds BOTH GPUs - run this before any other GPU work; ~10 s, waits for the engine to free VRAM)
bash /home/michael/strata-xpu/strata/p6/p6_stop.sh
```

| | |
|---|---|
| URL | `http://127.0.0.1:8099/` (app, Monitor tab) · `http://127.0.0.1:8099/api-monitor` (dash) · `/v1` (OpenAI + Anthropic) |
| server pid | **1021507** (engine child **1021510**); pid file `strata/p6/server.pid` |
| server log | `/home/michael/strata-xpu/logs/p6-server.log` (stdout/stderr of the server, incl. `ready:` and the LAN lines) |
| engine log | `/home/michael/strata-xpu/logs/i1-iq3s-engine.log` (**appended**: the engine's stderr — split plan, caches, `--stats`, the per-request stats lines above; P6's block starts at line 67) |
| stop | `bash /home/michael/strata-xpu/strata/p6/p6_stop.sh` → SIGTERM (the server's Ctrl+C path, QUIT to the engine) |
| holds | **both B70s** (26.8 + 32.4 GiB VRAM), 48.6 GiB RSS. One engine at a time (PLAN 9 rule 1): stop it before any other GPU run |

## 5. MCP (optional, deliverable 5) — available, not installed

`tools/strata_mcp.py` exists and answers, stdio transport, root = this repo:

```
$ /usr/bin/python3 tools/strata_mcp.py --help
usage: strata_mcp.py [-h] [--root ROOT]
Strata's MCP server (stdio): install, start, check and stop Strata from an AI assistant. See docs/MCP_SERVER.md.
claude mcp add strata -- python3 /home/michael/strata-xpu/strata/tools/strata_mcp.py     # docs/MCP_SERVER.md:32
```

Nothing was registered with any agent's config (not asked for); the HTTP API on 8099 is the documented path for
coding agents (`/v1/chat/completions`, `/v1/messages`, `/v1/models`).
