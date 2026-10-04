# S4 (t_30d9ccfb) — the served path fails a 32K request on the two-GPU split

**Status: reproduced, diagnosed, fixed on the branch (`e41fa26`), and re-proved through the server.** The failure is a
cross-card copy of a layer split's *mid-prompt checkpoint part*; the fix makes that copy run on the device that owns
the state. The config of record is untouched — the fix is the copy, not the config — so the two-GPU split, the resident
server and the checkpoints all stay.

Rig and raw output: `/home/michael/strata-xpu/strata/s4/` (the arms' raw files) and
`/home/michael/strata-xpu/strata/s4/S4-EVIDENCE.txt` (this write-up's raw output, assembled from those files by
`s4/s4_evidence.sh`). Branch `sycl-xpu`. Two engines' worth of binaries: the arm logs record the md5 of every binary
used — the **pre-fix** binary is `72ec74aeefd9f300335d40f038910afa` (D1's HEAD, kept at `d1/strata-after-histfix`),
the **fixed** binary is `e79b2ad632d37d32786228b660d35b29` (kept at `/home/michael/strata-xpu/s4/bin/strata-s4-fix`).

The arms all go through `serve.server` (never the engine's stdin protocol): one resident server per arm, started
exactly as `strata-sycl-iq3s.json` reads (config of record **verbatim**; only `log`, `port` and `model_name` are
per-arm), and one real `POST /v1/chat/completions` per length. `s4_cfg.py` prints the server's effective engine argv
(imported from `serve.server.engine_args`, so the `--layer-split auto` the server appends is in it) and its sha256.

| arm | binary | config | prompt tokens | result | wall |
|---|---|---|---|---|---|
| `s4-prefix-sweep` | pre-fix | of record (checkpoints ON) | 4,096 | **pass** | 22.3 s |
| | | | 16,052 | **pass** | 75.6 s |
| | | | 16,352 | **pass** | 77.3 s |
| | | | 16,452 | **FAIL** (HTTP 400, engine exit 1) | 71.0 s |
| `s4-prefix-32k` | pre-fix | of record | 32,277 | **FAIL** (HTTP 400, engine exit 1) | 71.3 s |
| `s4-pc0-32k` | pre-fix | of record + `--prompt-cache 0 --prompt-cache-every 0` | 32,277 | pass | 162.0 s |
| `s4-fix-32k` | fixed | of record | 32,277 | **pass** | 160.3 s |
| `s4-fix-32k-2` | fixed | of record (repeat) | 32,277 | pass | 160.1 s |
| `s4-pc0-32k-2` | fixed | + `--prompt-cache 0 --prompt-cache-every 0` (repeat) | 32,277 | pass | 162.1 s |
| `s4-fix-16400` | fixed | of record | 16,452 | **pass** (used to fail) | 78.3 s |
| `s4-ckpt-root` | fixed | + `--prompt-cache 6 --prompt-cache-every 0` | 32,277 | pass | 160.2 s |
| `s4-oldckpt-32k` | pre-fix | + `--prompt-cache 6 --prompt-cache-every 0` | 32,277 | pass | 160.1 s |
| `s4-ckpt-4k` | fixed | of record | 4,148 | pass | 22.2 s |
| `s4-pc0-4k` | fixed | + `--prompt-cache 0 --prompt-cache-every 0` | 4,148 | pass | 22.3 s |
| `s4-closure-ckpt-4k` | fixed | of record, `STRATA_SYCL_GRAPH=0` | 4,148 | pass | 22.3 s |
| `s4-closure-pc0-4k` | fixed | + `--prompt-cache 0 --prompt-cache-every 0`, `STRATA_SYCL_GRAPH=0` | 4,148 | pass | 22.2 s |

## 1. The reproduction (the card's finding), raw

`s4-prefix-32k`, one resident server on both B70s (`ZE_AFFINITY_MASK` unset in the shell, `gpu: [0,1]` → the server
exports `ZE_AFFINITY_MASK=0,1` and appends `--layer-split auto`), the config of record verbatim, one 32,277-token
`/v1/chat/completions`:

```
[strata] the engine reported an error: saving a checkpoint part failed
[strata] done: 0 tokens in 71 s (0.0 tok/s) (error, cancel=False)
```
```
HTTP 400: b'{"error": {"type": "invalid_request_error", "message": "saving a checkpoint part failed"}}'
```
and the engine's own stderr, the last three lines it wrote before exiting 1:

```
strata/sycl: memcpy failed: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
strata serve: checkpoint save: conversation snapshot running-state copy: invalid argument
strata serve: saving a checkpoint part failed
```

**It is deterministic and it is not "32K only".** Both failing requests died at the same place, ~71 s in, at prompt
token 16,384, with the same three lines; the four lengths in the sweep bracket it:

- 4,096 / 16,052 / 16,352 prompt tokens: **pass**, in one server lifetime, in that order.
- 16,452 prompt tokens: **fail**, and the server's last progress line is `reading the prompt: 15,872 of 16,452
  tokens, 71 s so far` — i.e. it died on the chunk that ends at **16,384**.

16,384 is the engine's own `--prompt-cache-every` default (`src/program/generate.cpp:396`), the cadence at which the
serve loop takes a **mid-prompt conversation checkpoint** (`pp_next_check = resume + o.prompt_cache_every`,
`generate.cpp:5056`, and `part_next` filled with it at `:5060`). P6's four served requests passed because they were
26, 32, 32 and 45 tokens — nothing near 16,384. P10's driver arms passed 32,256-token prompts because every one of
them passes `--prompt-cache 0 --prompt-cache-every 0` (`p10/p10_run_arm.sh:63`, `d1/d1_run_arm.sh:84`) and so never
takes a checkpoint at all. The card's prime suspect is confirmed, but the interesting half of it is the other one:
**the checkpoint is not the defect — the copy is.**

## 2. The two paths, diffed

| | P10/D1 driver arms (pass at 32K) | the config of record through `serve.server` (fail at 16,384) |
|---|---|---|
| engine | `build-sycl/strata` | same |
| pack / native / PLE / drafter / `--kv int8 --expert-cache auto --mmap-experts --prefill 512 --spec 4 --spec-min-p 0.5` | same | same |
| `--max-context` / `--kv-resident` | 32,768 (the driver arms) / 32,768 | 262,144 / 32,768 |
| layer split | `--layer-split auto` | `serve/server.py:628-629` appends `--layer-split auto` for `gpu: [0,1]` |
| device visibility | `ZE_AFFINITY_MASK` unset (split) | `serve/server.py:669` exports `ZE_AFFINITY_MASK=0,1` — the same two devices, no pinning |
| **conversation checkpoints** | **`--prompt-cache 0 --prompt-cache-every 0`** | **absent → the engine's defaults `prompt_cache = 6` (`generate.cpp:391`), `prompt_cache_every = 16384` (`:396`)** |
| `STRATA_DECODE_TIMING` | exported by the rig | in the config's `env` |

`--conversation-cache-mib` is 0 in both, so *parking* is off in both and the engine's parking-vs-split refusal
(`generate.cpp:1269-1271`) is not reached; the checkpoint machinery this card is about is a different thing that
lives in the session's own host RAM (`generate.cpp:4296` shows parking's allocation is `prompt_cache > 0 ?
conversation_cache_mib : 0`, i.e. zero here).

So the difference that matters is one flag pair. Removing it is a workaround, not the fix — the flags are how the
server reuses a chat's prefix between turns, and the served path is the delivered artifact:

| arm | checkpoints | 32,277-token request |
|---|---|---|
| `s4-prefix-32k` (pre-fix binary) | as the config of record reads | **fails** at token 16,384 |
| `s4-pc0-32k` (pre-fix binary) | `--prompt-cache 0 --prompt-cache-every 0` | passes (162.0 s, prefill 217.5 tok/s, decode 18.9 tok/s) |

## 3. The cause, named

`src/program/generate.cpp:4461-4472` (pre-fix numbering) is the mid-prompt callback a layer split uses to checkpoint
each stage's own part:

```cpp
stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
    if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
    part_next[i] = done + o.prompt_cache_every;
    ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
    part.ids.assign(cur.begin(), cur.begin() + done);
    if (!checkpoint_save(part, stage_ss, g)) { e = "saving a checkpoint part failed"; return false; }   // <-- line 4466
```

`stage_ss` is that stage's session: stage 1's running state is on **card 1**. The copy is

```
generate.cpp:4466  checkpoint_save(part, stage_ss, g)
generate.cpp:868   -> strata::core::conversation_checkpoint_save(c, ss, g)
conversation_state.cpp:35  cudaMemcpy(dst /*host*/, src /*the stage's device state*/, bytes, cudaMemcpyDefault)
cuda_runtime.h:946         -> memcpy_impl(..., &default_queue(), wait=true)          the queue of the CURRENT device
cuda_runtime.h:917-925     !du && su -> malloc_host + q->memcpy(tmp, src, bytes)     a D2H copy on that queue
```

and on this pair the two cards share **one** SYCL context (`cuda_runtime.h:344-352`: "the shared one when the driver
took it (the multi-GPU case)"), so `is_usm_pointer(src)` (`:841-843`) is true for card 1's pointer and the copy is
issued on card 0's queue — a **cross-device** transfer, which this pair has no peer path for (S1 measured a
device0 → device1 copy failing with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`). It fails the same way here, and the
error is what the two lines above the "part" error report: the shim's `memcpy` catch prints the driver's error
(`cuda_runtime.h:929`), and `conversation_state.cpp:37` builds "conversation snapshot running-state copy: …" from the
status it got back.

Why is the current device card 0 at that moment? Because a split's later stages run on a **helper thread**: prefill
hands the next chunk to the next stage through `std::async` (`src/prefill/prefill.cpp:2054`) and each stage calls its
own `on_stage_chunk` (`prefill.cpp:2051` for a stage with a successor, `:2080` for the last one). The shim's current
device is process-global, not per-thread (`cudaSetDevice` -> `backend().current`, `cuda_runtime.h:1192-1200`), and the
helper thread never sets it. Every other device-touching step in the split brackets itself with
`strata::core::OnDevice`; this callback did not — while the turn-boundary save 60 lines above does exactly that for
exactly this copy:

```cpp
generate.cpp:4397-4402   for (auto& st : stages) {          // a layer split's later stages: their sessions' part
                             const strata::core::OnDevice on(st->dev);
                             ...
                             checkpoint_save(part, st->ss, g)
```

That asymmetry is the defect: the same save, done correctly in one place and not in the other. The class behind it —
"a copy on a split must be issued by the device that owns the pointer, and the shim's current device is process-global
while the stages' chunks run on two threads" — is worth remembering; the targeted fix is in the next section.

## 4. The fix (commit `e41fa26`, 9 lines, one file)

```cpp
+                    const strata::core::OnDevice on(i == 0 ? 0 : stages[i - 1]->dev);
                     ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
```

`i == 0 ? 0 : stages[i - 1]->dev` is the split's own stage-device convention, the same expression as
`generate.cpp:2339` and `:4071`. The commented rationale is in the source; the claim it makes is the one this card
measured (16,452-token request, the 16,384-token checkpoint).

Nothing else changed: no config, no flag default, no shim, no verify/heartbeat code. `ZE_AFFINITY_MASK` stays unset
for the split, and the two-GPU configuration stays the configuration of record (PLAN §11 U11) — the fix is precisely
what makes it survive a checkpoint.

## 5. Proof through the server (not the driver)

`s4-fix-32k`: the fixed binary, `--config strata-sycl-iq3s.json` **verbatim** (checkpoints ON), one real request at the
length that used to fail, `POST /v1/chat/completions`, `max_tokens 256`, `temperature 0`:

```
HTTP 200, finish_reason length, prompt_tokens 32277, completion_tokens 256
strata serve: prompt 32277 tokens = 0 reused + 32277 read in 148685 ms (217.1 tok/s), 256 generated in 11507 ms (22.2 tok/s), drafts accepted 158 of 203, 2 checkpoints
strata decode timing: 99 windows, avg T 3.05, 2.59 tokens/window, 116.23 ms/window = verify 102.21 (GPU-reach wait 87.70 + per-layer host 0.92 + stage 1.21 + tail 11.01) + commit/emit 1.87 + draft 12.16; per layer-window: CPU experts 0.00 (0.00 entries), VRAM hits 30.51, PCIe 0.00
strata serve: expert tiers: GPU 144960 hits this request; since the start RAM 0 blobs, files 0 blobs 65582.7 MB read (the GGUF in place)
```

with, from the same engine's load lines, `layer split: CUDA1 runs layers 23-47, expert cache 12800 slots`,
`CUDA0 runs layers 0-22`, `layer split: 100% of the experts resident`, `24576 resident experts`, `5008 MiB of VRAM
free with everything loaded` — i.e. the delivered configuration, unchanged. **`2 checkpoints`** is the point: the
turn-boundary one *and* the 16,384-token mid-prompt one that killed the request before.

The response text (the model's own output; `content` is empty because this prompt starts a long reasoning block and
256 tokens do not finish it, exactly as in P6's request D, so the text is in `reasoning_content`) begins:

```
We need answer user's request. Need review provided document (source tree and documentation of one C++/CUDA engine? Actually Strata docs). User asks: "Please re…
```

And the request class that failed before, re-proved on its own server (`s4-fix-16400`, 16,452 prompt tokens):

```
HTTP 200; strata serve: prompt 16452 tokens = 0 reused + 16452 read in 75040 ms (219.2 tok/s), 64 generated in 3165 ms (20.2 tok/s), drafts accepted 35 of 48, 2 checkpoints
```

## 6. Regression check

`bash s4/s4_ctest.sh` (the D1 convention: `ZE_AFFINITY_MASK=0`, the oneAPI environment sourced first, the fixed
binary):

```
== S4 ctest 2026-10-03T18:34:25-06:00
HEAD: e41fa26 S4 (t_30d9ccfb): a split's mid-prompt checkpoint part is copied on ITS stage's device
engine binary: md5 e79b2ad632d37d32786228b660d35b29
== all tests whose name mentions the checkpoint path ==
  Test #42: conv_cache_test
1/1 Test #42: conv_cache_test ..................   Passed    0.00 sec
100% tests passed, 0 tests failed out of 1
90% tests passed, 5 tests failed out of 49
    5 - platform_memory_test (Failed)
   15 - elementwise_parity (Failed)
   16 - quantize_act_parity (Failed)
   19 - iq_multi_parity (Failed)
   45 - expert_multi_test (Failed)
```

**The same five failures, the same numbers, as D1's pre-fix baseline** (`d1/runs/ctest/full-graphdefault.log`: "5 tests
failed out of 49", those five ids and names, unmoved) — i.e. the suite is unchanged by the fix. `conv_cache_test` is
the checkpoint-path test that exists in this build and it passes. The other two conversation tests
(`conversation_snapshot_test`, `conversation_validation_test`) are behind `STRATA_BUILD_CONVERSATION_TESTS`, which is
**OFF** in `build-sycl` (`CMakeCache.txt`), so they are not among the 49 and were not run — and in any case they are
single-device tests: they cannot see a cross-card copy. The covered/covering evidence for this fix is the served arms
themselves.

## 7. Numerics

The served path exposes text, not token ids (the driver rigs' id protocol does not exist through `serve.server`), so
the statement is made on greedy text (`temperature 0`), field by field (`s4/s4_cmp.py` prints md5 per field):

| A | B | generated text |
|---|---|---|
| `s4-fix-32k` (fixed binary, config of record) | `s4-fix-32k-2` (the same, second run) | **byte-identical** (1264 chars, `b3a41ccc07f6c4d4`) |
| `s4-pc0-32k` (pre-fix binary, checkpoints off) | `s4-pc0-32k-2` (fixed binary, checkpoints off) | **byte-identical** (1326 chars, `554f3a29f1375481`) |
| `s4-pc0-32k-2` (checkpoints off) | `s4-fix-32k-2` (checkpoints on) | **differs** (554f… vs b3a4…) |
| `s4-ckpt-root` (fixed binary, `--prompt-cache 6 --prompt-cache-every 0`) | `s4-fix-32k-2` (fixed binary, config of record) | **byte-identical** (b3a4…) |
| `s4-oldckpt-32k` (pre-fix binary, `--prompt-cache 6 --prompt-cache-every 0`) | `s4-ckpt-root` (fixed binary, same flags) | **byte-identical** (b3a4…) |
| `s4-oldckpt-32k` (pre-fix binary, mid-prompt checkpoints off) | `s4-fix-32k-2` (fixed binary, mid-prompt checkpoints **on**) | **byte-identical** (b3a4…) |
| `s4-ckpt-4k` (fixed, 4,148 tokens, checkpoints on) | `s4-pc0-4k` (fixed, 4,148 tokens, checkpoints off) | **differs** (314 vs 280 chars) |
| `s4-closure-ckpt-4k` (`STRATA_SYCL_GRAPH=0`, checkpoints on) | `s4-closure-pc0-4k` (`STRATA_SYCL_GRAPH=0`, checkpoints off) | **differs** (same pair of texts) |
| `s4-ckpt-4k` (graph path, the default) | `s4-closure-ckpt-4k` (`STRATA_SYCL_GRAPH=0`) | **byte-identical** |

Read together:

1. **The fix changes no numerics — measured three ways on the path it touches.** (a) With the checkpoint machinery
   off, the pre-fix and the fixed binary give byte-identical text at this length (the code path the fix touches is not
   executed there). (b) With the checkpoint machinery on but the mid-prompt cadence off — the configuration the
   **pre-fix** binary can survive, since it never takes the checkpoint that killed it — the pre-fix binary and the
   fixed binary give byte-identical text (`b3a4…`), so the fix's own path is numerically inert. (c) The fixed binary
   with the mid-prompt checkpoint actually ON (the full config of record) gives that same `b3a4…` text, twice.
2. **The config of record's text is reproducible** run to run (two runs, byte-identical at 32K).
3. **`--prompt-cache` changes the served greedy text, at every length tested, and it is not this card's doing.**
   Checkpoints on vs off differ at 32,277 tokens (1326 vs 1264 chars) *and* at 4,148 tokens (314 vs 280 chars), with
   both binaries, and on both the graph path (D1's default) and the closure path (`STRATA_SYCL_GRAPH=0`) — while the
   graph and closure paths agree with each other byte for byte (`s4-ckpt-4k` == `s4-closure-ckpt-4k`). What it is
   *not*: the fix (points 1b/1c), the mid-prompt checkpoint (`--prompt-cache-every 0` gives the checkpoint-on text,
   and at 4K no mid-prompt checkpoint can fire at all), or the expert-cache auto-size (10,641-10,648 slots run to run;
   that does not track the text: off 10,648/10,645 both give 554f…, on 10,647/10,648/10,641 all give b3a4…).
   So the residue is in the **turn-boundary checkpoint path** (`generate.cpp:5334` -> `checkpoint_at` ->
   `:4396`/`:4397-4402`, the `1 checkpoints` line at 4K vs `0 checkpoints` with the flags off) or in something its
   device switches, syncs and host copies perturb — a pre-existing property of the served path that this card did not
   chase. **Already handed to a follow-up card** (§9). Both completions are coherent and of the same length class;
   nothing here says which is "right", and the engine's numerics of record stay the driver path's byte-identical
   greedy ids (D1).

## 8. Machine state left behind, and the relaunch/stop commands

**Left explicitly stopped: no engine, no server, port 8099 free, both cards free (card0 ~0.9 GiB — the desktop's —
card1 0.0 GiB).** Nothing else was restarted, because the next cards in this lane (D2/D3) need both GPUs; P6's
resident server was already down when this card started (D1 left it that way).

```bash
bash /home/michael/strata-xpu/strata/p6/p6_start.sh    # relaunch the resident server + dashboard on 127.0.0.1:8099
bash /home/michael/strata-xpu/strata/p6/p6_stop.sh     # stop it (server -> SIGTERM -> it QUITs the engine)
bash /home/michael/strata-xpu/strata/s4/s4_stop.sh     # the same stop path, for an S4 arm's server
```

The relaunched server is the **fixed** one: `p6_start.sh` starts `serve.server` with `strata-sycl-iq3s.json`, whose
`exe` is `build-sycl/strata` — the binary this card rebuilt (md5 `e79b2ad6…`). The arms above are that exact
invocation (same module, same config file, only `log`/`port`/`model_name` per-arm), and `s4-fix-32k` is the proof
that a 32K request now passes through it. `s4/bin/strata-s4-fix` is a copy of that binary.

## 9. Not validated

- **The `--prompt-cache` text difference (§7 point 3) is not explained.** It is *not* the fix (measured, three ways),
  not the mid-prompt checkpoint (measured), not the graph path (measured, `STRATA_SYCL_GRAPH=0` gives the same pair of
  texts), not the expert-cache slot count (measured). It is in the turn-boundary/root checkpoint's path — the only
  checkpoint that exists in the 4K arms — or in something that path perturbs. **Filed as `t_aae723be` (S5, at low
  priority so D2/D3 go first): reproduce it with the driver rig's id protocol, name the mechanism, fix or document.**
- Only one prompt family (the p10 32K depth-curve document) and only `temperature 0`, `max_tokens 64/256`,
  non-streaming; no sampling, no images, no control vectors, no tool calls.
- **Concurrency was not exercised.** The server serves one sequence at a time behind a FIFO; every arm here fired one
  request, and the checkpoint path's interaction with a queued second request (a *mount* of the checkpoint written by
  the first) was not re-tested. P10 measured that path crashing the split before the fix; it is the obvious next
  probe.
- No long-context re-test through the server: 128K/262K served requests were not fired (P6's long-context evidence is
  the 262K config at 26-45 tokens; the fix's 32,277-token request is the longest served request measured here). The
  prefill cost at 32K through the server is 217 tok/s (M6c's driver measurement: 344.8) — a served-path prefill
  difference this card did not investigate.
- The parking path (`--conversation-cache-mib > 0`) is still refused on a split (`generate.cpp:1269-1271`), untouched
  by this card: with parking on, save *and* restore go through a different, still-untested route.
- One measurement per configuration except where §7 says twice; no variance estimate. The sub-1% differences between
  arms (217.1 vs 217.5 tok/s prefill, 22.2 vs 22.5 decode) are inside that missing variance, not claimed as effects.
- The engine's release-path guarantee (P1b) was not re-run on the stall rig: the fix touches no verify/heartbeat code
  (the diff is one device guard around a checkpoint D2H copy, reachable only in `--serve` with `prompt_cache > 0`),
  and the suite's result is in §6.
- Nothing pushed: `origin` has no `sycl-xpu` branch (D1's note).

## 10. Rig and file map

All under `/home/michael/strata-xpu/strata/s4/` (committed) and `…/s4/runs/<arm>/` (raw, gitignored):

| file | what it is |
|---|---|
| `s4_serve_arm.sh` | one resident `serve.server` + a list of `<prompt_tokens>:<max_new>` real chat completions, in order, stopping at the first failure; per request keeps the response body, the client timing, and the engine's and server's own new log lines |
| `s4_cfg.py` | this arm's config (config of record verbatim, `log`/`port`/`model_name` per arm) + the server's effective engine argv + its sha256 + the `--extra` diff |
| `s4_prompt.py` | a text prompt of an exact token count (the pack's own pure-python tokenizer, `tools/strata_tokenizer.py`, the same loader P10 used) |
| `s4_ctest.sh` | the engine's suite (checkpoint tests by name, then the full suite) |
| `s4_stop.sh` | stop the server/engine and show both cards free |
| `s4_cmp.py`, `s4_text.py`, `s4_fields.py`, `s4_show.py`, `s4_engine_log.py` | the numerics/text/response readers |
| `s4_chain2.sh`, `s4_chain3.sh`, `s4_chain4.sh` | the second chain (repeat arms + the boundary re-proof), the 4K on/off probe, and the 4K probe with `STRATA_SYCL_GRAPH=0` |
| `s4_evidence.sh` | assembles `s4/S4-EVIDENCE.txt` from the arm directories, plus the git diff of the fix and the suite's result |

Arm data: `/home/michael/strata-xpu/strata/s4/runs/<arm>/` (`log.txt`, `cfg.json`, `argv.txt`, per-request
`engine-ctx<N>.txt`, `server-ctx<N>.txt`, `resp-ctx<N>.json[.timing.json]`, `markers.tsv`); the engines' own full logs
are `/home/michael/strata-xpu/logs/s4-<arm>-engine.log`, the servers'
`/home/michael/strata-xpu/logs/s4-<arm>-server.log`. The binaries are `d1/strata-after-histfix` (pre-fix, md5
`72ec74ae…`), `s4/bin/strata-s4-fix` (fixed, `e79b2ad6…`) and `build-sycl/strata` (the fixed one, restored).

Commits: `2c78930` (the rig), `e41fa26` (the fix), and the write-up/evidence commit after them. The
`--prompt-cache` finding is handed on as `t_aae723be` (S5).
