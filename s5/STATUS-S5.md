# S5 (t_aae723be) — `--prompt-cache` moves the served greedy ids: the prompt read's shape, not the checkpoint

**Verdict: measured, named and documented; no code change.** The served path's greedy ids really do depend on
`--prompt-cache`, reproducibly, at 4K and at 32K, on the split and on one card. The cause is **the segmentation of
the prompt read**, not the checkpoint machinery: with `--prompt-cache > 0` the read stops at the last turn token
(always the `<|im_start|>` of the reply's own header, i.e. within a few tokens of the end,
`src/program/generate.cpp:5310-5313`), is issued as **separate runs** (`:5326-5339`), and the header's few tokens
then go through the **decode windows** (`windows_ok`/`read_windows`, `:5111`/`:5119`, `--short-read`, default 64)
rather than through the batched prompt run that read the other 4,144. The last prompt tokens are therefore summed in
a **different order** than in a single-read (`--prompt-cache 0`) request, and this model amplifies that FP32-level
change into a different (equally legitimate) continuation. The checkpoint **save** itself is inert, proved: with the
boundary moved to the last prompt token the read is one segment again and **all 64 ids are byte-identical to
`--prompt-cache 0`** while the prompt line still reports `1 checkpoints`.

Rig and raw output: `/home/michael/strata-xpu/strata/s5/` (rig under `s5/*.sh`, `s5/*.py`, per-arm files under
`s5/runs/<tag>/`, the served arms under `s5/served/<tag>/`) and `s5/S5-EVIDENCE.txt` (this write-up's raw output,
assembled by `s5/s5_evidence.sh`). Branch `sycl-xpu`; config of record `strata-sycl-iq3s.json` **unmodified**, no
source change at all in this card. Every arm ran **one** binary,
`3dbea776ac8bf94414c199bdae05877c` (50,820,208 bytes, built 2026-10-03 23:52) — the build at this HEAD, i.e. S4's
fix plus D2b/D2c's load-phase kernel warm-ups. Those touch `src/program/generate.cpp` only at `:1942` and `:4650`
(`git diff e41fa26..HEAD -- src/program/generate.cpp`, 29 added lines: the MMVQ layout switch at its shipped
default and two warm-up calls whose own comment says "nothing it computes is read"), so the prompt-read and
checkpoint code this card cites is S4's, line for line — and the arms reproduce S4's own stored served answers
byte-identically at both lengths in both settings (§4), so neither S4's fix nor D2's warm-ups can be the difference.
Two B70s, no other GPU work in flight; `ZE_AFFINITY_MASK` unset for every split arm (PLAN §11 U11).

## 1. The rig: the driver's own `T` ids, on the ids the server itself sent

The served path exposes text, not ids, so the statement is made on the engine's own greedy `T <id>` lines — the same
protocol the D1/D2/D3 rigs use — driven directly (`s5/s5_drive.sh <tag> <GENLINE> [--extra ARGS] [--onecard N]`,
engine argv read out of the config of record's JSON, one `GEN <max_new> <ids>` line on stdin, ~70 s per arm at 4K).

The prompt is **the served arm's own prompt**, not an approximation: `s5/s5_ids.py` renders the chat template over
the request body `p10/p10_client.py` sends and tokenizes it with the pack's tokenizer — the two calls
`Service.prepare` makes (`serve/server.py:1141`) — and the check is the engine's own count against the S4 arm's:
**4,148 tokens** for `s4/runs/s4-ckpt-4k/prompt-ctx4096.txt` and **32,277** for `s4/runs/s4-fix-32k/prompt-ctx32768.txt`,
both exact. The ids are then tied back to text: the arm's ids detokenize **byte-identically** to the served arm's own
answer, both settings, both lengths (§4).

## 2. The id-level reproduction (acceptance item 1)

All arms: the config of record's args verbatim + `--serve`, ZE_AFFINITY_MASK unset + `--layer-split auto` (the server's
own `gpu: [0,1]`), `SYCL_CACHE_DIR=/home/michael/strata-xpu/sycl-cache/m6c`, prompt = 4,148 ids, greedy, 64 ids out.
`md5` is the md5 of the file of `T` lines (`s5/runs/<tag>/ids.md5`).

| arm | flags beyond the config of record | checkpoints | prompt read (PP lines) | 64 greedy ids, md5 | vs `s5-pc0` |
|---|---|---|---|---|---|
| `s5-pc0` | `--prompt-cache 0 --prompt-cache-every 0` | 0 | 512×8, **4147** (one run) | `7f863468c804113e677df220b69cefd9` | — |
| `s5-pc6` | *(none — the served setting)* | **1** | 512×8, **4143**, 4147 | `f820fbbaa99aacc66b0a11fcffbb0c39` | **differs**, first divergence at id 27/64 |
| `s5-pc6-ns` | `--short-read 0` (tail batched instead of windowed) | 1 | 512×8, 4143, 4147 | `400282fde73e74bbc1b86d906823537d` | differs, at id 27 |
| `s5-pc6-ns-c` | `--short-read 0`, repeat | 1 | 512×8, 4143, 4147 | `400282fde73e74bbc1b86d906823537d` | **identical to `-ns`** (repeatable) |
| `s5-pc6-ttnon` | `--turn-token -1` (no boundary found) | 0 | 512×8, 4147 | `7f863468c804113e677df220b69cefd9` | **identical to `s5-pc0`** |
| `s5-pc6-ttlast` | `--turn-token 198` (the id at `n-1`: boundary = end of read) | **1** | 512×8, 4147 (one run) | `7f863468c804113e677df220b69cefd9` | **identical to `s5-pc0`** |
| `s5-pc6-tt1tok` | `--turn-token 248068` (`<think>`, at `n-2`: a **1-token** tail) | 1 | 512×8, **4146**, 4147 | `f820fbbaa99aacc66b0a11fcffbb0c39` | **identical to `s5-pc6`** |
| `s5-pc6-tt1tok-ns` | `--turn-token 248068 --short-read 0` | 1 | 512×8, 4146, 4147 | `c71b24a6ddde00659613f6147802ede2` | differs, at id 2 |

Prompts differ between arms only where stated (none of them do — same 4,148 ids everywhere).

One card (`ZE_AFFINITY_MASK=0`, no `--layer-split`):

| arm | flags | checkpoints | 64 ids md5 | vs the split arm |
|---|---|---|---|---|
| `s5-pc0-1card` | `--prompt-cache 0 --prompt-cache-every 0` | 0 | `615a2f85bb1cf060bd6d6b08cdbd5134` | differs from `s5-pc0` |
| `s5-pc6-1card` | *(none)* | 1 | `586aba15a2f3ff532fcf8c9b54ad8205` | differs from `s5-pc0-1card` |

so **the effect is not split-specific** — but one card is **not a clean control for the checkpoint setting**: both
single-card arms run `0.77-0.78` CPU experts per layer-window (the full expert set does not fit one card), and the
GPU/CPU expert kernels round differently by a documented rule (`docs/DETAILS.md`, the `#152`/`#410` paragraph), which
moves the ids by itself. The single-card pair is evidence that the phenomenon survives without a split, not
attribution.

32K (32,277 ids, 256 greedy, the S4 length):

| arm | checkpoints | prompt read | 256 ids md5 | vs `s5-pc0-32k` |
|---|---|---|---|---|
| `s5-pc0-32k` | 0 | 148,323 ms (217.6 tok/s) | `ed002bb41c9faf3a42686925c81a1b3b` | — |
| `s5-pc6-32k` | **2** | 148,581 ms (217.2 tok/s) | `05996f80eeba03763a2110ad2f4badc9` | **differs**, first divergence at id **9**/256 |

Prompt read cost is a wash (217.6 vs 217.2 tok/s, same tokens), so this is not a cost/benefit question at this length.

## 3. What it is, and what it is not (acceptance item 2)

**Named, with `file:line`:**

1. **The trigger is the read boundary, not the checkpoint.** `o.prompt_cache > 0` makes the engine look for the last
   `<|im_start|>` (`turn_at`, `generate.cpp:5310-5313`) and then read the prompt as the **segments between boundaries**
   (`:5326-5339`): `[0, turn_at)` through the batched prompt path (`sp.run`), a checkpoint at `turn_at`
   (`:5363` → `checkpoint_at`, `:4394-4430`), and `[turn_at, n-1)` — the reply header, 4 ids here — through
   `read_windows` (`:5119`), because `windows_ok` (`:5111`) accepts any segment ≤ `--short-read` (64, `:408`).
   With `--prompt-cache 0` there is no boundary and the whole prompt is **one** `sp.run` (`:5339`, the `to == n-1`
   iteration): the last prompt token is summed inside the run's last 512-token chunk instead of in a run of its own.
2. **The checkpoint save is inert** — `s5-pc6-ttlast`: the boundary is the last prompt token, so the read is one
   segment *and* a checkpoint is still saved (`1 checkpoints`) → **all 64 ids equal `s5-pc0`**. Same for
   `s5-pc6-ttnon` (no boundary found → nothing saved, `0 checkpoints` → equal `s5-pc0`). So this is a **read/write of
   the same state with a different summation order**, not a perturbation of a later numeric choice — the answer to
   the card's lead (1).
3. **A 1-token tail is enough** — `s5-pc6-tt1tok` (boundary at `n-2`) gives **all 64 ids of `s5-pc6`**, i.e. the
   effect is in the last prompt token's own computation, not in the length of the header.
4. **The tail's path is a second, independent shape** — `s5-pc6-tt1tok-ns` (same boundary, tail read batched instead
   of windowed) gives a **third** sequence (first divergence at id 2). So for the same prompt token there are three
   continuations: computed inside the batched run's chunk, in a run of its own, or in a decode window. The prompt
   attention's summation order follows the launch shape; the engine's own words for exactly this family are
   `same accuracy, another summation order` (`src/kernels/sycl/qsa_prompt_attn.cpp:1035`, the comment on why the
   SYCL port takes the portable v1 kernel).
5. **The segment's chunk is chosen per segment** (`request_chunk`, `:3794-3800`: `min(--prefill, the segment rounded
   up to 256)`), and `lend` (`:5229`) only relayouts the prompt path's borrowed buffers when the new segment needs
   more than the loan in hand — a second, independent knob on the run's shape.

**Bounded statement of what it is NOT** (each with the arm that excludes it):

- not the checkpoint *save*, its device sync, or the state copy: `s5-pc6-ttlast` (1 checkpoint, byte-identical to
  `--prompt-cache 0`), `s5-pc6-ttnon` (0 checkpoints, identical);
- not the checkpoint *bookkeeping* (the chain, the retention policy, the extra host memory): same two arms;
- not the mid-prompt checkpoint cadence: `--prompt-cache-every 0` reproduces the `--prompt-cache 6` text (S4 §7) and
  at 4,148 tokens no mid-prompt checkpoint can fire; the 32K arm differs with 2 checkpoints and a
  `--prompt-cache-every 0` variant differs identically (S4 §7);
- not the graph path: `STRATA_SYCL_GRAPH=0` (closure) reproduces the graph path's text byte-for-byte (S4 §7); every
  arm here runs the default graph path;
- not the layer split: the one-card pair differs too (§2), though one card is confounded by CPU experts;
- not the expert-cache auto-size: it varies 10,641-10,648 slots run to run and does not track the text (S4 §7); every
  split arm runs 0.00 CPU experts;
- not the drafts as such: `s5-pc6` and `s5-pc6-tt1tok` have different draft acceptance (20/26 vs 21/27) and **all 64
  ids the same**, while `s5-pc6-ns` and its repeat have identical drafts and identical ids — what tracks is the read
  segment structure, not the draft pattern.

## 4. The served path, verbatim (acceptance items 1 and 4)

`s5/s5_serve.sh` = the S4 rig as-is (`s4/s4_serve_arm.sh`, one resident `serve.server`, one real
`POST /v1/chat/completions`), the config of record verbatim (checkpoints ON), both B70s:

| arm | request | result | engine's own line | answer |
|---|---|---|---|---|
| `s5-serve-4k` | 4096:64 | **HTTP 200**, 22.3 s | `prompt 4148 tokens = 0 reused + 4148 read in 18798 ms (220.7 tok/s), 64 generated in 3394 ms (18.9 tok/s), drafts accepted 20 of 26, 1 checkpoints` | 314 chars, **byte-identical to the driver arm `s5-pc6`'s ids** |
| `s5-serve-32k` | 32768:64 | **HTTP 200**, 151.5 s | `prompt 32277 tokens = 0 reused + 32277 read in 148712 ms (217.0 tok/s), 64 generated in 2680 ms (23.9 tok/s), drafts accepted 44 of 45, 2 checkpoints` | prefix of the driver arm `s5-pc6-32k`'s detokenized ids (all 316 chars) |

**S4 is not regressed:** a 32,277-token served request on the split still succeeds with the config of record's own
checkpoints and `2 checkpoints` in the engine's prompt line. Nothing in this card touches the engine: no source
change, no config change, no flag default (`git log` = the rig, the arms, the write-up and the `docs/DETAILS.md`
note).

The driver rig and the served path agree **exactly** on the ids, in both settings and at both lengths:

| driver arm | served arm it reproduces | text | result |
|---|---|---|---|
| `s5-pc0` (4,148 ids, 64) | `s4/runs/s4-pc0-4k` | 280 chars | **equal** |
| `s5-pc6` (4,148 ids, 64) | `s4/runs/s4-ckpt-4k` | 314 chars | **equal** |
| `s5-pc0-32k` (32,277 ids, 256) | `s4/runs/s4-pc0-32k-2` | 1,326 chars | **equal** |
| `s5-pc6-32k` (32,277 ids, 256) | `s4/runs/s4-fix-32k` | 1,264 chars | **equal** |

So `--prompt-cache` ON vs OFF is a real, reproducible, id-level property of the served path — and the difference is
one position's worth of rounding, not a broken read: 27 of 64 ids shared at 4K, 9 of 256 at 32K.

## 5. Why this is documented rather than patched (acceptance item 3)

Both sequences are legitimate greedy continuations of the same ids: the same weights, the same KV, the same head —
only the *shape* of the last prompt tokens' reduction differs (and the engine itself describes that family as
`same accuracy, another summation order`). There is no defect to fix at a low level:

1. **Checkpointing at the turn boundary is required to be a boundary.** The only way to make a
   `--prompt-cache > 0` read identical to a single read is to read the whole prompt in one run — and then the state
   *at* `turn_at` no longer exists to checkpoint (a checkpoint is the running state at that position, `:4394-4430`).
   Removing the turn-boundary checkpoint would drop the prefix reuse the server delivers, which the card forbids.
2. **Making the summation shape-independent is a kernel-numerics project.** The batched prompt attention would have
   to produce bit-identical results for a query inside a 512-token chunk, in a 4-token run and in a decode window.
   That is the same class of change the project has already refused twice for the decode attention (a faster kernel
   that moves ids is refused, and the SYCL port runs the portable v1 kernel for exactly this reason,
   `qsa_prompt_attn.cpp:1035`), and it is out of scope for this card.
3. **The configuration the user gets is the one that was measured**: the served text is the `--prompt-cache 6` text,
   reproducible run to run, and it is the text every served arm of S4/D1 reports.

What is documented instead (`docs/DETAILS.md`, the reproducible-greedy paragraph, committed with this card): the
tail read is visible on a **fresh** request (0 reused, nothing resumed) — the checkpoint at the reply header splits
the read and sends the header through the decode windows — with the measured 4,148/32,277 token numbers, the
`--turn-token` proof that the save is inert, the `same accuracy, another summation order` reference, and the
practical rule the engine's own docs already give for byte-identical repeats: **`--prompt-cache 0`**. That rule now
also covers *comparability*: the driver rigs (P10/D1/D2/D3) run `--prompt-cache 0`, so a served answer can only be
compared with them id for id when the server does too.

`docs/AMD_HIP.md:222-224`'s claim that a split's checkpoints "give exactly the tokens of a fresh read" is **left
untouched**: it is AMD's own measurement of a different backend's kernel, and the effect named here is a
summation-order property of *this* kernel set — it can hold there and not here. There is no ROCm host in this rig, so
it was not re-measured and is stated as not validated (§7).

## 6. Machine state left behind, and the relaunch/stop commands

**EXPLICITLY STOPPED**: no engine (`pgrep -x strata` empty), no `serve.server`, port 8099 free; card0 in its desktop
idle state, card1 free. Both GPUs are free for D2/D3. Relaunch the config of record:
`bash /home/michael/strata-xpu/strata/p6/p6_start.sh`; stop: `p6_stop.sh` or `s4/s4_stop.sh`. One arm again:

```
cd /home/michael/strata-xpu/strata
bash s5/s5_drive.sh s5-pc6 s5/prompts/gen-ctx4096.txt                    # the served setting
bash s5/s5_drive.sh s5-pc0 s5/prompts/gen-ctx4096.txt --extra "--prompt-cache 0 --prompt-cache-every 0"
/usr/bin/python3 s5/s5_cmp.py table s5-pc0 s5-pc6
bash s5/s5_serve.sh s5-serve-4k 4096:64                                  # the served path
bash s5/s5_evidence.sh                                                    # reassemble S5-EVIDENCE.txt
```

## 7. Not validated

- **AMD's claim was not re-measured**: no ROCm/AMD host here, and the effect is kernel-set specific (`docs/AMD_HIP.md`
  is left as AMD's measurement).
- **One measurement per configuration** except `s5-pc6-ns`/`-c` (identical), and the 4K arms all repeat S4's texts
  byte for byte across cards; no formal variance estimate.
- **No 128K/262K id-level arms** — the card's "32K if cheap" was taken; 128K/262K served requests were not repeated.
- **The one-card pair does not attribute the difference** (0.77-0.78 CPU experts per layer-window on one card, the
  documented `#152`/`#410` rounding rule); it only shows the phenomenon survives without a split.
- **No instrumentation**: the attribution is by arm (which read shape produced which ids), not by a kernel counter or
  a logit-difference measurement of the perturbation's size.
- **The "no drafts at all" discriminator could not run**: the engine refuses `--spec 1` for a native IQ pack
  (`--spec T (T >= 2)`), and `--mtp-max-t 2` was not run.
- **Concurrency** (a queued second request, the checkpoint MOUNT path) untouched.
- **Nothing pushed**: `origin` has no `sycl-xpu` branch; the work is committed on the branch (`git log`), as this
  profile's convention.
