# P7 (card `t_6789d6df`) — the drafter's provenance made faithful, and what running without it costs

Mike, 2026-10-03: *"does strata use the headless approach? If so follow strata faithfully. I dont think strata needed the
MTP drafter model from the W4A16 model we downloaded, so it probably shouldnt be used."*

**Answers, measured:** (1) upstream is **not** headless — the resident server **cannot** run without the drafter at all
(`generate.cpp:3866-3871`: `--serve` needs `--mtp`); (2) he is right about the artifact: the drafter is now built
upstream's way from the **public canonical checkpoint** at a **pinned revision**, SHA256-verified 31/31, from a neutral
path with no W4A16 provenance anywhere in it; (3) the rebuild is **provably the same drafter** — byte for byte, down to
the same 150 output tokens at the same md5; (4) running without a drafter is now a **measured** option, in the bench
path where upstream makes `--mtp` optional, and it costs **-39% decode** at 4K (12.99 vs 21.37 tok/s).

## 1. Provisioned upstream's way (`tools/mtp_fetch.py`, pinned revision)

`bash p7/p7_provision.sh` -> `inventory` -> `fetch` -> `verify` -> `mtp_pack.py --experts q2_0` -> `mtp_rt.py` ->
`cp data/draft_vocab.bin`. Raw output: `~/strata-xpu/logs/p7_provision.log`. **No W4A16 checkpoint and no
`tools/mtp_w4a16_adapter.py` is involved at any step.**

```
== revision: STRATA_MTP_REVISION=<unset>
32:PINNED_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
===================== 1a. inventory (headers only) =====================
# MTP block in the BF16 checkpoint
31 tensors, 5.214 GB, in 28 shards.
===================== 1c. verify (SHA256 of the pinned revision) =====================
STRATA_MTP_REVISION override : <unset - the pinned revision is used>
PINNED_REVISION              : de4b8e4d43b917e7706784d8bb445c9af86a3540
REVISION read this run       : de4b8e4d43b917e7706784d8bb445c9af86a3540
repo read                    : https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/
hashes apply (pinned)        : True

verify: 31/31 tensors SHA256-verified against the pinned revision
mtp-manifest.json: 31 tensors, 5.214 GB of tensor bytes
  verify exit=0
===================== 2. mtp_pack.py --experts q2_0 =====================
wrote /home/michael/strata-xpu/mtp/mtp-q2_0.gguf: 0.889 GB of tensor data
===================== 3. mtp_rt.py =====================
experts.bin 707788800 B, dense.bin 116099072 B, 29 tensors -> /home/michael/strata-xpu/mtp/rt
===================== 4. draft_vocab.bin =====================
  cp exit=0  (425196 B)
```

**Revision used: the pinned one** (`de4b8e4d43b917e7706784d8bb445c9af86a3540`) — `STRATA_MTP_REVISION` was **unset**, so
no override was applied and the SHA256 set applies (`hashes apply (pinned): True`).

## 2. Where it lives

| | path | size |
|---|---|---|
| the runtime the engine loads | `~/strata-xpu/mtp/rt` (`dense.bin` 110.7 MiB + `experts.bin` 675 MiB + `dense.txt` + `draft_vocab.bin`) | 786 MB |
| the packed GGUF it came from | `~/strata-xpu/mtp/mtp-q2_0.gguf` | 889 MB |
| the verified canonical tensors (the provenance record) | `~/strata-xpu/mtp/canonical/` (31 tensors + `mtp-manifest.json` + `mtp-inventory.json/.md`) | 4.9 GB |

**No `strata-w4a16` path is referenced by the config of record.** `strata-sycl-iq3s.json` `--mtp` is now
`/home/michael/strata-xpu/mtp/rt` (the file is gitignored, so it is a local edit; the `_note` was rewritten to state
where the drafter comes from and that the W4A16 route is out of scope).

```
-          "--mtp", "/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0",
+          "--mtp", "/home/michael/strata-xpu/mtp/rt",
```

## 3. It IS the same drafter: 31/31 tensors and both runtime files, byte for byte

`python3 p7/p7_digest_compare.py` (raw output: `p7/p7-digest-comparison.json`; every digest **recomputed here from the
bytes on disk**, not read from either manifest, so a stale manifest cannot fake agreement):

```
digest comparison: 31/31 tensors IDENTICAL, 0 differ, 0 recorded digests stale

RUNTIME DIRS THE ENGINE LOADS
  canonical -> /home/michael/strata-xpu/mtp/rt
  W4A16     -> /run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/rt-q2_0
dense.bin           116099072 B  IDENTICAL   c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a
experts.bin         707788800 B  IDENTICAL   09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f
dense.txt                1851 B  IDENTICAL   991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd
draft_vocab.bin      ABSENT on the W4A16 side (see §4)
```

So the old artifact's **weights are the canonical head** — the defect really was provenance, not numbers. The two
built-in sanity checks in `mtp_fetch.py` (`fetch` hashes what it writes, `verify` re-hashes the pinned set) both pass,
and the W4A16 manifest's own recorded digests are consistent with the bytes on disk (0 stale).

## 4. The one real difference: upstream's `draft_vocab.bin` step was missing

`docs/ORCA.md:39` ends the recipe with `cp data/draft_vocab.bin mtp/rt/draft_vocab.bin`; the W4A16-derived artifact
never had it (its `rt-q2_0/` holds only `dense.bin`, `dense.txt`, `experts.bin`). With the file present the engine
builds the draft head over that token subset instead of the whole vocabulary (`src/core/mtp.cpp:389-410`):

```
strata mtp: draft head over 106299 tokens (212.9 MiB)      <- with draft_vocab.bin (canonical, faithful)
(no such line without it: the draft head IS the native head, which is already in VRAM)
```

Serving both, same prompt, same 4K config of record, 256 max-new (`~/strata-xpu/p7/runs/`, logs in
`logs/p7_arms2_4096.log` and `logs/p7_arms_4096.log`):

| arm (`--mtp` target) | drafts accepted/offered | windows | tokens/window | draft ms/window | decode ms | decode tok/s | output md5 |
|---|---|---|---|---|---|---|---|
| `new` = canonical rt + `draft_vocab.bin` (repeat a) | 101 / 140 | 50 | 3.00 | 11.65 | 6662.3 | **22.5** | `ad985c23` |
| `new` = same arm again (repeat b) | 101 / 140 | 50 | 3.00 | 11.47 | 6669.9 | **22.5** | `ad985c23` |
| `new-nodv` = canonical rt, no `draft_vocab.bin` | 102 / 139 | 49 | 3.06 | 19.20 | 6986.2 | 21.5 | `ad985c23` |
| `old` = W4A16-derived rt (as served so far) | 102 / 139 | 49 | 3.06 | 19.19 | 7011.1 | 21.4 | `ad985c23` |

Read it in one line: **the two rt directories with the same (absent) `draft_vocab.bin` are the same run** (102/139,
3.06 tokens/window, 21.4-21.5 tok/s, `draft` 19.19 vs 19.20 ms/window) — the W4A16-derived artifact and the canonical
rebuild are indistinguishable, which is what byte-identical weights predict. The `new` arm reproduces itself to 0.1%
(6662.3 vs 6669.9 ms, 101/140 both times, `draft` 11.65 vs 11.47 ms), so the one arm that differs does so **because it
follows the last step of upstream's recipe**: the draft head over the 106,299-token subset is **-7.5 ms per window** at
drafting, i.e. **+5% decode** — twice the arm's own run-to-run spread. The **150 generated tokens are the same ids at the
same md5 (`ad985c23`) in all four runs**, so this changes no answer — and it is also the md5 P5's shipped 4K arm
produced, i.e. the rebuild reproduces the served stream exactly.

## 5. What the drafter costs in VRAM — the card's "+0 MiB" is wrong, measured

The engine prints the drafter's own allocation, and an A/B in the bench path (same flags, only `--mtp` differs) shows it
in the free-VRAM figure the expert cache is sized from and in the reserve:

```
with the drafter:   strata mtp: draft layer loaded, 795 MiB of VRAM (experts 675, dense 111), files read in 0.34 s
                    strata generate: expert cache auto: 26.20 GiB free, 700 MiB reserved (+218 MiB for the draft head) -> 10204 slots
                    strata mtp: draft head over 106299 tokens (212.9 MiB)
without it:         strata generate: expert cache auto: 26.97 GiB free, 700 MiB reserved (+0 MiB for the draft head) -> 10602 slots
```

- The drafter holds **795 MiB of VRAM** for its own weights (675 experts + 111 dense) on CUDA0 (839 MiB at 32K, where
  its KV state is bigger), plus **212.9 MiB** for the draft head over `draft_vocab.bin`'s subset = **~1,008 MiB**, not 0.
- The auto expert cache reads free VRAM *after* the drafter is loaded (26.20 vs 26.97 GiB at 4K; 25.77 vs 26.59 GiB at
  32K) and additionally reserves the draft head's 218 MiB, so the drafter leaves the cache **398 slots smaller** at 4K
  (10204 vs 10602) and **419 smaller** at 32K (10029 vs 10448) — it does compete with the expert cache, by ~400 experts
  on the card that holds it.
- Where "+0 MiB" came from: the `(+N MiB for the draft head)` field is only the *reserve* (`bind_bytes`,
  `generate.cpp:2689-2690`), and it reads **+0 MiB in the serve path** (where the native head is not loaded at sizing
  time) and **+218 MiB in the bench path** — i.e. the serve build sizes the cache as if the draft head were free and the
  212.9 MiB comes out of the un-reserved free VRAM afterwards. The weights (795 MiB) are allocated before either and are
  simply already out of `free_b`. `--serve`'s own line ends the ambiguity:
  `strata serve: 5780 MiB of VRAM free with everything loaded` (bench, headless: `26.97 GiB free` before the cache).
  Old ablation cards quoting "+0 MiB of VRAM for the draft head" were quoting this reserve field, not a measurement.

## 6. The headless arm — what running without a drafter costs

**It cannot be measured through the server**: `--serve` refuses to start without a head
(`generate.cpp:3866-3871`, raw message below), which is the code-level confirmation of *"Strata's persistent server also
requires the MTP runtime"* (`docs/ORCA.md`) and of `tools/test_setup_golden.json` writing `--mtp` in every config.

```
strata generate: token graph hit path: 24576 resident experts, decided on the device
strata serve: needs --spec T, --mtp DIR and --prefill CHUNK (and a fillable --expert-cache; the graphed hit path
              additionally needs --expert-profile P)
engine exited 2
```

So the control is the engine's own bench path (`strata generate`), where upstream *does* make `--mtp` optional
(`if (!o.mtp.empty())` guards every use; `--mtp` ignored below `--spec 2`, `generate.cpp:2514-2519`). Both arms there run
the same flags — `--spec 4 --spec-min-p 0.5` (so suffix/prompt-lookup drafting stays on), `--kv int8`,
`--expert-cache auto --expert-profile`, `--mmap-experts`, same prompt, 256 tokens. Caveat stated up front: the bench path
**cannot take `--layer-split`** (it is `--serve`-only, `generate.cpp:1299`), so these two arms run on **one** GPU
(prefill 135-139 tok/s, not the two-GPU 244 tok/s of the serve arms); the comparison between the two arms is apples to
apples, the absolute tok/s is not the config of record's.

### 4K (prompt-needle-ctx4096, 3,831 ids, 256 generated)

| arm | decode tok/s | prefill tok/s | tokens/round | drafts accepted/offered | suffix lookup | rounds |
|---|---|---|---|---|---|---|
| drafter (`--mtp ~/strata-xpu/mtp/rt`) | **21.37** | 135.56 | **3.10** | 174 / 233 (0.747) | 28 windows, 92/98 | 83 |
| headless (`--spec 4`, no `--mtp`) | **12.99** | 138.51 | **1.66** | 103 / 448 (0.230) | 45 windows, 103/121 | 155 |

Raw engine lines:

```
== with the drafter ==
speculation              83 rounds of 6, drafts accepted 174 of 233 (0.747), 3.10 tokens per round
window sizes             T1:5 T2:5 T3:5 T4:61 T5:0 T6:7  (min draft probability 0.50)
suffix drafts            28 windows, drafts accepted 92 of 98
accepted per round       0:18 1:11 2:9 3:40 4:0 5:5
decode                   256 tokens in 11979.9 ms  ->  21.37 tok/s
prefill                  3831 tokens in 28260.6 ms  ->  135.56 tok/s  (time to first token 28453.4 ms)

== headless ==
speculation              155 rounds of 6, drafts accepted 103 of 448 (0.230), 1.66 tokens per round
suffix drafts            45 windows, drafts accepted 103 of 121
accepted per round       0:116 1:9 2:3 3:23 4:1 5:3
decode                   256 tokens in 19714.8 ms  ->  12.99 tok/s
prefill                  3831 tokens in 27659.6 ms  ->  138.51 tok/s  (time to first token 27844.6 ms)
```

What that says:
- **The drafter buys +64% decode here** (12.99 -> 21.37 tok/s, 1.645x), which is entirely the tokens-per-pass
  difference: 3.10 vs 1.66. Prefill does not move (135.6 vs 138.5 tok/s, -2.1%, i.e. the null result MTP deserves in
  prefill, within run noise).
- Without a head, the engine is **not** at 1.0 tokens/pass: prompt lookup alone reaches **1.66** on this prompt (155
  rounds, of which 45 were lookup windows) — but note this needle prompt is heavily repetitive, which is exactly the
  case where suffix lookup is strong; on ordinary text it would be weaker. The drafter adds 1.87x on top of the lookup
  engine (3.10 vs 1.66).
- Headless is not "the same work minus the drafter": it runs **87% more verify windows** (155 vs 83) and 187% more
  rounds of lookups, i.e. it pays the window cost almost twice as often.

### 32K (prompt-ctx32768, 32,255 ids, 256 generated, `--kv-resident 32768`)

| arm | decode tok/s | prefill tok/s | tokens/round | drafts accepted/offered | suffix lookup | rounds |
|---|---|---|---|---|---|---|
| drafter | **21.63** | 259.65 | **3.07** | 174 / 213 (0.817) | 6 windows, 20/24 | 84 |
| headless | **10.60** | 260.25 | **1.38** | 71 / 547 (0.130) | 36 windows, 71/94 | 188 |

```
== with the drafter ==
speculation              84 rounds of 6, drafts accepted 174 of 213 (0.817), 3.07 tokens per round
suffix drafts            6 windows, drafts accepted 20 of 24
decode                   256 tokens in 11833.3 ms  ->  21.63 tok/s
prefill                  32255 tokens in 124223.7 ms  ->  259.65 tok/s  (time to first token 124445.9 ms)

== headless ==
speculation              188 rounds of 6, drafts accepted 71 of 547 (0.130), 1.38 tokens per round
suffix drafts            36 windows, drafts accepted 71 of 94
decode                   256 tokens in 24161.6 ms  ->  10.60 tok/s
prefill                  32255 tokens in 123940.6 ms  ->  260.25 tok/s  (time to first token 124130.1 ms)
```

The 4K result holds and grows with depth: **+104% decode (10.60 -> 21.63 tok/s, 2.04x)** at 32K, tokens/round 1.38 vs
3.07, drafts accepted 0.130 vs 0.817, verify windows 188 vs 84 (+124%). Prefill again does not move (260.25 vs 259.65
tok/s, +0.2%) — the null result MTP deserves there. The lookup-only arm's own advantage shrinks with length (1.66
tokens/round at 4K -> 1.38 at 32K, its suffix windows 45 -> 36), while the drafter's acceptance *rises* (0.747 -> 0.817),
so the gap between headless and MTP widens exactly as the context a real session runs at.

## 7. The rule (recorded in the config of record, PLAN/BRIEF and the `strata-approach` skill)

**The drafter is the model's own MTP head, obtained from the model's canonical checkpoint with upstream's tools, at the
pinned revision, with SHA256 verification of every tensor — never from the W4A16 checkpoint and never through
`tools/mtp_w4a16_adapter.py`.** The recipe is exactly:

```
python3 tools/mtp_fetch.py inventory --out <dir>          # headers only
python3 tools/mtp_fetch.py fetch     --out <dir>          # the 31 mtp.* tensors, range reads, resumable
python3 tools/mtp_fetch.py verify    --out <dir>          # 31/31 SHA256 against the pinned revision; exit 3 = bad
python3 tools/mtp_pack.py --src <dir> --experts q2_0 --out mtp/mtp-q2_0.gguf
python3 tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin            # do not skip this (docs/ORCA.md:39)
```

Do not skip the last line: it is what makes the draft head cheap (212.9 MiB over 106,299 tokens instead of the whole
vocabulary) and it is worth ~7.5 ms per verify window at 4K here.

## 8. Retirement recommendation (needs Mike's sign-off — nothing deleted)

The W4A16-derived artifact is now **redundant and provably so** (identical weights, and it is missing a step of
upstream's recipe). Deleting it frees, on the external volume:

- `/run/media/michael/2208B12208B0F63F/strata-w4a16/mtp-bf16/` — **8.1 GB** total: `rt-q2_0/` 786 MB, `tensors/` 4.9 GB,
  `mtp-q2_0.gguf` 889 MB, `mtp-q4_0.gguf` 1.6 GB (the q4_0 arm, never served), manifests/reports.
- The W4A16 checkpoint (`~/.cache/.../Qwen3.8-Flash-Next-W4A16-AutoRound`, 169 GB) and the rest of the
  `/run/media/.../strata-w4a16` working set (199 GB total) are separate, bigger reclamations that PLAN U13 already
  flagged; P7 only establishes that the `mtp-bf16` subdirectory inside it is safe.

**Recommendation: delete `.../strata-w4a16/mtp-bf16/` (8.1 GB) and keep `~/strata-xpu/mtp/` (the faithful drafter + the
canonical tensors + the rsync-able gguf).** Waiting on Mike's word.

## 9. Not validated

- One sample per arm except the serve `new` arm, which was repeated (6662.3 vs 6669.9 ms, 101/140 both times) — the 4K
  differences that matter are the draft-phase ms (7.5 ms) and the acceptance counters, not the ±0.1 tok/s.
- Arms run for this card: 4 serve arms at 4K (canonical+vocab, that arm repeated, canonical-without-vocab, W4A16-derived)
  and 4 bench arms (drafter/headless x 4K/32K).
- The headless control runs on one GPU (bench path; `--layer-split` is `--serve`-only), so its absolute tok/s is not
  comparable with the serve arms — only the drafter/headless pair is.
- One prompt family (the lane's needle prompt) and two lengths; the lookup-heavy nature of that prompt flatters the
  headless arm.
- `--spec-min-p 0.5` and `--spec 4` as in the config of record; other window settings were not swept.
- No correctness check beyond "the 150 ids of the three serve arms are identical": the bench arms' ids were not compared
  token-by-token with the serve arms'.
