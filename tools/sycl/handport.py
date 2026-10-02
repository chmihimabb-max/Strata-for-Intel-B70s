#!/usr/bin/env python3
"""tools/sycl/handport.py - build a hand-ported SYCL translation unit for one of the 5 asm files.

PLAN.md §2.3 names 16 inline-PTX sites in 5 files; PLAN.md D1 says those files are NOT translated by
tools/sycl/syclify.py (it refuses any file containing `asm`) but are hand-ported and committed at
src/kernels/sycl/<name>.cpp, with tools/sycl/exceptions.txt as the single list.

How this tool makes that reproducible.  The hand port is expressed as an explicit, reviewable list of
text replacements over the UNMODIFIED .cu (tools/sycl/handport/<name>.py), and the result is then run through
the *same* transform every other file gets.  That is deliberate:

  * the launch/`__shared__`/`__constant__`/shape-injection rules of the transform are the ones M1-M2 proved,
    and hand-copying 1000 lines of kernel body would re-introduce every transcription bug the transform
    already fixed;
  * every replacement is a single exact occurrence (`REPLACEMENTS` is checked for that), so a change in the .cu
    fails this tool loudly instead of silently drifting;
  * the committed output at src/kernels/sycl/<name>.cpp is what CMake builds - nothing is generated at build
    time, so the build cannot disagree with the reviewed file.

Usage:
  tools/sycl/handport.py qsa_prompt_attn                    # writes src/kernels/sycl/qsa_prompt_attn.cpp
  tools/sycl/handport.py --all
  tools/sycl/handport.py --check qsa_prompt_attn            # regenerate + diff against the committed file

The patched intermediate is written under <repo>/../syclhand/<relpath> (never /tmp), so the generated #line
markers still name src/kernels/cuda/<name>.cu with the ORIGINAL line numbers.
"""
import argparse
import difflib
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SPEC_DIR = os.path.join(HERE, "handport")
MIRROR_ROOT = os.path.abspath(os.path.join(REPO, "..", "syclhand"))
OUT_DIR = os.path.join(REPO, "src", "kernels", "sycl")

# the 5 hand ports, in PLAN.md §2.3 order; the .cu basename is what exceptions.txt lists
HAND_PORTS = ["qsa_select", "qsa_prompt_attn", "native_qsa_score", "fused_gr", "verify_kernels"]

BANNER = """// HAND PORT of src/kernels/cuda/{name}.cu for the SYCL backend - PLAN.md §2.3, Risk 7.  DO NOT EDIT: rebuild with
//   tools/sycl/handport.py {name}
// from the UNMODIFIED .cu plus the replacement list in tools/sycl/handport/{name}.py.  The replacements remove
// the {nasms} inline-PTX site(s) this file carries; everything else is the mechanical translation of the same
// transform (tools/sycl/syclify.py) the other 50-5 files go through at build time.
//
// WHY THIS FILE EXISTS AT ALL: the guard the CUDA sources use,
//   #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// is TRUE when __CUDA_ARCH__ is undefined - which is exactly the case under SYCL - so a naive shim would compile
// CUDA asm into the device image (PLAN.md Risk 7).  Measured for this port: the generator refuses to emit a file
// with even one surviving asm token, so this file has none (plan-evidence/M3-risk7-asm.txt counts them over the
// whole generated set).
"""


def load_spec(name):
    path = os.path.join(SPEC_DIR, name + ".py")
    if not os.path.exists(path):
        sys.exit(f"handport: no spec {path}")
    spec = importlib.util.spec_from_file_location("hp_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_replacements(text, repls, name):
    for i, (old, new) in enumerate(repls):
        n = text.count(old)
        if n != 1:
            sys.exit(f"handport: {name} replacement #{i} matches {n} times, expected exactly 1:\n"
                     f"---- first 200 chars ----\n{old[:200]}\n-------------------------")
        text = text.replace(old, new, 1)
    return text


def spec_load(name):
    return load_spec(name)


def build(name, quiet=False):
    """Return (patched_path, generated_text)."""
    spec_mod = spec_load(name)
    cu = os.path.join(REPO, "src", "kernels", "cuda", name + ".cu")
    if not os.path.exists(cu):
        sys.exit(f"handport: no such source {cu}")
    with open(cu, "r", encoding="utf-8") as fh:
        original = fh.read()
    patched = apply_replacements(original, spec_mod.REPLACEMENTS, name)

    # the mirrored intermediate keeps the ORIGINAL relative path so the #line markers are exact
    rel = os.path.relpath(cu, REPO)
    mirror = os.path.join(MIRROR_ROOT, rel)
    os.makedirs(os.path.dirname(mirror), exist_ok=True)
    with open(mirror, "w", encoding="utf-8") as fh:
        fh.write(patched)

    out = os.path.join(OUT_DIR, name + ".cpp")
    sys.path.insert(0, HERE)
    import syclify  # noqa: E402

    rc = syclify.main([mirror, out, "--exceptions", "", "--repo-root", MIRROR_ROOT, "--quiet"])
    if rc != 0:
        sys.exit(f"handport: syclify refused the patched {name}")
    with open(out, "r", encoding="utf-8") as fh:
        body = fh.read()

    nasms = len(re.findall(r"(__asm__|\basm\s*\(|asm\s+volatile)", body))
    banner = BANNER.format(name=name, nasms=spec_mod.ASM_SITES)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(banner)
        fh.write(body)
    if not quiet:
        print(f"handport: {name}.cu -> src/kernels/sycl/{name}.cpp "
              f"(replaced {len(spec_mod.REPLACEMENTS)} region(s), {spec_mod.ASM_SITES} asm site(s); "
              f"asm tokens left in the output: {nasms})")
    if nasms:
        sys.exit(f"handport: {name}: {nasms} asm token(s) survived - Risk 7 not satisfied")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the hand-ported SYCL TUs (PLAN.md §2.3)")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true", help="regenerate and diff against the committed file")
    a = ap.parse_args(argv)
    names = HAND_PORTS if a.all or not a.names else a.names
    rc = 0
    for n in names:
        if a.check:
            out = os.path.join(OUT_DIR, n + ".cpp")
            before = open(out, "r", encoding="utf-8").read() if os.path.exists(out) else ""
            build(n, quiet=True)
            after = open(out, "r", encoding="utf-8").read()
            if before == after:
                print(f"CHECK OK   {n}")
            else:
                rc = 1
                print(f"CHECK DIFF {n}")
                sys.stdout.writelines(list(difflib.unified_diff(
                    before.splitlines(True), after.splitlines(True), "committed", "regenerated"))[:40])
        else:
            build(n)
    return rc


if __name__ == "__main__":
    sys.exit(main())
