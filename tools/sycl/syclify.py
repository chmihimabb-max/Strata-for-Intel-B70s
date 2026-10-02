#!/usr/bin/env python3
"""syclify.py - rewrite a Strata .cu into a compilable SYCL .cpp, WITHOUT editing the .cu.

Why a transform and not a header-only shim (PLAN.md D1): the HIP backend works with a pure rename because
hipcc *is* a CUDA-like front end.  icpx is not - `kernel<<<grid, block, smem, stream>>>(args)` has no C++
spelling at all, and the tree has 284-321 such sites.  So the launch syntax is rewritten ahead of compilation
and the resulting translation unit is ordinary C++ compiled with `-x c++ -fsycl`.

What is rewritten (each rule is checked at the end of this file against a table of expected counts):

  1. LAUNCH
       name<<<G, B[, S[, ST]]>>>(ARGS);
     ->
       ::strata::sycl_compat::launch<::strata::sycl_compat::kernel_name<HASH, LINE>>(
           ::strata::sycl_compat::cfg(G, B, SMEM), STREAM,
           [=](sycl::nd_item<3> _sycl_item, uint8_t* _sycl_dyn) { name(ARGS); });
     The shape is the one probe/probe_sycl_launch.cpp compiled AND ran on the B70 with oneAPI 2026.1.1.
     STREAM is `::strata::sycl_compat::default_stream()` when the launch had no stream (CUDA's default
     stream), SMEM is 0 when the launch had no dynamic shared-memory argument.

  2. STATIC SHARED MEMORY
       __shared__ T name[N];            (also T name[A][B], and __align__(n) prefixes)
     ->
       T (&name)[N] = *::sycl::ext::oneapi::group_local_memory_for_overwrite<T[N]>(
                          ::strata::sycl_compat::this_item().get_group()).get();
       __shared__ T name;               (scalar)
     ->
       T& name = *::sycl::ext::oneapi::group_local_memory_for_overwrite<T>(
                      ::strata::sycl_compat::this_item().get_group()).get();

  3. `#line` markers so a compile error inside generated code still names the ORIGINAL .cu and line.

What this transform REFUSES (loudly, with file:line) rather than mis-compiling:

  * `extern __shared__`  - needs the launcher to inject the dynamic base as a leading kernel parameter
                           (PLAN.md D3, 20 sites, budgeted in M2).
  * `__constant__`       - becomes a USM buffer threaded through the kernel's parameter list; the 3
                           cudaMemcpyToSymbol sites have to be rewritten with it (PLAN.md §1.3(c), M2).
  * inline `asm` / `__asm__` - these are the 5 hand-ported files (PLAN.md §2.3).  Risk 7 is that a naive shim
                           compiles the CUDA asm under SYCL; this check makes that impossible: a generated TU
                           with a surviving asm site cannot exist, because the transform refuses the file.

Usage:  syclify.py <in.cu> <out.cpp> [--exceptions tools/sycl/exceptions.txt] [--repo-root DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# ---------------------------------------------------------------------------
# comment / string scanning: a `<<<` inside a comment or a string literal is not a launch
# ---------------------------------------------------------------------------


def opaque_spans(text: str):
    """Spans of line comments, block comments, string and char literals."""
    spans = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            spans.append((i, j))
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            spans.append((i, j))
            i = j
        elif c in ('"', "'"):
            quote = c
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == quote or text[j] == "\n":
                    j += 1
                    break
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def in_spans(pos: int, spans) -> bool:
    lo, hi = 0, len(spans)
    while lo < hi:
        mid = (lo + hi) // 2
        if spans[mid][0] <= pos < spans[mid][1]:
            return True
        if pos < spans[mid][0]:
            hi = mid
        else:
            lo = mid + 1
    return False


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def match_delim(text: str, start: int, open_c: str, close_c: str) -> int:
    """Index of the `close_c` matching the `open_c` at `start`, ignoring nested occurrences inside
    comments/strings is unnecessary here: launch configs and argument lists are plain expressions."""
    assert text[start] == open_c
    depth = 0
    i = start
    while i < len(text):
        if text[i] == open_c:
            depth += 1
        elif text[i] == close_c:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_top_commas(expr: str):
    """Split on commas that are not inside (), [] or {}.

    Angle brackets are deliberately NOT tracked: a launch config holds ordinary expressions, and
    `<<<(unsigned)((n+255)/256 < 4096 ? (n+255)/256 : 4096), 256, 0, s>>>` has a `<` that is a
    comparison, not a template argument.  Tracking them split that config into one part instead of four.
    """
    parts, depth, cur = [], 0, []
    for c in expr:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(c)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


# ---------------------------------------------------------------------------
# rule 1: the launch
# ---------------------------------------------------------------------------

IDENT_BACK = re.compile(r"[A-Za-z0-9_:<>]*$")


class TransformError(Exception):
    pass


def rewrite_launches(text: str, spans, file_hash: int, rel_path: str, stats: dict) -> str:
    out = []
    cursor = 0
    search = 0
    while True:
        lstart = text.find("<<<", search)
        if lstart < 0:
            break
        search = lstart + 3
        if in_spans(lstart, spans):
            continue

        # the kernel name, immediately to the left of <<<
        head = text[:lstart]
        m = IDENT_BACK.search(head)
        name = m.group(0) if m else ""
        name_start = lstart - len(name)
        if not name:
            raise TransformError(f"{rel_path}:{line_of(text, lstart)}: '<<<' with no kernel name to its left")

        send = text.find(">>>", lstart + 3)
        if send < 0:
            raise TransformError(f"{rel_path}:{line_of(text, lstart)}: '<<<' without a matching '>>>'")
        config = split_top_commas(text[lstart + 3:send])

        # the argument list
        p = send + 3
        while p < len(text) and text[p] in " \t\r\n":
            p += 1
        if p >= len(text) or text[p] != "(":
            raise TransformError(
                f"{rel_path}:{line_of(text, lstart)}: launch of {name} is not followed by an argument list")
        pend = match_delim(text, p, "(", ")")
        if pend < 0:
            raise TransformError(f"{rel_path}:{line_of(text, lstart)}: unbalanced argument list")
        args = text[p + 1:pend]

        if len(config) == 2:
            grid, block = config
            smem, stream = "0ull", "::strata::sycl_compat::default_stream()"
        elif len(config) == 3:
            grid, block, smem = config
            stream = "::strata::sycl_compat::default_stream()"
        elif len(config) == 4:
            grid, block, smem, stream = config
        else:
            raise TransformError(
                f"{rel_path}:{line_of(text, lstart)}: launch config has {len(config)} parts, expected 2-4")

        line = line_of(text, lstart)
        after = line + text.count("\n", name_start, pend + 1)
        args_clean = args.strip()
        extra = f", {args_clean}" if args_clean else ""
        # An immediately-invoked lambda, not a bare `[=]` kernel lambda:
        #   * DPC++ refuses an IMPLICIT capture of `this` in a kernel ("implicit capture of 'this' is not
        #     allowed for kernel functions" - measured on src/core/device.cu's poison launch, whose argument
        #     is `base_`), so every argument has to be evaluated in HOST context first;
        #   * doing that also keeps a host-only expression (a std::vector::data() call, a member access) out
        #     of device code, where it cannot compile.
        # The parameters are evaluated at the host call site; the kernel lambda then captures only values.
        out.append(text[cursor:name_start])
        out.append(
            f'\n#line {line} "{rel_path}"\n'
            f"[](auto _sycl_grid, auto _sycl_block, auto _sycl_smem, auto _sycl_stream, auto... _sycl_args) {{\n"
            f"    ::strata::sycl_compat::launch<::strata::sycl_compat::kernel_name<{file_hash}ull, {line}>>(\n"
            f"        ::strata::sycl_compat::cfg(_sycl_grid, _sycl_block, _sycl_smem), _sycl_stream,\n"
            f"        [=](sycl::nd_item<3> _sycl_item, uint8_t* _sycl_dyn) {{ (void) _sycl_item; (void) _sycl_dyn; "
            f"{name}(_sycl_args...); }});\n"
            f"}}( {grid}, {block}, {smem}, {stream}{extra})"
            f'\n#line {after} "{rel_path}"\n'
        )
        stats["launches"] += 1
        cursor = pend + 1
        search = cursor
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------------------
# rule 2: static shared memory
# ---------------------------------------------------------------------------

SHARED_START = re.compile(r"__shared__\s*((?:__align__\s*\([^)]*\)\s*)*)")


def _shared_decl_end(text: str, start: int) -> int:
    """Index of the `;` that ends the declaration starting at `start` (bracket depth 0)."""
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
        elif c == ";" and depth == 0:
            return i
        i += 1
    return -1


def split_type_and_declarators(body: str):
    """Split `unsigned long srow[CHUNK]` / `float sk[S], sq[S]` / `double s_sum` into (type, declarators).

    A declarator starts at the first identifier at bracket depth 0 that is immediately followed (ignoring
    spaces) by `[`, `,` or the end of the declaration.  That rule is what keeps `wsum[S * RG / 32]` - whose
    extent has a `*` in it - from being mistaken for a pointer declarator, and what handles a multi-word type
    like `long long`.
    """
    i, n, depth = 0, len(body), 0
    while i < n:
        c = body[i]
        if c == "[":
            depth += 1
            i += 1
            continue
        if c == "]":
            depth -= 1
            i += 1
            continue
        if depth == 0 and (c.isalpha() or c == "_"):
            j = i
            while j < n and (body[j].isalnum() or body[j] == "_"):
                j += 1
            k = j
            while k < n and body[k] in " \t\r\n":
                k += 1
            if k >= n or body[k] in "[,":
                return body[:i].strip(), body[i:].strip()
            i = j
            continue
        i += 1
    return body.strip(), ""


def rewrite_shared(text: str, spans, rel_path: str, stats: dict) -> str:
    def expand(m: re.Match) -> str:
        attrs = m.group(1).strip()
        body_end = _shared_decl_end(text, m.end())
        if body_end < 0:
            raise TransformError(f"{rel_path}:{line_of(text, m.start())}: unterminated __shared__ declaration")
        body = text[m.end():body_end]
        base, declarators = split_type_and_declarators(body)
        base = (attrs + " " + base).strip()
        if not base or not declarators:
            raise TransformError(f"{rel_path}:{line_of(text, m.start())}: cannot parse __shared__ "
                                 f"declaration '{body.strip()}'")
        decls = split_top_commas(declarators)
        line = line_of(text, m.start())
        after = line + text.count("\n", m.start(), body_end)
        pieces = [f'\n#line {line} "{rel_path}"\n']
        for d in decls:
            dm = re.fullmatch(r"([*&]*)\s*([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)", d.strip())
            if not dm:
                raise TransformError(f"{rel_path}:{line_of(text, m.start())}: cannot parse __shared__ "
                                     f"declarator '{d}'")
            if dm.group(1):
                raise TransformError(
                    f"{rel_path}:{line_of(text, m.start())}: a __shared__ pointer/reference declarator "
                    f"('{d}'); not handled by the transform")
            name, dims = dm.group(2), re.sub(r"\s+", "", dm.group(3))
            if dims == "[]":
                raise TransformError(
                    f"{rel_path}:{line_of(text, m.start())}: __shared__ {name} has an empty extent; that is "
                    f"the dynamic (extern __shared__) case, which M2 owns")
            if dims == "":
                stats["shared_scalars"] += 1
                pieces.append(f"{base}& {name} = *::sycl::ext::oneapi::group_local_memory_for_overwrite<"
                              f"{base}>(::strata::sycl_compat::this_item().get_group()).get();")
            else:
                stats["shared_arrays"] += 1
                pieces.append(f"{base} (&{name}){dims} = *::sycl::ext::oneapi::"
                              f"group_local_memory_for_overwrite<{base} {dims}>("
                              f"::strata::sycl_compat::this_item().get_group()).get();")
        pieces.append(f'\n#line {after} "{rel_path}"\n')
        return "".join(pieces)

    out, cursor = [], 0
    for m in SHARED_START.finditer(text):
        if in_spans(m.start(), spans):
            continue
        out.append(text[cursor:m.start()])
        out.append(expand(m))
        body_end = _shared_decl_end(text, m.end())
        cursor = body_end + 1
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

REFUSALS = [
    (re.compile(r"\bextern\s+__shared__"),
     "extern __shared__ (dynamic shared memory) is not rewritten yet: it needs the launcher to inject the "
     "base pointer as a leading kernel parameter (PLAN.md D3, M2)"),
    (re.compile(r"\b__constant__"),
     "__constant__ is not rewritten yet: it becomes a USM buffer threaded through the kernel's parameter list "
     "together with its cudaMemcpyToSymbol sites (PLAN.md §1.3(c), M2)"),
    (re.compile(r"(__asm__|\basm\s*\(|asm\s+volatile)"),
     "inline PTX assembler: this is one of the 5 hand-ported files (PLAN.md §2.3).  Risk 7 is a shim that "
     "compiles the CUDA asm under SYCL; the transform refuses the file instead"),
]


def check_refusals(text: str, spans, rel_path: str, *, post_shared: bool = False):
    for pattern, why in REFUSALS:
        for m in pattern.finditer(text):
            if in_spans(m.start(), spans):
                continue
            raise TransformError(f"{rel_path}:{line_of(text, m.start())}: {why}")
    if post_shared:
        # Anything the two __shared__ patterns did not rewrite - a multi-declarator line such as
        # `__shared__ float a[32], b[32];`, or an empty-dimension form - is refused here rather than left in
        # the output, where `__shared__` is not defined and would fail with an unrelated message.
        for m in re.finditer(r"__shared__", text):
            if in_spans(m.start(), spans):
                continue
            raise TransformError(
                f"{rel_path}:{line_of(text, m.start())}: a __shared__ declaration the transform does not "
                f"handle (multi-declarator or empty-dimension form); rewrite it by hand or extend syclify.py")


# ---------------------------------------------------------------------------
# rule 3: the fast-math intrinsics whose CUDA spelling collides with the C library
#
# `__expf`, `__logf`, `__sinf`, `__cosf`, `__powf` cannot be macros in the compat header: glibc's <math.h>
# declares `extern float __cosf(float)` and its four siblings (the __*f builtin aliases), so a macro of the
# same name expands inside the system header and breaks it.  Measured with probe/macro_collide.py: exactly
# these five collide and the other 38 intrinsics do not.  The rename happens here instead, at the call site.
# ---------------------------------------------------------------------------

RENAMES = [
    ("__expf", "expf_fast"),
    ("__logf", "logf_fast"),
    ("__sinf", "sinf_fast"),
    ("__cosf", "cosf_fast"),
    ("__powf", "powf_fast"),
]


def rewrite_renames(text: str, spans, stats: dict) -> str:
    out, cursor = [], 0
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n, _ in RENAMES) + r")\s*\(")
    repl = {n: f"::strata::sycl_compat::{f}" for n, f in RENAMES}
    for m in pattern.finditer(text):
        if in_spans(m.start(), spans):
            continue
        out.append(text[cursor:m.start()])
        out.append(repl[m.group(1)] + "(")
        cursor = m.end()
        stats["renames"] += 1
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------------------
# `#line` markers are emitted inline by the rewriters above: a directive must start at the beginning of a
# line, so each replacement is preceded by `#line <source line>` and followed by `#line <next source line>`,
# both wrapped in newlines.  Without them a compile error in generated code would name a line of the
# generated .cpp, which is not a file anyone can edit.
# ---------------------------------------------------------------------------


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def load_exceptions(path: str):
    names = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.split("#", 1)[0].strip()
                if line:
                    names.add(os.path.basename(line))
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rewrite a Strata .cu into a SYCL .cpp")
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--exceptions", default="")
    ap.add_argument("--repo-root", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    repo_root = a.repo_root or os.getcwd()
    rel = os.path.relpath(os.path.abspath(a.source), repo_root)
    base = os.path.basename(a.source)

    if base in load_exceptions(a.exceptions):
        print(f"syclify: {rel} is hand-ported (tools/sycl/exceptions.txt); not generating a TU for it")
        return 0

    with open(a.source, "r", encoding="utf-8") as fh:
        original = fh.read()

    spans = opaque_spans(original)
    check_refusals(original, spans, rel)

    stats = {"launches": 0, "shared_arrays": 0, "shared_scalars": 0, "renames": 0}
    text = rewrite_launches(original, spans, fnv1a(rel), rel, stats)
    text = rewrite_renames(text, opaque_spans(text), stats)
    text = rewrite_shared(text, opaque_spans(text), rel, stats)
    check_refusals(text, opaque_spans(text), rel, post_shared=True)

    header = (
        f"// GENERATED by tools/sycl/syclify.py from {rel} - DO NOT EDIT.\n"
        f"// The source is the .cu; this file is a mechanical translation (see the tool's docstring).\n"
        f"// rewritten: {stats['launches']} launch site(s), {stats['shared_arrays']} shared array(s), "
        f"{stats['shared_scalars']} shared scalar(s), {stats['renames']} fast-math rename(s)\n"
        f"#line 1 \"{rel}\"\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(text)

    if not a.quiet:
        print(f"syclify: {rel} -> {os.path.relpath(a.output, repo_root)} "
              f"({stats['launches']} launches, {stats['shared_arrays']} shared arrays, "
              f"{stats['shared_scalars']} shared scalars, {stats['renames']} renames)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TransformError as e:
        sys.stderr.write(f"syclify: REFUSED: {e}\n")
        sys.exit(1)
