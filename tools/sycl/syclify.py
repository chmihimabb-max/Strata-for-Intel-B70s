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


def kernel_name_back(text: str, end: int):
    """Start and end offsets of the kernel name immediately to the left of `end`.

    A plain `[A-Za-z0-9_:<>]*` scan is not enough: the tree launches templated kernels whose argument list
    contains a COMMA - `mmvq_multi_kernel<TY, 1><<<...>>>` (iq_kernels.cu:1303), `s_gemv_kernel<T, WARP>`
    and friends - and the naive scan stopped at the comma, produced the name `1`, and left
    `mmvq_multi_kernel<TY, ` in the output.  This walks template argument lists backwards with nesting, then
    the identifier before them.
    """
    i = end
    while i > 0 and text[i - 1] in " \t\r\n":
        i -= 1
    j = i
    while j > 0:
        c = text[j - 1]
        if c == ">":
            depth, k = 0, j
            while k > 0:
                k -= 1
                if text[k] == ">":
                    depth += 1
                elif text[k] == "<":
                    depth -= 1
                    if depth == 0:
                        break
            if k <= 0 or depth != 0:
                break
            j = k
        elif c.isalnum() or c == "_" or c == ":":
            j -= 1
        else:
            break
    return j, i


class TransformError(Exception):
    pass


def macro_continuation_spans(text: str):
    """Spans of `#define` bodies (including their backslash continuations)."""
    spans = []
    i = 0
    while True:
        j = text.find("#define", i)
        if j < 0:
            break
        k = text.rfind("\n", 0, j)
        if text[k + 1:j].strip() != "":
            i = j + 7
            continue
        end = text.find("\n", j)
        while end >= 0 and text[end - 1] == "\\":
            nxt = text.find("\n", end + 1)
            if nxt < 0:
                end = len(text)
                break
            end = nxt
        if end < 0:
            end = len(text)
        spans.append((j, end))
        i = max(end, j + 7)
    return spans


def rewrite_launches(text: str, spans, file_hash: int, rel_path: str, stats: dict,
                     const_args: str = "") -> str:
    """Rewrite every `name<<<...>>>(...)` into an immediately-invoked lambda + launcher call.

    `const_args` is the trailing argument list (leading ", " included) for the runtime-filled `__constant__`
    tables of this TU: it is appended at the host call site AND fixed into the kernel call inside the lambda,
    matching the parameter declaration `append_kernel_params` adds to the kernel's own signature.
    """
    out = []
    cursor = 0
    search = 0
    macro_spans = macro_continuation_spans(text)
    while True:
        lstart = text.find("<<<", search)
        if lstart < 0:
            break
        search = lstart + 3
        if in_spans(lstart, spans):
            continue
        in_macro = any(s <= lstart < e for s, e in macro_spans)

        # the kernel name, immediately to the left of <<<
        name_start, name_end = kernel_name_back(text, lstart)
        name = text[name_start:name_end]
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
        # The dynamic shared-memory base travels as the kernel's FIRST parameter (PLAN.md D3): the launcher
        # hands it to the kernel lambda as `_sycl_dyn`, and the kernel FUNCTION - which is where an
        # `extern __shared__` declaration lives - receives it here.  Every kernel gets it, so every launch
        # passes it, whether or not that kernel uses dynamic shared memory (nullptr when smem == 0).
        #
        # NO kernel-name tag: `launch()` submits an unnamed kernel.  A launch inside a function template or a
        # `#define` body is instantiated several times with different lambdas under one source position, and a
        # named kernel would then be defined several times under one name (measured; probe12).
        #
        # The launch SHAPE is computed once, in the host lambda body, and captured by the kernel lambda: the
        # kernel needs it for threadIdx/blockIdx/blockDim/gridDim (the launcher submits a 1-D nd_range so that
        # the sub-group equals CUDA's warp - see the note on `this_item()` in cuda_runtime.h).
        #
        # The constants arrive through the SAME parameter pack as the ordinary arguments (they are appended at
        # the host call site), so the kernel call must NOT name them a second time - that would evaluate a
        # host expression (the USM handle's .dev) inside device code.
        kwargs = "_sycl_args..." if (args_clean or const_args) else ""
        call = f"{name}(_sycl_dyn, _sycl_shape{', ' if kwargs else ''}{kwargs})"
        host_args = f"{extra}{const_args}"
        if in_macro:
            # A launch inside a `#define` body: the replacement has to stay on ONE logical line, so this form
            # carries no `#line` markers and no newlines (they would end the macro definition and break every
            # following line).  dequant_bf16.cu:206 is the site that proved it.
            stats["macro_launches"] += 1
            out.append(text[cursor:name_start])
            out.append(
                "[](auto _sycl_grid, auto _sycl_block, auto _sycl_smem, auto _sycl_stream, auto... _sycl_args) "
                "{ const ::strata::sycl_compat::launch_shape _sycl_shape = "
                "::strata::sycl_compat::shape_of(_sycl_grid, _sycl_block); "
                "::strata::sycl_compat::launch(::strata::sycl_compat::cfg(_sycl_grid, _sycl_block, _sycl_smem), "
                "_sycl_stream, [=](sycl::nd_item<1> _sycl_item, uint8_t* _sycl_dyn) { (void) _sycl_item; "
                "(void) _sycl_dyn; "
                f"{call}; }}); }}"
                f"( {grid}, {block}, {smem}, {stream}{host_args})")
            stats["launches"] += 1
            cursor = pend + 1
            search = cursor
            continue
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
            f"    const ::strata::sycl_compat::launch_shape _sycl_shape =\n"
            f"        ::strata::sycl_compat::shape_of(_sycl_grid, _sycl_block);\n"
            f"    ::strata::sycl_compat::launch(::strata::sycl_compat::cfg(_sycl_grid, _sycl_block, _sycl_smem),\n"
            f"                                   _sycl_stream,\n"
            f"        [=](sycl::nd_item<1> _sycl_item, uint8_t* _sycl_dyn) {{ (void) _sycl_item; (void) _sycl_dyn; "
            f"{call}; }});\n"
            f"}}( {grid}, {block}, {smem}, {stream}{host_args})"
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
# rule 2b: dynamic shared memory (D3) and `__constant__` tables (PLAN.md §1.3(c))
#
# `extern __shared__ T name[];`  ->  `T* name = reinterpret_cast<T*>(_sycl_dyn);`
#   The launcher already allocates the dynamic extent as a `local_accessor<uint8_t,1>` and hands the base
#   pointer to the kernel lambda as `_sycl_dyn` (cuda_runtime.h's `launch`), so the only missing piece was
#   this rewrite.  CUDA's rule that several `extern __shared__` declarations in one kernel all alias that
#   same base is preserved (every one of them starts at offset 0), and the 20 sites in the tree are all
#   inside kernel bodies.
#
# `__constant__ T name[N] = {...};`   ->  `constexpr T name[N] = {...};`
# `__constant__ T name[N][M];`        ->  a `const_storage` handle on the host + a by-value `const_ref2d`
#                                         parameter appended to every kernel in the TU and every launch site,
#                                         under the array's OWN NAME so no kernel text changes.
#   Measured in probe/probe10_const.cpp: DPC++ device code reads a constexpr global fine, and sycl::
#   device_global cannot carry a host-filled table in 2026.1 (host `.get()` throws).
# ---------------------------------------------------------------------------

EXTERN_SHARED = re.compile(r"\bextern\s+__shared__\s*((?:__align__\s*\([^)]*\)\s*)*)")
CONSTANT_START = re.compile(r"\b__constant__\s*((?:__align__\s*\([^)]*\)\s*)*)")
DECLARATOR = re.compile(r"([*&]*)\s*([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)")


def rewrite_extern_shared(text: str, spans, rel_path: str, stats: dict) -> str:
    """`extern __shared__ T name[];` -> a pointer into the launcher's dynamic local memory."""
    def expand(m: re.Match) -> str:
        attrs = m.group(1).strip()
        body_end = _shared_decl_end(text, m.end())
        if body_end < 0:
            raise TransformError(f"{rel_path}:{line_of(text, m.start())}: unterminated extern __shared__")
        body = text[m.end():body_end]
        base, declarators = split_type_and_declarators(body)
        base = (attrs + " " + base).strip()
        if not base or not declarators:
            raise TransformError(f"{rel_path}:{line_of(text, m.start())}: cannot parse extern __shared__ "
                                 f"'{body.strip()}'")
        line = line_of(text, m.start())
        after = line + text.count("\n", m.start(), body_end)
        pieces = [f'\n#line {line} "{rel_path}"\n']
        for d in split_top_commas(declarators):
            dm = DECLARATOR.fullmatch(d.strip())
            if not dm or dm.group(3) != "[]":
                raise TransformError(
                    f"{rel_path}:{line_of(text, m.start())}: extern __shared__ declarator '{d.strip()}' is "
                    f"not the empty-extent array form; the transform only handles `T name[]`")
            name = dm.group(2)
            pieces.append(f"{base}* {name} = reinterpret_cast<{base}*>(_sycl_dyn);")
            stats["extern_shared"] += 1
        pieces.append(f'\n#line {after} "{rel_path}"\n')
        return "".join(pieces)

    out, cursor = [], 0
    for m in EXTERN_SHARED.finditer(text):
        if in_spans(m.start(), spans):
            continue
        out.append(text[cursor:m.start()])
        out.append(expand(m))
        cursor = _shared_decl_end(text, m.end()) + 1
    out.append(text[cursor:])
    return "".join(out)


class ConstantPlan:
    """What the `__constant__` declarations of one translation unit became."""

    def __init__(self):
        self.runtime = []      # [{'name', 'base', 'dims': int, 'stride'}] - the runtime-filled ones

    def kernel_params(self) -> str:
        """The parameter declarations appended to every kernel in the TU."""
        out = []
        for c in self.runtime:
            if c["dims"] == 1:
                out.append(f"::strata::sycl_compat::const_ref1d<{c['base']}> {c['name']}")
            else:
                out.append(f"::strata::sycl_compat::const_ref2d<{c['base']}, {c['stride']}> {c['name']}")
        return ", ".join(out)

    def launch_args(self) -> str:
        """The argument expressions appended at every launch site (host side, evaluated per call)."""
        out = []
        for c in self.runtime:
            if c["dims"] == 1:
                out.append(f"::strata::sycl_compat::const_ref1d<{c['base']}>{{_strata_ct_{c['name']}.dev}}")
            else:
                out.append(f"::strata::sycl_compat::const_ref2d<{c['base']}, {c['stride']}>"
                           f"{{_strata_ct_{c['name']}.dev}}")
        return (", " + ", ".join(out)) if out else ""


def rewrite_constants(text: str, spans, rel_path: str, stats: dict):
    """Rewrite the `__constant__` declarations.  Returns (text, ConstantPlan)."""
    plan = ConstantPlan()

    def parse_dims(dims: str):
        parts = re.findall(r"\[([^\]]*)\]", dims)
        vals = []
        for p in parts:
            p = p.strip()
            vals.append(int(p) if p.isdigit() else p)
        return vals

    def expand(m: re.Match) -> str:
        attrs = m.group(1).strip()
        body_end = _shared_decl_end(text, m.end())
        if body_end < 0:
            raise TransformError(f"{rel_path}:{line_of(text, m.start())}: unterminated __constant__")
        body = text[m.end():body_end]
        # split the initializer off at the first '=' that is not inside brackets/parens
        depth, cut = 0, -1
        for i, ch in enumerate(body):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "=" and depth == 0:
                cut = i
                break
        decl = body[:cut] if cut >= 0 else body
        init = body[cut + 1:] if cut >= 0 else None
        base, declarators = split_type_and_declarators(decl)
        base = (attrs + " " + base).strip()
        dm = DECLARATOR.fullmatch(declarators.strip()) if declarators else None
        if not base or not dm or not dm.group(3):
            raise TransformError(f"{rel_path}:{line_of(text, m.start())}: cannot parse __constant__ "
                                 f"'{body.strip()}'")
        name = dm.group(2)
        dims = parse_dims(dm.group(3))
        line = line_of(text, m.start())
        after = line + text.count("\n", m.start(), body_end)
        if init is not None:
            # static initializer -> constexpr (measured: device code reads it, probe10)
            stats["constant_constexpr"] += 1
            return (f'\n#line {line} "{rel_path}"\n'
                    f"constexpr {base} {name}{dm.group(3).strip()} ={init};"
                    f'\n#line {after} "{rel_path}"\n')
        if len(dims) == 1:
            stride = 1
        elif len(dims) == 2:
            stride = dims[1]
        else:
            raise TransformError(
                f"{rel_path}:{line_of(text, m.start())}: a runtime-filled __constant__ with {len(dims)} "
                f"dimensions is not handled; the shim has const_ref1d/const_ref2d (PLAN.md §1.3(c))")
        if not isinstance(stride, int):
            raise TransformError(f"{rel_path}:{line_of(text, m.start())}: __constant__ {name} has a non-"
                                 f"constant inner extent '{dims[-1]}', which a device view cannot stride")
        plan.runtime.append({"name": name, "base": base, "dims": len(dims), "stride": stride})
        stats["constant_runtime"] += 1
        return (f'\n#line {line} "{rel_path}"\n'
                f"static ::strata::sycl_compat::const_storage<{base}> _strata_ct_{name};"
                f'\n#line {after} "{rel_path}"\n')

    out, cursor = [], 0
    for m in CONSTANT_START.finditer(text):
        if in_spans(m.start(), spans):
            continue
        out.append(text[cursor:m.start()])
        out.append(expand(m))
        cursor = _shared_decl_end(text, m.end()) + 1
    out.append(text[cursor:])
    text = "".join(out)

    # The host-side fill names the constant; after the declaration rewrite the name only exists as a kernel
    # parameter, so cudaMemcpyToSymbol's first argument has to move to the storage handle.
    for c in plan.runtime:
        pattern = re.compile(r"\bcudaMemcpyToSymbol\s*\(\s*" + re.escape(c["name"]) + r"\s*,")
        text, n = pattern.subn(f"cudaMemcpyToSymbol(_strata_ct_{c['name']},", text)
        stats["symbol_copies"] += n
        if n == 0:
            raise TransformError(
                f"{rel_path}: __constant__ {c['name']} has no initializer and no cudaMemcpyToSymbol site; "
                f"nothing would ever fill it")
    return text, plan


GLOBAL_START = re.compile(r"\b__global__")

# File-scope `__device__` VARIABLES with a static initializer: DPC++ refuses a non-const global in device code
# ("SYCL kernel cannot use a non-const global variable", native_mmvq.cu:420 - `__device__ __align__(4) int8_t
# iq4nl_values[16] = {...}`).  A read-only table becomes `constexpr`, exactly like a statically initialized
# `__constant__` (measured in probe10).  Only declarations that are clearly a VARIABLE with an array extent and
# an initializer are touched - a function definition (no `[` dims, or a `(` before the `=`) is left alone.
DEVICE_GLOBAL = re.compile(r"\b__device__\s+((?:__align__\s*\([^)]*\)\s*)*)")

# Attributes that sit BETWEEN `__global__` and the parameter list - recognised by kernel_params_open, which
# steps over them together with their parentheses.
GLOBAL_ATTRS = re.compile(r"\s*(?:__launch_bounds__|__align__|__attribute__)")


def kernel_params_open(text: str, after_global: int):
    """Index of the `(` that opens the parameter list of the `__global__` function at `after_global`.

    Attributes between `__global__` and the parameter list must be stepped over: `__launch_bounds__` is a
    macro that expands to nothing, so an injected parameter inside it disappears silently - which is how
    iq_kernels.cu's `__global__ void __launch_bounds__(128) mmvq_kernel(...)`, sampler.cu's
    `__global__ void __launch_bounds__(1024)\nsampler_one_block_kernel(...)` and kv_stream.cu's
    `__launch_bounds__(RT)` kept their old signature.  The attribute is recognised by the word immediately
    before the `(`, so a return type or a line break in between does not matter.
    """
    p = after_global
    while True:
        i = text.find("(", p)
        if i < 0:
            return -1
        j = i
        while j > 0 and text[j - 1] in " \t\r\n":
            j -= 1
        k = j
        while k > 0 and (text[k - 1].isalnum() or text[k - 1] == "_"):
            k -= 1
        if text[k:j] in ("__launch_bounds__", "__align__", "__attribute__"):
            close = match_delim(text, i, "(", ")")
            if close < 0:
                return -1
            p = close + 1
            continue
        return i


def rewrite_device_globals(text: str, spans, rel_path: str, stats: dict) -> str:
    """`__device__ T name[N] = {...};` (file scope) -> `constexpr T name[N] = {...};`."""
    def expand(m: re.Match) -> str:
        attrs = m.group(1).strip()
        body_end = _shared_decl_end(text, m.end())
        if body_end < 0:
            return m.group(0)
        body = text[m.end():body_end]
        if "(" in body or "=" not in body or "[" not in body:
            return m.group(0)          # a function definition or a scalar; not this rule's business
        decl, _, init = body.partition("=")
        base, declarators = split_type_and_declarators(decl)
        dm = DECLARATOR.fullmatch(declarators.strip()) if declarators else None
        if not base or not dm or not dm.group(3):
            return m.group(0)
        base = (attrs + " " + base).strip()
        line = line_of(text, m.start())
        after = line + text.count("\n", m.start(), body_end)
        stats["device_globals"] += 1
        return (f'\n#line {line} "{rel_path}"\n'
                f"constexpr {base} {dm.group(2)}{dm.group(3).strip()} ={init};"
                f'\n#line {after} "{rel_path}"\n')

    out, cursor = [], 0
    for m in DEVICE_GLOBAL.finditer(text):
        if in_spans(m.start(), spans):
            continue
        repl = expand(m)
        if repl == m.group(0):
            continue
        out.append(text[cursor:m.start()])
        out.append(repl)
        cursor = _shared_decl_end(text, m.end()) + 1
    out.append(text[cursor:])
    return "".join(out)


DEVICE_FN = re.compile(r"\b__device__\s+(?!.*\b__global__)")
COORD_MACROS = ("threadIdx", "blockIdx", "blockDim", "gridDim")


def _fn_def_extent(text: str, at: int):
    """(name, params_open, params_close, body_end) for the function definition whose `(` is at `at`."""
    close = match_delim(text, at, "(", ")")
    if close < 0:
        return None
    # the name is the identifier before the '('
    j = at
    while j > 0 and text[j - 1] in " \t\r\n":
        j -= 1
    k = j
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] == "_"):
        k -= 1
    name = text[k:j]
    brace = text.find("{", close)
    if not name or brace < 0:
        return None
    # the body must open right after the parameter list (allow an attribute/const/noexcept in between)
    between = text[close + 1:brace]
    if between.strip().strip(";").strip() not in ("", "const", "noexcept"):
        return None
    depth, i = 0, brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return name, at, close, i
        i += 1
    return None


def inject_shape_into_device_helpers(text: str, spans, rel_path: str, stats: dict) -> str:
    """`__device__` helpers that use threadIdx/blockIdx/blockDim/gridDim take the launch shape too.

    `threadIdx` and friends are macros over `_sycl_shape`, which normally lives in the KERNEL's parameter list
    (injected by inject_dyn_param).  A `__device__` helper is compiled on its own, so it needs its own copy:
    the parameter is added under the same name and every call site inside the TU passes it on.  Helpers that
    (transitively) call an affected helper are affected too, so the parameter always has a value to forward.
    Sites that only use `threadIdx.x` in a 1-D launch do not need this, but they get it anyway when they appear
    in the same function - the alternative is a second, subtler rule).
    """
    affected = set()
    for m in re.finditer(r"\b__device__\b", text):
        if in_spans(m.start(), spans) or not text.startswith("__device__", m.start()):
            continue
        p = text.find("(", m.end())
        if p < 0:
            continue
        ext = _fn_def_extent(text, p)
        if ext is None:
            continue
        name, _, _, body_end = ext
        body = text[ext[2]:body_end]
        if any(c in body for c in COORD_MACROS):
            affected.add(name)
    if not affected:
        return text
    # fixpoint: a helper that calls an affected helper needs the shape to pass on
    for _ in range(4):
        grew = False
        for m in re.finditer(r"\b__device__\b", text):
            if in_spans(m.start(), spans):
                continue
            p = text.find("(", m.end())
            if p < 0:
                continue
            ext = _fn_def_extent(text, p)
            if ext is None:
                continue
            name, _, _, body_end = ext
            if name in affected:
                continue
            body = text[ext[2]:body_end]
            if any(re.search(r"\b" + re.escape(a) + r"\s*\(", body) for a in affected):
                affected.add(name)
                grew = True
        if not grew:
            break

    # 1. add the parameter to each affected definition
    out, cursor = [], 0
    defs = []
    for m in re.finditer(r"\b__device__\b", text):
        if in_spans(m.start(), spans):
            continue
        p = text.find("(", m.end())
        if p < 0:
            continue
        ext = _fn_def_extent(text, p)
        if ext is None or ext[0] not in affected:
            continue
        defs.append(ext)
    for name, popen, pclose, _ in defs:
        inner = text[popen + 1:pclose].strip()
        repl = ("const ::strata::sycl_compat::launch_shape& _sycl_shape"
                if inner in ("", "void")
                else f"const ::strata::sycl_compat::launch_shape& _sycl_shape, {inner}")
        out.append(text[cursor:popen + 1])
        out.append(repl)
        cursor = pclose
        stats["shape_helpers"] += 1
    out.append(text[cursor:])
    text = "".join(out)
    spans = opaque_spans(text)

    # 2. pass it at every call site (the definitions were already rewritten, so they no longer match a call)
    def_starts = set()
    for m in re.finditer(r"\b__device__\b", text):
        if in_spans(m.start(), spans):
            continue
        p = text.find("(", m.end())
        if p >= 0:
            ext = _fn_def_extent(text, p)
            if ext is not None and ext[0] in affected:
                def_starts.add(ext[1])
    out, cursor = [], 0
    matches = []
    for name in affected:
        # the call may carry an explicit template argument list (`sampled_tail_warp<true>(...)`)
        call_re = r"\b" + re.escape(name) + r"\s*(?:<[^<>]*(?:<[^<>]*>[^<>]*)*>)?\s*\("
        for m in re.finditer(call_re, text):
            if in_spans(m.start(), spans) or (m.end() - 1) in def_starts:
                continue
            matches.append(m)
    matches.sort(key=lambda mm: mm.start())
    for m in matches:
        if m.start() < cursor:
            continue
        out.append(text[cursor:m.end()])
        out.append("_sycl_shape, ")
        cursor = m.end()
        stats["shape_callsites"] += 1
    out.append(text[cursor:])
    return "".join(out)


def inject_dyn_param(text: str, spans, rel_path: str, stats: dict) -> str:
    """Prepend `uint8_t* _sycl_dyn` to the parameter list of every `__global__` function (PLAN.md D3).

    The dynamic shared-memory base has to reach the KERNEL FUNCTION, not just the lambda at the launch site:
    `extern __shared__` is declared inside the kernel body, and the body is a function of its own.  Every
    kernel therefore takes it first, and every launch passes the launcher's `_sycl_dyn` there - nullptr when
    the launch asked for no dynamic shared memory, which is exactly CUDA's behaviour.
    """
    out, cursor = [], 0
    for m in GLOBAL_START.finditer(text):
        if in_spans(m.start(), spans):
            continue
        p = kernel_params_open(text, m.end())
        if p < 0:
            continue
        close = match_delim(text, p, "(", ")")
        if close < 0:
            continue
        inner = text[p + 1:close].strip()
        repl = ("uint8_t* _sycl_dyn, ::strata::sycl_compat::launch_shape _sycl_shape"
                if inner in ("", "void")
                else f"uint8_t* _sycl_dyn, ::strata::sycl_compat::launch_shape _sycl_shape, {inner}")
        out.append(text[cursor:p + 1])
        out.append(repl)
        cursor = close
        stats["kernels_dyn_param"] += 1
    out.append(text[cursor:])
    return "".join(out)


def append_kernel_params(text: str, spans, rel_path: str, params: str, stats: dict) -> str:
    """Append `params` to the parameter list of every `__global__` function in the TU.

    The constants are threaded through the kernel's own parameter list (PLAN.md §1.3(c)), so every kernel in
    the file has to receive them - including kernels that do not read the table, whose parameter is simply
    unused.  An empty parameter list (`foo()`) is replaced rather than extended, so the result is never
    `foo(, x)`.
    """
    out, cursor = [], 0
    for m in GLOBAL_START.finditer(text):
        if in_spans(m.start(), spans):
            continue
        p = kernel_params_open(text, m.end())
        if p < 0:
            continue
        close = match_delim(text, p, "(", ")")
        if close < 0:
            continue
        inner = text[p + 1:close].strip()
        repl = params if inner in ("", "void") else f"{inner}, {params}"
        out.append(text[cursor:p + 1])
        out.append(repl if inner in ("", "void") else " " + repl)
        cursor = close
        stats["kernel_params_extended"] += 1
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

REFUSALS = [
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
        # Anything the __shared__ / extern __shared__ / __constant__ patterns did not rewrite is refused here
        # rather than left in the output, where the token is undefined and would fail with an unrelated
        # message.  This is the check that makes a silently-skipped declaration impossible.
        for token, why in (("__shared__", "a __shared__ declaration the transform does not handle "
                                          "(multi-declarator or empty-dimension form)"),
                           ("__constant__", "a __constant__ declaration the transform does not handle")):
            for m in re.finditer(re.escape(token), text):
                if in_spans(m.start(), spans):
                    continue
                raise TransformError(f"{rel_path}:{line_of(text, m.start())}: {why}; rewrite it by hand or "
                                     f"extend syclify.py")


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
    # M2: glibc declares `rsqrtf` (a GNU extension, not a C standard function) so DPC++ has no device
    # wrapper for it either - a kernel calling it fails with "SYCL kernel cannot call an undefined function
    # without SYCL_EXTERNAL attribute" (elementwise.cu:108, native_gr_norm.cu:72).  `__isnanf` is the same
    # case (bits/mathcalls.h expands __isnanf from isnan): native_router.cu:70.
    ("rsqrtf", "rsqrtf_fast"),
    ("__isnanf", "isnanf_dev"),
    # M4: the PLAIN `isnan` from src/prefill/kernels.cu:43,472.  Same failure mode as `__isnanf` above - the
    # name resolves to a host-only glibc inline, so a kernel calling it fails with "SYCL kernel cannot call an
    # undefined function without SYCL_EXTERNAL attribute".  The word boundary keeps `__isnanf(` out of it (no
    # boundary between '_' and 'i'), so the two rules do not fight over the same call sites.
    ("isnan", "isnan_dev"),
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

    stats = {"launches": 0, "shared_arrays": 0, "shared_scalars": 0, "renames": 0,
             "extern_shared": 0, "constant_constexpr": 0, "constant_runtime": 0,
             "symbol_copies": 0, "kernel_params_extended": 0, "kernels_dyn_param": 0,
             "macro_launches": 0, "device_globals": 0, "shape_helpers": 0, "shape_callsites": 0}
    text, plan = rewrite_constants(original, spans, rel, stats)
    text = rewrite_launches(text, opaque_spans(text), fnv1a(rel), rel, stats,
                            const_args=plan.launch_args())
    text = rewrite_renames(text, opaque_spans(text), stats)
    text = rewrite_device_globals(text, opaque_spans(text), rel, stats)
    text = rewrite_extern_shared(text, opaque_spans(text), rel, stats)   # before the static __shared__ rule:
    text = rewrite_shared(text, opaque_spans(text), rel, stats)          # its pattern also matches `__shared__`
    text = inject_dyn_param(text, opaque_spans(text), rel, stats)
    text = inject_shape_into_device_helpers(text, opaque_spans(text), rel, stats)
    if plan.runtime:
        text = append_kernel_params(text, opaque_spans(text), rel, plan.kernel_params(), stats)
    check_refusals(text, opaque_spans(text), rel, post_shared=True)

    header = (
        f"// GENERATED by tools/sycl/syclify.py from {rel} - DO NOT EDIT.\n"
        f"// The source is the .cu; this file is a mechanical translation (see the tool's docstring).\n"
        f"// rewritten: {stats['launches']} launch site(s), {stats['shared_arrays']} shared array(s), "
        f"{stats['shared_scalars']} shared scalar(s), {stats['renames']} fast-math rename(s), "
        f"{stats['extern_shared']} extern __shared__ base(s), "
        f"{stats['constant_constexpr']} static __constant__ -> constexpr, "
        f"{stats['constant_runtime']} runtime __constant__ threaded as a kernel parameter "
        f"({stats['kernel_params_extended']} kernel signature(s) extended, "
        f"{stats['symbol_copies']} cudaMemcpyToSymbol site(s))\n"
        f"#line 1 \"{rel}\"\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(text)

    if not a.quiet:
        print(f"syclify: {rel} -> {os.path.relpath(a.output, repo_root)} "
              f"({stats['launches']} launches, {stats['shared_arrays']} shared arrays, "
              f"{stats['shared_scalars']} shared scalars, {stats['renames']} renames, "
              f"{stats['extern_shared']} extern shared, "
              f"{stats['constant_constexpr']}+{stats['constant_runtime']} constants)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TransformError as e:
        sys.stderr.write(f"syclify: REFUSED: {e}\n")
        sys.exit(1)
