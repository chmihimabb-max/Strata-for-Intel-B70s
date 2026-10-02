#!/usr/bin/env python3
"""spirv_ops.py - count the opcodes of a SPIR-V module.

PLAN.md Risk 5 asks for "the instruction-level finding if the compiler contracts it".  oneAPI ships no
spirv-dis, so the question is answered by reading the binary: a SPIR-V module is a 5-word header followed by
instructions whose first word is (word_count << 16) | opcode.

    tools/sycl/spirv_ops.py <module.spv> [--top N] [--only OPCODE[,OPCODE...]]

The opcodes that matter for Risk 5 are the dot-product family - OpSDot (4450), OpUDot (4451), OpSUDot (4452)
and the KHR variants (4457, 4458, 4459) - and the multiply/bitcast families they would replace.  Producing the
module:

    icpx -fsycl -fsycl-device-only -O3 -std=c++20 -emit-llvm -c bench/micro/sycl_dp4a_cost.cpp -o /tmp/x.bc
    sycl-post-link -O3 -device-globals -split=auto -o /tmp/t.txt /tmp/x.bc
    file-table-tform -extract=Code -drop_titles -o /tmp/c.txt /tmp/t.txt
    while read f; do llvm-spirv -o "$f.spv" "$f"; done < /tmp/c.txt
    tools/sycl/spirv_ops.py /tmp/<code>.spv.spv
"""
import sys
import struct

NAMES = {
    1: "OpUndef", 5: "OpName", 11: "OpExtInstImport", 12: "OpExtInst", 14: "OpMemoryModel",
    16: "OpEntryPoint", 17: "OpExecutionMode", 21: "OpTypeVoid", 22: "OpTypeBool", 23: "OpTypeInt",
    24: "OpTypeFloat", 25: "OpTypeVector", 27: "OpTypeFunction", 28: "OpConstantTrue", 29: "OpConstantFalse",
    30: "OpConstant", 33: "OpTypePointer", 41: "OpConstantNull", 43: "OpConstantComposite",
    46: "OpFunction", 47: "OpFunctionParameter", 48: "OpFunctionEnd", 49: "OpFunctionCall",
    54: "OpFunctionCall", 56: "OpVariable", 61: "OpLoad", 62: "OpStore", 63: "OpCopyMemory",
    65: "OpAccessChain", 67: "OpInBoundsAccessChain", 70: "OpVectorExtractDynamic",
    79: "OpVectorShuffle", 80: "OpCompositeConstruct", 81: "OpCompositeExtract", 82: "OpCompositeInsert",
    86: "OpFNegate", 88: "OpIAdd", 89: "OpFAdd", 90: "OpISub", 91: "OpFSub", 92: "OpIMul", 93: "OpFMul",
    96: "OpUDiv", 99: "OpFDiv", 100: "OpUMod", 103: "OpShiftRightLogical", 104: "OpShiftRightArithmetic",
    105: "OpShiftLeftLogical", 124: "OpBitcast", 126: "OpSNegate", 127: "OpFMod", 128: "OpSRem",
    129: "OpSMod", 135: "OpIEqual", 138: "OpFOrdEqual", 148: "OpBitwiseOr", 149: "OpBitwiseXor",
    150: "OpBitwiseAnd", 154: "OpSelect", 156: "OpSConvert", 157: "OpFConvert", 158: "OpQuantizeToF16",
    171: "OpControlBarrier", 224: "OpLabel", 245: "OpPhi", 246: "OpLoopMerge", 247: "OpSelectionMerge",
    248: "OpLabel2", 249: "OpBranch", 250: "OpBranchConditional", 251: "OpSwitch",
    253: "OpKill", 254: "OpReturn", 255: "OpReturnValue", 256: "OpUnreachable",
    260: "OpAtomicLoad", 263: "OpAtomicStore", 264: "OpAtomicExchange", 265: "OpAtomicCompareExchange",
    266: "OpAtomicCompareExchangeWeak", 267: "OpAtomicIIncrement", 268: "OpAtomicIDecrement",
    269: "OpAtomicIAdd", 270: "OpAtomicISub", 271: "OpAtomicSMin", 272: "OpAtomicUMin", 273: "OpAtomicSMax",
    274: "OpAtomicUMax", 275: "OpAtomicAnd", 276: "OpAtomicOr", 277: "OpAtomicXor",
    305: "OpGroupNonUniformBallot", 320: "OpGroupNonUniformShuffle", 321: "OpGroupNonUniformShuffleXor",
    351: "OpGroupNonUniformIAdd", 353: "OpGroupNonUniformFAdd",
    4000: "OpGroupAll", 4400: "OpGroupNonUniformElect",
    # the dot-product family - the whole point of the scan
    4450: "OpSDot", 4451: "OpUDot", 4452: "OpSUDot",
    4457: "OpSDotKHR", 4458: "OpUDotKHR", 4459: "OpSUDotKHR",
    # Intel extensions that show up around integer/vector code
    5650: "OpSubgroupBlockReadINTEL", 5651: "OpSubgroupBlockWriteINTEL",
    6084: "OpArithmeticFenceINTEL", 6085: "OpLifetimeStartINTEL",
}

DOT_FAMILY = {4450, 4451, 4452, 4457, 4458, 4459}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    top = 0
    only = None
    i = 2
    while i < len(argv):
        if argv[i] == "--top":
            top = int(argv[i + 1])
            i += 2
        elif argv[i] == "--only":
            only = {int(x) for x in argv[i + 1].split(",")}
            i += 2
        else:
            print(f"unknown argument {argv[i]}")
            return 2
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) % 4 != 0 or len(data) < 20:
        print(f"{path}: not a SPIR-V module ({len(data)} bytes)")
        return 1
    words = struct.unpack(f"<{len(data) // 4}I", data)
    magic, version, generator, bound, schema = words[:5]
    print(f"{path}: magic 0x{magic:08x} version 0x{version:08x} generator 0x{generator:08x} "
          f"bound {bound} schema {schema} words {len(words)}")
    counts = {}
    dot_iters = 0
    pos = 5
    while pos < len(words):
        w = words[pos]
        wc = w >> 16
        oc = w & 0xFFFF
        if wc == 0:
            break
        counts[oc] = counts.get(oc, 0) + 1
        if oc in DOT_FAMILY:
            dot_iters += 1
        pos += wc
    total = sum(counts.values())
    print(f"instructions: {total}; distinct opcodes: {len(counts)}")
    print(f"dot-product opcodes (OpSDot/OpUDot/OpSUDot/KHR): {sum(v for k, v in counts.items() if k in DOT_FAMILY)}")
    for oc in sorted(DOT_FAMILY):
        if oc in counts:
            print(f"  {oc} {NAMES.get(oc, '?'):<12} x{counts[oc]}")
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    if only is not None:
        items = [kv for kv in items if kv[0] in only]
    elif top:
        items = items[:top]
    else:
        items = items[:25]
    print("top opcodes:")
    for oc, n in items:
        print(f"  {oc:5d} {NAMES.get(oc, '(unmapped)'):<28} x{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
