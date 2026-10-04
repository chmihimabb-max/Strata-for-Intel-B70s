#!/usr/bin/env python3
"""D2b (card t_2b6b6797): read two arms' per-layer residual ladders and name the first divergence.

`STRATA_DUMP_LADDER` (src/core/verify.cpp) writes, for the FIRST decode window:
    int32 hdr[5] = { n_entries, hc, n_embd, pos0, T }  then n_entries * max_t_ * hc * n_embd floats
Entries are strided by max_t_ (the window buffer's width), which is NOT in the header - it is recovered from
the file size.  Entry 0 is the R the stage started from; entry k is the R ENTERING stage-local layer k-1,
i.e. the output of the layer before it, so entry l+1 = R after stage-local layer l-1.  The last entry is the
stage's final residual (the text at verify.cpp:1182 writes slot n_layers - lb_ + 1).

A layer split runs one Verifier PER STAGE in one process and each writes the ladder, so the second stage used
to clobber the first's file; with STRATA_DUMP_LADDER_PERSTAGE=1 the stage's first layer is appended to the
name (`<path>.lb<lb_>`).  Both stages are merged here by layer index.

    d2b_ladder.py LADDER_A LADDER_B      # both may be an unstaged path or the `<path>.lbNN` glob parent
"""
import sys
import glob
import numpy as np

HC, N_EMBD, N_LAYERS = 4, 2560, 48


def read_ladder(path):
    raw = open(path, "rb").read()
    n_entries, hc, n_embd, pos0, T = np.frombuffer(raw[:20], dtype="<i4")
    body = np.frombuffer(raw[20:], dtype="<f4")
    stride = body.size // int(n_entries)
    if stride * int(n_entries) != body.size:
        sys.exit("%s: %d floats is not %d entries of a whole stride" % (path, body.size, n_entries))
    per = int(T) * int(hc) * int(n_embd)
    assert (hc, n_embd) == (HC, N_EMBD), "unexpected geometry %d/%d" % (hc, n_embd)
    ent = body.reshape(int(n_entries), stride)[:, :per].reshape(int(n_entries), int(T), HC, N_EMBD)
    # the stage's own first layer: n_entries = n_layers - lb + 2
    lb = N_LAYERS + 2 - int(n_entries)
    return {"path": path, "T": int(T), "pos0": int(pos0), "lb": lb, "R": ent.astype(np.float64)}


def stage_files(path):
    """the per-stage files of one arm: <path>.lb0, <path>.lb24, ... if present, else the single path
    (the `<path>.lbNN.bo` companion is a different layout and is read separately)"""
    got = [f for f in sorted(glob.glob(path + ".lb*")) if not f.endswith(".bo")]
    return got if got else [path]


def stage_bos(path):
    return sorted(glob.glob(path + ".lb*.bo"))


def read_bo(path):
    """the `.bo` companion: int32[5] = { nb, 1, n_embd, pos0, T }, then nb * T * n_embd floats.
    Slot k is layer (lb_ + k) - the header carries no lb_, so it is recovered from nb (nb = n_layers - lb_)."""
    raw = open(path, "rb").read()
    nb, one, n_embd, pos0, T = np.frombuffer(raw[:20], dtype="<i4")
    lb = N_LAYERS - int(nb)
    body = np.frombuffer(raw[20:], dtype="<f4")
    per = int(T) * int(n_embd)
    if body.size != int(nb) * per:
        sys.exit("%s: %d floats is not %d rows x %d (%d x %d)" % (path, body.size, nb, per, T, n_embd))
    rows = body.reshape(int(nb), int(T), int(n_embd))
    return {lb + k: (rows[k, 0].astype(np.float64), "%s layer %d" % (path.split("/")[-1], lb + k))
            for k in range(int(nb))}


def merge(path):
    """layer index -> (label, R[T, HC, N]) for one arm, merged across stages"""
    out = {}
    for f in stage_files(path):
        L = read_ladder(f)
        lb, R = L["lb"], L["R"]
        n = R.shape[0]
        # slot 0 = the stage's input R; slot k>=1 = R entering global layer lb + k - 1
        for k in range(n):
            if not np.any(R[k]):
                continue                       # the unstaged tail of a stage's buffer is not data
            if k == 0:
                key, label = -1, "initial R (stage input)"
            else:
                key, label = lb + k - 2, "R after layer %d" % (lb + k - 2)
            if key not in out:
                out[key] = (label + "  [%s, lb=%d slot %d]" % (f.split("/")[-1], lb, k), R[k])
    return out


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    A, B = merge(a_path), merge(b_path)
    keys = sorted(set(A) & set(B))
    print("layers present in both arms: %d (%s)" % (len(keys), ", ".join("%d" % k for k in keys[:3]) + " ..."))
    first = None
    print("\n  after layer      A rms      B rms    rel_rms(A,B)   max|diff|   rel_rms(max)  differing/200*T")
    for k in keys:
        label, ra = A[k]
        _, rb = B[k]
        d = ra - rb
        na, nb = np.linalg.norm(ra), np.linalg.norm(rb)
        rel = np.linalg.norm(d) / (na if na > 0 else 1e-30)
        mx = np.abs(d).max()
        rms = np.sqrt((ra ** 2).mean())
        # per-channel worst relative, to expose a single bad stream
        ndiff = int((ra.reshape(-1) != rb.reshape(-1)).sum())
        ntot = ra.size
        print("  %-15s %.6e %.6e   %.3e      %.3e     %.3e       %d/%d"
              % (("%d  %s" % (k, label.split("  [")[0])) if k >= 0 else label,
                 np.sqrt((ra ** 2).mean()), np.sqrt((rb ** 2).mean()), rel, mx, mx / (rms if rms else 1e-30),
                 ndiff, ntot))
        if first is None and ndiff > 0:
            first = (k, label, rel, mx, rms)
    if first is None:
        print("\nVERDICT: the two arms' ladders are bit-identical at every shared layer")
    else:
        k, label, rel, mx, rms = first
        print("\nFIRST DIVERGENCE: layer %d (%s)" % (k, label))
        print("  rel_rms %.3e, max|diff| %.3e, layer rms %.3e (max/rms %.3e)" % (rel, mx, rms, mx / (rms or 1e-30)))

    # ---- the attention-half ladder (`.bo`): a different tensor class at the same layers
    bo_a, bo_b = {}, {}
    for f in stage_bos(a_path):
        bo_a.update(read_bo(f))
    for f in stage_bos(b_path):
        bo_b.update(read_bo(f))
    if bo_a and bo_b:
        keys = sorted(set(bo_a) & set(bo_b))
        print("\n  attention-half output (`.bo`, slot == layer)   %d layers in both arms" % len(keys))
        print("  layer      A rms      B rms    rel_rms(A,B)   max|diff|  differing/2560")
        firstb = None
        for k in keys:
            ra, _ = bo_a[k]
            rb, _ = bo_b[k]
            d = ra - rb
            na = np.linalg.norm(ra)
            ndiff = int((ra != rb).sum())
            print("  %-8d %.6e %.6e   %.3e      %.3e    %d"
                  % (k, np.sqrt((ra ** 2).mean()), np.sqrt((rb ** 2).mean()),
                     np.linalg.norm(d) / (na if na > 0 else 1e-30), np.abs(d).max(), ndiff))
            if firstb is None and ndiff > 0:
                firstb = k
        print("FIRST `.bo` DIVERGENCE: layer %s" % ("none (bit-identical)" if firstb is None else firstb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
