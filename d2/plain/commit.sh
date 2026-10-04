#!/usr/bin/env bash
# D2b (t_2b6b6797): commit this card's milestone.  Explicit paths only - `i3/` and `scripts/` were untracked
# before this card and are not mine to add.
set +e
cd /home/michael/strata-xpu/strata || exit 1
git add src/core/verify.cpp src/kernels/gr_parity.cpp src/kernels/sycl/fused_gr.cpp src/kernels/cuda/fused_gr.cu
git add d2/.gitignore d2/STATUS-D2.md d2/plain
git add d2/runs/d2b-fused-4096 d2/runs/d2b-plain-4096 d2/runs/d2p-1c-fused-4096 d2/runs/d2p-1c-plain-4096 d2/runs/d2b-fused2-4096
git status --short | head -30
echo "== committing =="
git commit -q -F d2/plain/commit-msg.txt
echo "rc=$?"
git log --oneline -2
git status --short | head -10
