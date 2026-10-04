#!/usr/bin/env bash
# D2b (card t_f93760a1): commit this card.  The arm logs go in the same places D1/D2/D2a put theirs
# (d2/runs/<tag>/{err,log,out,timeline}.txt); ask.txt, drive.pid and stdin.fifo stay out of git.
set +e
SRC=/home/michael/strata-xpu/strata
cd "$SRC" || exit 1
git add include/strata/kernels/native_mmvq.hpp src/kernels/cuda/native_mmvq.cu src/program/generate.cpp
git add d2b
git add -f d2b/runs/ctest/*.log
git add d2/runs/d2b-cold-4096 d2/runs/d2b-ctloff-cold-4096 d2/runs/d2b-wu-cold-4096 \
        d2/runs/d2b-rebase-4096 d2/runs/d2b-ctloff-4096 d2/runs/d2b-wu-4096 d2/runs/d2b-wu-32768
git rm --cached -q d2/runs/d2b-*/stdin.fifo 2>/dev/null
git status --short | head -50
echo "=== staged files: $(git diff --cached --name-only | wc -l)"
git commit -q -F d2b/commit-msg-1.txt
echo "commit rc=$?"
git log --oneline -1
echo "=== files in the commit: $(git show --stat --oneline HEAD | tail -1)"
echo "=== engine binary md5: $(md5sum < build-sycl/strata | cut -c1-32)"
