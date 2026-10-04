#!/usr/bin/env bash
# D2b (t_2b6b6797): the ladder/window-state dumps are ~1-2 MB each per arm and are regenerable from the arm
# command in d2/runs/<tag>/log.txt, so they stay on disk and out of the commit (like the other run scratch).
set +e
R=/home/michael/strata-xpu/strata
rm -f "$R/d2/plain/raw-ladder-fused-2c.bin" "$R/d2/plain/raw-ladder-fused-2c.bin.bo"
rm -f "$R/d2/plain/raw-ladder-plain-2c.bin" "$R/d2/plain/raw-ladder-plain-2c.bin.bo"
rm -f "$R/d2/plain/raw-ladder-fused-1c.bin" "$R/d2/plain/raw-ladder-fused-1c.bin.bo"
rm -f "$R/d2/plain/raw-ladder-plain-1c.bin" "$R/d2/plain/raw-ladder-plain-1c.bin.bo"
rm -f "$R/d2/plain/raw-state-fused-2c.bin" "$R/d2/plain/raw-state-plain-2c.bin"
rm -f "$R/d2/plain/raw-state-fused-1c.bin" "$R/d2/plain/raw-state-plain-1c.bin"
ls -la "$R/d2/plain/"
