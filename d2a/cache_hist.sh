#!/bin/bash
# D2a forensics: when did the SYCL program cache get new entries?
C=/home/michael/strata-xpu/sycl-cache/m6c/13665086051514919768
echo "cache dir entries: $(ls $C | wc -l)"
find $C -maxdepth 1 -mindepth 1 -printf '%TY-%Tm-%Td %TH:%TM:%TS %f\n' | sort | awk '{print $1, $2}' | uniq -c | tail -40
