#!/usr/bin/env bash
# Which syscall / kernel wait is each thread of the (still alive) traced engine in?
P=${1:?pid}
echo "pid $P: $(grep -E '^State|^Threads' /proc/$P/status | tr '\n' ' ')"
echo "-- threads --"
ls /proc/$P/task | while read -r tid; do
  comm=$(cat /proc/$P/task/$tid/comm 2>/dev/null)
  wchan=$(cat /proc/$P/task/$tid/wchan 2>/dev/null)
  sysno=$(awk '{print $1}' /proc/$P/task/$tid/syscall 2>/dev/null)
  echo "$tid $comm wchan=$wchan syscall=$sysno"
done
echo "-- the syscall names of interest: 202 futex 230 clock_nanosleep 232 epoll_wait 7 poll 202 futex --"
