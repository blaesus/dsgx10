#!/bin/bash
# The 8 GiB guard beside passes.sh on rank 2 (it shares the box with the engines' ring jobs): every 15 s, while a
# passes.sh runs, SIGTERM the reference's python if its RSS passes CAP_KIB or the box's MemAvailable falls under
# MIN_AVAIL_KIB. passes.sh then retries the pass after 120 s from its last finished layer (up to 8 tries). A
# systemd user scope with MemoryMax would cap it harder, but rank 2's user manager does not linger, so a scope
# dies with the last login session. Run detached beside passes.sh, from the ref directory:
#
#   setsid nohup ./guard.sh >> ~/altar1-ref/logs/guard.log 2>&1 < /dev/null &
CAP_KIB=${CAP_KIB:-8388608}
MIN_AVAIL_KIB=${MIN_AVAIL_KIB:-8388608}
echo "$(date -Is) guard: rss cap ${CAP_KIB} KiB, MemAvailable floor ${MIN_AVAIL_KIB} KiB"
sleep 5
while pgrep -f 'passes\.sh' > /dev/null; do
  avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  for pid in $(pgrep -f 'python -I glmref\.py'); do
    rss=$(awk '/^VmRSS:/ {print $2}' /proc/$pid/status 2>/dev/null)
    [ -z "$rss" ] && continue
    if [ "$rss" -gt "$CAP_KIB" ] || [ "$avail" -lt "$MIN_AVAIL_KIB" ]; then
      echo "$(date -Is) guard: kill $pid (rss ${rss} KiB, MemAvailable ${avail} KiB)"
      kill -TERM "$pid"
    fi
  done
  sleep 15
done
echo "$(date -Is) guard: no passes.sh, exiting"
