#!/bin/bash
# Lending the ring (docs/engine.md, "Borrowing the ring"), end to end against the mock engine: a test
# instance of service.sh (DSGB10_MOCK=1) with its own home, port and ring lock, and jobs through
# ring.sh on that lock. Nothing here touches the service's run directory, the ranks or the real ring
# lock: ssh and pkill are stand-ins that only log, and the test checks that none was called. Never run
# run4.sh from here: its stray reap kills every dsgb10_rank on the boxes.
#   engine/server/tests/yield_test.sh        # YT_DIR (a short path: Unix sockets), YT_PORT (18897)
set -u
D=$(cd "$(dirname "$0")/.." && pwd)
SVC=$D/service.sh RING=$D/ring.sh
FE=${DSGB10_FRONT_BIN:-$D/../../frontend/target/release/dsfront}
T=${YT_DIR:-${TMPDIR:-/tmp}/dsgb10-yt.$$}
PORT=${YT_PORT:-18897}
IDLE=4 HOLD=10 GRACE=2  # the lending idle time, dsfront's hold limit, a preempted job's grace: short
pass=0 fail=0
ok() {
  echo "  ok: $*"
  pass=$((pass + 1))
}
bad() {
  echo "  FAIL: $*"
  fail=$((fail + 1))
}
now() { date +%s.%N; }
since() { python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$1" "$2"; }
between() { python3 -c "import sys; x, lo, hi = map(float, sys.argv[1:]); sys.exit(not lo <= x <= hi)" "$1" "$2" "$3"; }
until_() {  # seconds command...: poll every 0.25 s until the command succeeds
  local end=$((SECONDS + $1))
  shift
  until "$@"; do
    ((SECONDS < end)) || return 1
    sleep 0.25
  done
}

[ -x "$FE" ] || {
  echo "no $FE (cargo build --release in frontend/)"
  exit 2
}
ss -ltn | grep -q "127.0.0.1:$PORT " && {
  echo "port $PORT is in use"
  exit 2
}
rm -rf "$T" && mkdir -p "$T/shim" && chmod 700 "$T"
for c in ssh pkill; do  # nothing here may reach a rank: these only log
  printf '#!/bin/sh\necho "%s $*" >> "%s/shim.calls"\nexit 1\n' "$c" "$T" > "$T/shim/$c" && chmod +x "$T/shim/$c"
done
export PATH=$T/shim:$PATH
[ "$(type -P ssh)" = "$T/shim/ssh" ] && [ "$(type -P pkill)" = "$T/shim/pkill" ] || {
  echo "the stand-ins for ssh and pkill are not first in PATH: stop"
  exit 2
}
head -c 24 /dev/urandom | base64 > "$T/key" && chmod 600 "$T/key"
printf 'Authorization: Bearer %s\n' "$(cat "$T/key")" > "$T/hdr" && chmod 600 "$T/hdr"
export DSGB10_MOCK=1 DSGB10_HOME=$T/svc RING_LOCK=$T/ring.lock DSGB10_ADDR=127.0.0.1 DSGB10_PORT=$PORT
export DSGB10_KEY_FILE=$T/key DSGB10_FRONT_BIN=$FE DSGB10_YIELD_IDLE_S=$IDLE DSGB10_HOLD_S=$HOLD
export DSGB10_ENGINE_ARGS="--delay-ms 5" START_WAIT=60 LOCK_WAIT=600 OWNER=yield_test PREEMPT_GRACE=$GRACE
LOG=$T/svc/logs/service.log WANTS=$T/ring.wants

# the service, before: its supervisor and restart count, to compare at the end (its state may change
# meanwhile: it lends the ring to other jobs)
PROD_RUN=$HOME/dsgb10/service/run
prod() { echo "supervisor $(cat "$PROD_RUN/supervisor.pid" 2> /dev/null), $(cat "$PROD_RUN/restarts" 2> /dev/null)"; }
PROD0=$(prod)

req() {  # name max_tokens [content]: a completion in the background; $T/<name>.{t0,t1,code,json,hdr}
  local body
  body=$(python3 -c "import json, sys; print(json.dumps({'model': 'm', 'messages': [{'role': 'user', 'content': sys.argv[2]}], 'max_tokens': int(sys.argv[1])}))" "$2" "${3:-Say hello.}")
  (
    now > "$T/$1.t0"
    curl -s -m 300 -D "$T/$1.hdr" -o "$T/$1.json" -w '%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H @"$T/hdr" -H 'Content-Type: application/json' -d "$body" > "$T/$1.code"
    now > "$T/$1.t1"
  ) &
}
sreq() {  # name max_tokens: a streamed completion in the background (SREQ_MAX: the client's limit, s); $T/<name>.{t0,t1,w,sse}
  local body
  body=$(python3 -c "import json, sys; print(json.dumps({'model': 'm', 'messages': [{'role': 'user', 'content': 'Say hello.'}], 'max_tokens': int(sys.argv[1]), 'stream': True}))" "$2")
  (
    now > "$T/$1.t0"
    curl -sN -m "${SREQ_MAX:-300}" -o "$T/$1.sse" -w '%{http_code} %{time_starttransfer}' "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H @"$T/hdr" -H 'Content-Type: application/json' -d "$body" > "$T/$1.w"
    now > "$T/$1.t1"
  ) &
}
wcode() { cut -d' ' -f1 "$T/$1.w" 2> /dev/null; }
ttfb() { cut -d' ' -f2 "$T/$1.w" 2> /dev/null; }
waiting() { "$FE" admin --socket "$T/svc/run/front.admin" status 2> /dev/null | sed -n 's/.* waiting \([0-9]*\) .*/\1/p'; }
code() { cat "$T/$1.code" 2> /dev/null; }
done_() { [ -s "$T/$1.t1" ]; }
tokens() { python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(d['usage']['completion_tokens'], d['choices'][0]['finish_reason'])" "$T/$1.json" 2> /dev/null; }
job() {  # name seconds: a ring job (ring.sh, the test lock) that records when it got the ring and ended
  "$RING" bash -c "date +%s.%N > $T/$1.start; sleep $2; date +%s.%N > $T/$1.end" > "$T/$1.log" 2>&1 &
  echo $! > "$T/$1.pid"
}
tree_job() {  # name: a long job of three processes, one of which ignores SIGTERM; records its process group
  "$RING" bash -c "ps -o pgid= -p \$\$ | tr -d ' ' > $T/$1.pg; date +%s.%N > $T/$1.start
    sleep 300 & bash -c 'trap \"\" TERM; sleep 300' & wait; date +%s.%N > $T/$1.end" > "$T/$1.log" 2>&1 &
  echo $! > "$T/$1.pid"
}
rc_of() { wait "$(cat "$T/$1.pid")"; }  # a job's exit status, once it ends
started() { [ -s "$T/$1.start" ]; }
ended() { [ -s "$T/$1.end" ]; }
state() { cut -d' ' -f1 "$T/svc/run/state" 2> /dev/null; }
serving() { [ "$(state)" = serving ] && [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ]; }
starts() { grep -c "engine started" "$LOG"; }
held() { ls "$WANTS" 2> /dev/null | grep -q '\.held$'; }  # a driver's held want is queued
hcode() { curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health"; }
hbody() { curl -s -m 2 "http://127.0.0.1:$PORT/health" | tr -d '\n'; }
answers() { [ "$(hcode)" != 000 ]; }  # dsfront listens
says_lent() { [ "$(hcode)" = 503 ] && hbody | grep -q 'the ring is lent'; }
unheld() { ! held; }

echo "1. the test service starts (mock engine; home $T/svc, port $PORT, lock $RING_LOCK)"
"$SVC" start > "$T/start.log" 2>&1 && ok "serving: $(grep -m1 '^service:' "$T/start.log")" || bad "start: $(tail -n 3 "$T/start.log")"
req r1 16
until_ 30 done_ r1
[ "$(code r1)" = 200 ] && ok "a completion: 200 ($(tokens r1))" || bad "a completion: $(code r1)"

echo "2. no lending while requests come; lent $IDLE s after the last one"
job j1 3
t_end=$((SECONDS + 3 * IDLE))
i=0
while ((SECONDS < t_end)); do
  req "a$i" 24
  until_ 30 done_ "a$i"
  [ "$(code "a$i")" = 200 ] || bad "a request while the job waits: $(code "a$i")"
  i=$((i + 1))
  sleep 0.7
done
last=$(cat "$T/a$((i - 1)).t1")
started j1 && bad "the job got the ring while requests came" || ok "$i requests over $((3 * IDLE)) s: the job still waits"
grep -q "waiting for the ring: .*ring.sh bash" "$LOG" && ok "the supervisor saw the job's want" || bad "no want in the log"
until_ 30 started j1 && t=$(since "$last" "$(cat "$T/j1.start")") &&
  between "$t" "$IDLE" $((IDLE + 8)) && ok "lent: the job started ${t} s after the last request" || bad "the job started ${t:-never} s after the last request"
grep -q "lending the ring" "$LOG" && [ "$(state)" = lent ] && ok "state lent; $(grep -c 'rank 0 exited' "$LOG") engine stop" || bad "state $(state)"
c=$(curl -s -m 2 -w ' %{http_code}' "http://127.0.0.1:$PORT/health" | tr -d '\n')
[ "${c##* }" = 503 ] && ok "/health while lent: ${c}" || bad "/health while lent: $c"
until_ 30 serving && ok "back after the job: serving; $(starts) engine starts" || bad "not serving after the job: $(state)"
grep -q "the ring is back after .* (the job is done)" "$LOG" && ok "$(grep -o 'the ring is back after .*' "$LOG" | tail -n 1)" || bad "no retake in the log"

echo "3. a request in flight is never cut"
req long 2400 "MOCK_LONG 3000"
sleep 1
job j2 2
until_ 60 done_ long
[ "$(code long)" = 200 ] && [ "$(tokens long)" = "2400 length" ] && ok "the long request: 200, $(tokens long), $(since "$(cat "$T/long.t0")" "$(cat "$T/long.t1")") s" ||
  bad "the long request: $(code long) $(tokens long)"
until_ 30 started j2 && t=$(since "$(cat "$T/long.t1")" "$(cat "$T/j2.start")") && between "$t" "$IDLE" $((IDLE + 8)) &&
  ok "the job started ${t} s after it ended" || bad "the job started ${t:-never} s after the long request ended"
until_ 30 ended j2 && until_ 30 serving || bad "not serving after j2"

echo "4. a request that comes during a lend preempts the job; the job queued behind it waits"
s0=$(starts)
tree_job j3
until_ 30 started j3 || bad "j3 did not start"
job j4 1
until_ 10 test -e "$WANTS/$(cat "$T/j4.pid")" && ok "j4 waits behind j3" || bad "j4 registered no want"
req held 16
sreq sheld 16
until_ 60 done_ held
t=$(since "$(cat "$T/held.t0")" "$(cat "$T/held.t1")")
[ "$(code held)" = 200 ] && between "$t" 0 $((GRACE + 10)) && ok "the held request: 200 after $t s (the grace: $GRACE s)" ||
  bad "the held request: $(code held) after $t s: $(head -c 300 "$T/held.json")"
until_ 60 done_ sheld
[ "$(wcode sheld)" = 200 ] && between "$(ttfb sheld)" 0 1.5 && grep -q '^:.*waiting for the engine (the ring is lent' "$T/sheld.sse" &&
  grep -q '"content":"Mock' "$T/sheld.sse" && tail -n 2 "$T/sheld.sse" | grep -q '^data: \[DONE\]' &&
  ok "a held stream: 200 and its first bytes after $(ttfb sheld) s, $(grep -c '^:' "$T/sheld.sse") comments while it waited, then its answer" ||
  bad "a held stream: $(cat "$T/sheld.w"): $(head -c 300 "$T/sheld.sse")"
rc_of j3
rc=$?
[ "$rc" = 76 ] && ! ended j3 && ok "j3 was preempted: exit 76, $(grep -m1 -o 'preempted: [^(]*' "$T/j3.log")" || bad "j3: exit $rc, $(tail -n 2 "$T/j3.log")"
pg=$(cat "$T/j3.pg" 2> /dev/null)
[ -n "$pg" ] && [ -z "$(pgrep -g "$pg")" ] && ok "nothing of j3's process group is left (one of its processes ignored SIGTERM)" ||
  bad "j3's process group $pg: $(pgrep -g "${pg:-0}" | tr '\n' ' ')"
grep -q "requests wait ([0-9]*): the job is preempted" "$LOG" && ok "the supervisor reclaimed the ring for it" || bad "no reclaim in the log"
started j4 && t=$(since "$(cat "$T/held.t1")" "$(cat "$T/j4.start")") && python3 -c "import sys; sys.exit(float(sys.argv[1]) <= 0)" "$t" &&
  bad "j4 started before the held request was answered" || ok "j4 did not go before the held request"
until_ 30 started j4 && t=$(since "$(cat "$T/held.t1")" "$(cat "$T/j4.start")") && between "$t" "$IDLE" $((IDLE + 8)) &&
  ok "j4 got the ring ${t} s after the held request ended" || bad "j4 started ${t:-never} s after the held request"
until_ 30 ended j4 && until_ 30 serving || bad "not serving after j4"
rc_of j4 && ok "j4 ran to its end: exit 0" || bad "j4: exit $?"
[ $(($(starts) - s0)) = 2 ] && ok "two engine starts: after j3, after j4" || bad "$(($(starts) - s0)) engine starts"

echo "5. a job with PREEMPT=0 runs to its end; a request held past the limit ($HOLD s) gets 503 with Retry-After, a stream an error event"
PREEMPT=0 job j5 $((HOLD + 8))
until_ 30 started j5 || bad "j5 did not start"
req late 16
sreq slate 16
SREQ_MAX=2 sreq sgone 16
sleep 1
w3=$(waiting)
until_ 10 done_ sgone
sleep 0.5
w2=$(waiting)
[ "$w3" = 3 ] && [ "$w2" = 2 ] && ok "a stream whose client left while it waited is counted out (waiting $w3, then $w2)" ||
  bad "waiting $w3, then $w2, after a waiting stream's client left"
until_ 30 done_ late
t=$(since "$(cat "$T/late.t0")" "$(cat "$T/late.t1")")
ra=$(grep -i '^retry-after:' "$T/late.hdr" | tr -d '\r' | cut -d' ' -f2)
[ "$(code late)" = 503 ] && [ -n "$ra" ] && between "$t" "$HOLD" $((HOLD + 2)) && ok "503 after $t s, Retry-After: $ra: $(head -c 160 "$T/late.json")" ||
  bad "late: $(code late) after $t s, Retry-After '$ra'"
until_ 30 done_ slate
t=$(since "$(cat "$T/slate.t0")" "$(cat "$T/slate.t1")")
[ "$(wcode slate)" = 200 ] && between "$(ttfb slate)" 0 1.5 && [ "$(grep -c '^:' "$T/slate.sse")" -ge 2 ] && grep -q '"code":503' "$T/slate.sse" &&
  tail -n 2 "$T/slate.sse" | grep -q '^data: \[DONE\]' && between "$t" "$HOLD" $((HOLD + 2)) &&
  ok "a stream held past the limit: 200 at once, $(grep -c '^:' "$T/slate.sse") comments, an error event after $t s: $(grep -o '"message":"[^"]*' "$T/slate.sse" | head -c 100)" ||
  bad "a stream held past the limit: $(cat "$T/slate.w") after $t s: $(head -c 400 "$T/slate.sse")"
until_ 40 ended j5 && rc_of j5 && ok "j5 ran to its end while the request waited: exit 0" || bad "j5: exit $?, $(tail -n 2 "$T/j5.log")"
until_ 30 serving && req after 16 && until_ 30 done_ after && [ "$(code after)" = 200 ] && ok "after the job: 200" || bad "after j5: $(code after)"

echo "6. jobs in a row, while no request comes, follow one another without the engine in between"
until_ 30 serving
s0=$(starts)
job j6 3
job j7 2
until_ 40 ended j6 && until_ 40 ended j7 || bad "j6, j7 did not both run"
a=j6 b=j7
python3 -c "import sys; sys.exit(float(sys.argv[1]) > float(sys.argv[2]))" "$(cat "$T/j6.start")" "$(cat "$T/j7.start")" || a=j7 b=j6
t=$(since "$(cat "$T/$a.end")" "$(cat "$T/$b.start")")
between "$t" 0 3 && ok "$b started $t s after $a ended" || bad "$b started $t s after $a ended"
until_ 30 serving
[ $(($(starts) - s0)) = 1 ] && ok "one engine start, after both" || bad "$(($(starts) - s0)) engine starts"

echo "7. a stale want (its job killed) is dropped, and nothing is lent"
l0=$(grep -c "lending the ring" "$LOG")
req s1 16
until_ 30 done_ s1
mkdir -p "$WANTS"
bash -c "exec 7> $WANTS/stale.new; flock 7; echo stale >&7; mv $WANTS/stale.new $WANTS/999999; exec sleep 300" &
sp=$!
until_ 5 test -e "$WANTS/999999"
kill -9 $sp
wait $sp 2> /dev/null
until_ 10 test ! -e "$WANTS/999999" && ok "the stale want is gone" || bad "the stale want is still there"
sleep $((IDLE + 3))
[ "$(state)" = serving ] && [ "$(grep -c "lending the ring" "$LOG")" = "$l0" ] && ok "still serving; nothing lent" || bad "state $(state)"

echo "8. stop while lent: the job (PREEMPT=0) keeps the ring; a waiting request is answered at once"
PREEMPT=0 job j8 12
until_ 30 started j8 || bad "j8 did not start"
req stopped 16
sreq sstopped 16
sleep 2
t0=$(now)
"$SVC" stop > "$T/stop.log" 2>&1
t=$(since "$t0" "$(now)")
until_ 10 done_ stopped
[ "$(code stopped)" = 503 ] && ok "the waiting request: 503 at the stop ($(head -c 120 "$T/stopped.json"))" || bad "the waiting request: $(code stopped)"
until_ 10 done_ sstopped
grep -q '"message":"the server is shutting down"' "$T/sstopped.sse" && tail -n 2 "$T/sstopped.sse" | grep -q '^data: \[DONE\]' &&
  ok "the waiting stream: its error event at the stop" || bad "the waiting stream at the stop: $(cat "$T/sstopped.w"): $(head -c 300 "$T/sstopped.sse")"
between "$t" 0 8 && ok "stop took $t s" || bad "stop took $t s"
kill -0 "$(cat "$T/j8.pid")" 2> /dev/null && ! ended j8 && ok "the job runs on" || bad "the job was cut"
flock -n "$RING_LOCK" true && bad "the ring lock is free while the job runs" || ok "the job holds the ring lock"
until_ 20 ended j8 && ok "the job ended by itself" || bad "the job did not end"
"$SVC" status > /dev/null 2>&1
[ $? = 3 ] && ok "the test service is down" || bad "status after the stop: $?"
[ "$(stat -c %a "$T/svc/run/front.table" 2> /dev/null)" = 600 ] && ok "dsfront saved its session table, mode 600: $(grep 'session table' "$T/svc/logs/front.log" | tail -n 1)" ||
  bad "no session table saved: $(grep 'session table' "$T/svc/logs/front.log" | tail -n 1)"

echo "9. a ring job inside a ring job runs at once: the lock comes with it"
"$SVC" start > "$T/start2.log" 2>&1 && until_ 30 serving || bad "the test service did not start again: $(tail -n 2 "$T/start2.log")"
"$RING" bash -c "date +%s.%N > $T/n.start; $RING bash -c 'date +%s.%N > $T/n.inner'; echo \$? > $T/n.rc" > "$T/n.log" 2>&1 &
echo $! > "$T/n.pid"
until_ 30 test -s "$T/n.rc" && [ "$(cat "$T/n.rc")" = 0 ] && t=$(since "$(cat "$T/n.start")" "$(cat "$T/n.inner")") && between "$t" 0 2 &&
  ok "the inner ring.sh ran ${t} s after the outer one got the ring" || bad "nested: $(cat "$T/n.rc" 2> /dev/null), $(tail -n 3 "$T/n.log")"
[ "$(grep -c 'the ring is busy' "$T/n.log")" = 1 ] && ok "only the outer one waited for the ring" || bad "nested waits: $(grep -c 'the ring is busy' "$T/n.log")"
rc_of n && until_ 30 serving && ok "back to serving" || bad "after the nested job: $(state)"
grep -q "session table: [1-9][0-9]* sessions from" "$T/svc/logs/front.log" && [ ! -e "$T/svc/run/front.table" ] &&
  ok "the restarted dsfront took its table back, and the file is gone: $(grep 'session table' "$T/svc/logs/front.log" | tail -n 1)" ||
  bad "the table at the start: $(grep 'session table' "$T/svc/logs/front.log" | tail -n 1); $(ls "$T/svc/run" | tr '\n' ' ')"

echo "10. a driver's held want keeps the ring lent between its parts; a job that waits in the gap goes first"
until_ 30 serving
s0=$(starts)
b0=$(grep -c "the ring is back" "$LOG")
# the driver: its want held; part d1 (3 s), 5 s later part d2 (2 s); then its want dropped
bash -c ". '$D/ringlock.sh'; ring_want_hold 'yield_test driver'
  '$RING' bash -c 'date +%s.%N > $T/d1.start; sleep 3; date +%s.%N > $T/d1.end'; sleep 5
  '$RING' bash -c 'date +%s.%N > $T/d2.start; sleep 2; date +%s.%N > $T/d2.end'; ring_want_drop" > "$T/drv.log" 2>&1 &
drv=$!
until_ 30 started d1 || bad "the driver's first part did not start"
held=$(ls "$WANTS" 2> /dev/null | grep '\.held$')
[ -n "$held" ] && ok "the driver's want is held: $(head -n 1 "$WANTS/$held")" || bad "no held want: $(ls "$WANTS" 2> /dev/null | tr '\n' ' ')"
job jg 1
until_ 10 test -e "$WANTS/$(cat "$T/jg.pid")" || bad "jg registered no want"
until_ 30 started jg && t=$(since "$(cat "$T/d1.end")" "$(cat "$T/jg.start")") && between "$t" 0 3 &&
  ok "jg, which waited behind d1, got the ring ${t} s after d1 ended (in the driver's gap)" || bad "jg started ${t:-never} s after d1 ended"
until_ 30 started d2 && t=$(since "$(cat "$T/jg.end")" "$(cat "$T/d2.start")") && between "$t" 0 6 &&
  ok "d2 got the ring ${t} s after jg ended" || bad "d2 started ${t:-never} s after jg ended"
wait $drv
until_ 10 unheld && ok "the driver's want is gone" || bad "the held want is still there"
until_ 30 serving || bad "not serving after the driver"
b1=$(grep -c "the ring is back" "$LOG")
[ $(($(starts) - s0)) = 1 ] && [ $((b1 - b0)) = 1 ] && ok "the ring came back once, after the driver's last part; one engine start" ||
  bad "the ring came back $((b1 - b0)) times; $(($(starts) - s0)) engine starts"
# a driver killed outright: its want goes within a few seconds
bash -c ". '$D/ringlock.sh'; ring_want_hold 'yield_test killed driver'; exec sleep 300" > /dev/null 2>&1 &
kp=$!
until_ 5 held || bad "the second driver held no want"
kill -9 $kp
wait $kp 2> /dev/null
until_ 6 unheld && ok "a driver killed with SIGKILL: its held want went within 6 s" ||
  bad "the killed driver's want is still there"

echo "11. forget: refused while the service runs; stop --forget stops it, then removes dsfront's saved table"
printf 'x\n' > "$T/svc/run/front.table" && chmod 600 "$T/svc/run/front.table"
"$SVC" forget > "$T/forget1.log" 2>&1
rc=$?
[ "$rc" = 1 ] && [ -f "$T/svc/run/front.table" ] && ok "forget while it runs: refused ($(head -n 1 "$T/forget1.log"))" ||
  bad "forget while it runs: rc $rc, $(head -n 2 "$T/forget1.log" | tr '\n' '|')"
"$SVC" stop --forget > "$T/forget2.log" 2>&1
rc=$?
"$SVC" status > /dev/null 2>&1
st=$?
[ "$rc" = 0 ] && [ "$st" = 3 ] && [ ! -e "$T/svc/run/front.table" ] && ok "stop --forget: $(tail -n 1 "$T/forget2.log")" ||
  bad "stop --forget: rc $rc, status $st, $(ls "$T/svc/run" | tr '\n' ' '), $(tail -n 2 "$T/forget2.log" | tr '\n' '|')"

echo "12. jobs take the ring in the order they asked for it, even when a younger one finds the lock free"
bash -c "exec 9>> '$RING_LOCK'; flock 9; exec sleep 30" &  # the lock, held without a want
hp=$!
until_ 5 bash -c "! flock -n '$RING_LOCK' true" || bad "the holder did not take the lock"
job q1 1
until_ 10 test -e "$WANTS/$(cat "$T/q1.pid")" || bad "q1 registered no want"
sleep 0.3
job q2 1
until_ 10 test -e "$WANTS/$(cat "$T/q2.pid")" || bad "q2 registered no want"
kill $hp
wait $hp 2> /dev/null
job q3 1  # the lock is free and q1, q2 are between polls: q3 must still wait for them
until_ 30 ended q1 && until_ 30 ended q2 && until_ 30 ended q3 || bad "q1-q3 did not all run"
o=$(for q in q1 q2 q3; do echo "$(cat "$T/$q.start") $q"; done | sort -n | cut -d' ' -f2 | tr '\n' ' ')
[ "$o" = "q1 q2 q3 " ] && ok "in the order they asked: $o" || bad "the order: $o"
python3 -c "import sys; s = [float(open(sys.argv[1] + '/' + q + '.' + e).read()) for q in ('q1', 'q2', 'q3') for e in ('start', 'end')]; sys.exit(not (s[1] <= s[2] and s[3] <= s[4]))" "$T" &&
  ok "one after another" || bad "they overlapped"
grep -q "jobs wait for the ring before this one" "$T/q3.log" && ok "q3: $(grep -o 'jobs wait for the ring before this one' "$T/q3.log")" || bad "q3: $(head -n 1 "$T/q3.log")"

echo "13. a dead want, and a driver's held want, hold no job back"
mkdir -p "$WANTS"
bash -c "exec 7> '$WANTS/888888.new'; flock 7; printf 'dead\n1 888888\n' >&7; mv '$WANTS/888888.new' '$WANTS/888888'; exec sleep 300" &
sp=$!
until_ 5 test -e "$WANTS/888888"
kill -9 $sp
wait $sp 2> /dev/null
bash -c "exec 7> '$WANTS/777777.held.new'; flock 7; printf 'held\n1 777777\n' >&7; mv '$WANTS/777777.held.new' '$WANTS/777777.held'; exec sleep 300" &
hp=$!
until_ 5 test -e "$WANTS/777777.held"
until_ 10 flock -n "$RING_LOCK" true || bad "the lock is still held after q3"  # q3's ring.sh lets go a moment after its end
t0=$(now)
job q4 1
until_ 10 started q4 && t=$(since "$t0" "$(cat "$T/q4.start")") && between "$t" 0 1.5 &&
  ok "q4 got the ring at once (${t} s), past a dead want and a held one, both older" || bad "q4 started ${t:-never} s after it asked"
kill $hp
wait $hp 2> /dev/null
rm -f "$WANTS/777777.held" "$WANTS/888888"
until_ 30 ended q4 || bad "q4 did not end"

echo "14. a start while a job holds the ring, and no request comes: dsfront answers at once; the engine starts after the job"
s0=$(starts)
job js1 8
until_ 10 started js1 || bad "js1 did not start"
t0=$(now)
START_WAIT=0 "$SVC" start > "$T/start3.log" 2>&1
until_ 10 answers && t=$(since "$t0" "$(now)") && ok "dsfront answers $t s after the start: /health $(hcode), $(hbody)" ||
  bad "dsfront did not answer: $(tail -n 3 "$LOG" | tr '\n' '|')"
until_ 10 test "$(state)" = waiting && ok "state waiting: $(cut -d' ' -f3- "$T/svc/run/state" | cut -c1-120)" || bad "state $(state)"
until_ 10 says_lent && ok "/health while it waits: 503, $(hbody)" || bad "/health while it waits: $(hcode), $(hbody)"
grep -q ' service: ' "$T/ring.owner" && bad "ring.owner says service while js1 has the ring" || ok "ring.owner is still js1's"
until_ 30 ended js1 && rc_of js1 && ok "js1 ran to its end: exit 0" || bad "js1: exit $?, $(tail -n 2 "$T/js1.log")"
until_ 30 serving && t=$(since "$(cat "$T/js1.end")" "$(now)") && ok "serving $t s after js1 ended" || bad "not serving after js1: $(state)"
grep -q "the ring is ours after .* of waiting at the start (the job is done)" "$LOG" && [ $(($(starts) - s0)) = 1 ] &&
  ok "$(grep -o 'the ring is ours after .*' "$LOG" | tail -n 1); one engine start" ||
  bad "$(($(starts) - s0)) engine starts; $(grep -o 'the ring is .*' "$LOG" | tail -n 1)"
"$SVC" stop > /dev/null 2>&1

echo "15. a start while a job holds the ring, and requests come: they wait, the job is preempted, and they are answered"
s0=$(starts)
r0=$(grep -c "requests wait ([0-9]*): the job is preempted" "$LOG")
tree_job js2
until_ 10 started js2 || bad "js2 did not start"
START_WAIT=0 "$SVC" start > "$T/start4.log" 2>&1
until_ 10 answers || bad "dsfront did not answer"
until_ 10 says_lent && [ "$(state)" = waiting ] && ok "waiting; /health 503, $(hbody)" || bad "state $(state), /health $(hcode), $(hbody)"
req hs 16
sreq shs 16
until_ 60 done_ hs
t=$(since "$(cat "$T/hs.t0")" "$(cat "$T/hs.t1")")
[ "$(code hs)" = 200 ] && between "$t" 0 $((GRACE + 10)) && ok "the request: 200 after $t s (the grace: $GRACE s)" ||
  bad "the request: $(code hs) after $t s: $(head -c 300 "$T/hs.json")"
until_ 60 done_ shs
[ "$(wcode shs)" = 200 ] && between "$(ttfb shs)" 0 1.5 && grep -q '^:.*waiting for the engine (the ring is lent' "$T/shs.sse" &&
  grep -q '"content":"Mock' "$T/shs.sse" && tail -n 2 "$T/shs.sse" | grep -q '^data: \[DONE\]' &&
  ok "the stream: 200 and its first bytes after $(ttfb shs) s, $(grep -c '^:' "$T/shs.sse") comments while it waited, then its answer" ||
  bad "the stream: $(cat "$T/shs.w"): $(head -c 300 "$T/shs.sse")"
rc_of js2
rc=$?
[ "$rc" = 76 ] && ! ended js2 && ok "js2 was preempted: exit 76" || bad "js2: exit $rc, $(tail -n 2 "$T/js2.log")"
[ $(($(grep -c "requests wait ([0-9]*): the job is preempted" "$LOG") - r0)) = 1 ] &&
  grep -q "the ring is ours after .* of waiting at the start (the job is done; requests wait)" "$LOG" &&
  ok "$(grep -o 'the ring is ours after .*' "$LOG" | tail -n 1)" || bad "no reclaim at the start in the log"
until_ 30 serving && [ $(($(starts) - s0)) = 1 ] && ok "serving; one engine start" || bad "after js2: $(state), $(($(starts) - s0)) engine starts"
"$SVC" stop > /dev/null 2>&1

echo "16. a stop while the start waits for the ring: dsfront goes, the job keeps the ring"
s0=$(starts)
job js3 10
until_ 10 started js3 || bad "js3 did not start"
START_WAIT=0 "$SVC" start > "$T/start5.log" 2>&1
until_ 10 says_lent && [ "$(state)" = waiting ] || bad "not waiting: $(state), /health $(hcode)"
t0=$(now)
"$SVC" stop > "$T/stop5.log" 2>&1
rc=$?
t=$(since "$t0" "$(now)")
[ "$rc" = 0 ] && [ "$(state)" = stopped ] && ! answers && between "$t" 0 8 && ok "stopped in $t s; dsfront is gone" ||
  bad "the stop: rc $rc, $t s, state $(state), /health $(hcode)"
! ended js3 && ! flock -n "$RING_LOCK" true && ok "js3 runs on with the ring" || bad "js3 lost the ring"
until_ 20 ended js3 && rc_of js3 && ok "js3 ran to its end: exit 0" || bad "js3: exit $?"
[ "$(starts)" = "$s0" ] && ok "no engine start" || bad "$(($(starts) - s0)) engine starts"

echo "17. a start behind two jobs in turn: status names the one that holds the ring now"
s0=$(starts)
job js4 6
until_ 10 started js4 || bad "js4 did not start"
START_WAIT=0 "$SVC" start > "$T/start6.log" 2>&1
until_ 10 says_lent && [ "$(state)" = waiting ] || bad "not waiting: $(state), /health $(hcode)"
job js5 4
until_ 10 test -e "$WANTS/$(cat "$T/js5.pid")" || bad "js5 is not queued"
st1() { "$SVC" status 2> /dev/null | head -n 1; }
st1 | grep -q "held by: .* pid $(cat "$T/js4.pid") " && ok "while js4 has the ring: $(st1 | cut -c1-110)" || bad "status: $(st1)"
until_ 20 started js5 || bad "js5 did not start"
st1 | grep -q "held by: .* pid $(cat "$T/js5.pid") " && ok "then js5: the first line names js5's pid ($(cat "$T/js5.pid"))" ||
  bad "status after js4: $(st1); ring.owner: $(cut -c1-100 "$T/ring.owner")"
until_ 30 serving && [ $(($(starts) - s0)) = 1 ] && ok "serving after js5; one engine start" || bad "after js5: $(state), $(($(starts) - s0)) starts"
"$SVC" stop > /dev/null 2>&1

echo "18. the service was not touched, nor any rank"
[ -s "$T/shim.calls" ] && bad "ssh or pkill was called: $(head -n 3 "$T/shim.calls" | tr '\n' '|')" || ok "no ssh, no pkill"
[ "$(prod)" = "$PROD0" ] && ok "its supervisor and restart count are as before: $PROD0 (state: $(cut -d' ' -f1 "$PROD_RUN/state"))" ||
  bad "before: $PROD0 | after: $(prod)"
ls "$T/svc/run" | tr '\n' ' ' | sed 's/^/    test run dir: /'
echo
grep -c "" "$LOG" | sed 's/^/    service.log lines: /'
for j in "$T"/j*.pid; do kill "$(cat "$j")" 2> /dev/null; done
"$SVC" stop > /dev/null 2>&1
[ "${YT_KEEP:-0}" = 1 ] || rm -rf "$T"
echo "$pass passed, $fail failed"
[ "$fail" = 0 ]
