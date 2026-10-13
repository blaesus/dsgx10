# The fleet file for the scripts (fleet/homelab.conf says what each key means), read as the engine reads
# it (engine/comm/fleet.cpp): KEY=value lines and # comments; an unknown key, a key twice or any other
# line is an error. Sourced by ringlock.sh (so by run4.sh, ring.sh and service.sh) and engine/comm/run4.sh.
# FLEET=FILE in the environment names another fleet file; the default is this project's own.
FLEET=${FLEET:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/fleet/homelab.conf}

# fleet_load FILE: every key as FLEET_<KEY> (FLEET_GID_INDEX, FLEET_RANK1_SSH, ...), and FLEET_WORKERS:
# ranks 1-3's SSH addresses. 1, saying why on stderr, if FILE is not a fleet file the scripts can use.
fleet_load() {
  local f=$1 line n=0 key val k v
  local skip='^[[:blank:]]*(#.*)?$' re='^([A-Z][A-Z0-9_]*)=([A-Za-z0-9._/~:,@+%{}-]*)([[:blank:]]+(#.*)?)?$'
  local known='^(GID_INDEX|NEXT_DEV[01]|PREV_DEV[01]|BIN_DIR|LOG_DIR|SERVICE_DIR|RING_LOCK|RANK[0-3]_(NAME|HOSTNAME|SSH|NEXT_IP[01]|PREV_IP[01]|WEIGHTS|HASHMAP|ENGRAM|ENGRAM_READ_BATCH|SPILL))$'
  local -A seen=()
  if [ ! -r "$f" ] || [ -d "$f" ]; then
    echo "fleet: cannot read the fleet file $f (FLEET=FILE)" >&2
    return 1
  fi
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    line=${line%$'\r'}
    [[ $line =~ $skip ]] && continue
    if ! [[ $line =~ $re ]]; then
      echo "fleet: $f:$n: not KEY=value (a value has no spaces or quotes): $line" >&2
      return 1
    fi
    key=${BASH_REMATCH[1]} val=${BASH_REMATCH[2]}
    if ! [[ $key =~ $known ]]; then
      echo "fleet: $f:$n: unknown key $key" >&2
      return 1
    fi
    if [ -n "${seen[$key]:-}" ]; then
      echo "fleet: $f:$n: $key again (line ${seen[$key]})" >&2
      return 1
    fi
    seen[$key]=$n
    printf -v "FLEET_$key" '%s' "$val"
  done < "$f"
  for k in BIN_DIR LOG_DIR SERVICE_DIR RING_LOCK RANK1_SSH RANK2_SSH RANK3_SSH; do
    v="FLEET_$k"
    if [ -z "${!v:-}" ]; then
      echo "fleet: $f: no $k" >&2
      return 1
    fi
  done
  FLEET_WORKERS=("$FLEET_RANK1_SSH" "$FLEET_RANK2_SSH" "$FLEET_RANK3_SSH")
}

# fleet_get KEY: a key's value as written ("" if the file has none), for keys made from a rank:
# fleet_get "RANK${r}_SPILL"
fleet_get() {
  local v="FLEET_$1"
  echo "${!v:-}"
}

# fleet_local PATH: a path value for this box (rank 0): a leading ~/ is $HOME. On another box a value
# goes into its command line as written, and that box's shell expands the ~/ (fleet_rel: for scp).
fleet_local() {
  case $1 in
    "~") echo "$HOME" ;;
    "~/"*) echo "$HOME/${1#\~/}" ;;
    *) echo "$1" ;;
  esac
}

# fleet_rel PATH: a path value for scp's remote side, where it is relative to the home directory
fleet_rel() {
  case $1 in
    "~") echo . ;;
    "~/"*) echo "${1#\~/}" ;;
    *) echo "$1" ;;
  esac
}

# fleet_rank VALUE R: the value with every {rank} as R, as the engine reads a path (its ~/ stays)
fleet_rank() {
  local p='{rank}'
  echo "${1//"$p"/$2}"
}

# fleet_model_files R: rank R's weight file, hash map and Engram shards, quoted for a command line
# (tools/evict_cache.py takes them, and expands their ~/ on the box that runs it)
fleet_model_files() {
  local e w h
  w=$(fleet_rank "$(fleet_get "RANK${1}_WEIGHTS")" "$1") h=$(fleet_rank "$(fleet_get "RANK${1}_HASHMAP")" "$1")
  e=$(fleet_rank "$(fleet_get "RANK${1}_ENGRAM")" "$1")
  [ -z "$w" ] || printf "'%s' " "$w"
  [ -z "$h" ] || printf "'%s' " "$h"
  [ -z "$e" ] || printf "'%s' '%s' " "$e/engram-l1-r${1}of4.bin" "$e/engram-l14-r${1}of4.bin"
}
