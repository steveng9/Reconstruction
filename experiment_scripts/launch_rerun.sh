#!/usr/bin/env bash
# launch_rerun.sh — start/stop/monitor the detached post-encoding-fix re-run.
#
# Workers are started with setsid + nohup, so they are reparented to init and
# survive logout, SSH drops, and terminal death. Each worker is independent:
# killing one (or having one killed by the OOM reaper) leaves the rest running
# and its in-flight job is returned to the queue by `reap`.
#
#   ./experiment_scripts/launch_rerun.sh start [AIM] [GEN] [ATK]
#   ./experiment_scripts/launch_rerun.sh status
#   ./experiment_scripts/launch_rerun.sh tail        # follow all worker logs
#   ./experiment_scripts/launch_rerun.sh stop        # graceful (SIGTERM)
#   ./experiment_scripts/launch_rerun.sh restart [AIM] [GEN] [ATK]
#
# Safe to run `start` repeatedly: workers already running are left alone and
# only the shortfall is launched.

set -uo pipefail

# Resolve the repository root (the directory containing paths.py) without
# hard-coding an absolute path, and default the data root beneath it.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || {
  echo "error: cannot resolve script directory" >&2; exit 1; }
while [ ! -f "$REPO_ROOT/paths.py" ] && [ "$REPO_ROOT" != "/" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -f "$REPO_ROOT/paths.py" ] || {
  echo "error: cannot locate repository root (no paths.py found above $0)" >&2; exit 1; }
: "${RECON_DATA_ROOT:=$REPO_ROOT/data}"


REPO="$REPO_ROOT"
SCRIPTS="$REPO/experiment_scripts"
LOGDIR="$REPO/outfiles/rerun"
PIDDIR="$LOGDIR/pids"
CONDA_ENV="recon_"
# Invoke the env's interpreter directly rather than via `conda run`: conda run
# forks a wrapper, so the pidfile would hold the wrapper's pid and a SIGTERM to
# it would never reach the worker (the worker would keep running after `stop`).
PYBIN="$(conda info --base)/envs/$CONDA_ENV/bin/python"

# Pin every numeric library to a single thread. sklearn/LightGBM/TabPFN inherit
# the OpenMP default of one thread per core, so on this 48-core box each job
# child opened ~50 threads; with ~26 concurrent children the load average hit
# 300+ and the box thrashed on context switches while doing less real work.
# Parallelism here comes from running many single-threaded workers, not from
# threads inside one worker — the two must not be multiplied together.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# The pool is split by job kind on purpose. Generation and attacks have wildly
# different runtimes (AIM at large epsilon runs for hours; a RandomForest scores
# in seconds), and a single undifferentiated pool lets every worker grab a slow
# generate job and starve thousands of cheap attacks behind it.
# Generation is split again, this time by generator. AIM's runtime is monotone
# in epsilon and dwarfs everything else (adult 10k: ~2 min at eps=0.1, ~3.5 h at
# eps=10; >6 h and timing out at eps>=300 even on the 1k datasets), so an
# uncapped pool hands every free worker an AIM job and starves the cheap work
# behind it. AIM_N is a ceiling, not a reservation: fast-lane workers borrow AIM
# jobs once no non-AIM generation is left, so nothing idles at the end of a run.
DEFAULT_AIM=4
DEFAULT_GEN=8
DEFAULT_ATK=6

mkdir -p "$LOGDIR" "$PIDDIR"

live_workers() {
  local n=0
  for f in "$PIDDIR"/*.pid; do
    [ -e "$f" ] || continue
    if kill -0 "$(cat "$f")" 2>/dev/null; then n=$((n+1)); else rm -f "$f"; fi
  done
  echo "$n"
}

launch_pool() {
  local prefix="$1" want="$2"; shift 2
  local i=0 started=0
  while [ "$started" -lt "$want" ] && [ "$i" -lt 64 ]; do
    i=$((i+1))
    local id; id=$(printf "%s%02d" "$prefix" "$i")
    if [ -f "$PIDDIR/$id.pid" ] && kill -0 "$(cat "$PIDDIR/$id.pid")" 2>/dev/null; then
      started=$((started+1)); continue
    fi
    # -u keeps output unbuffered so `tail` is live.
    setsid nohup "$PYBIN" -u "$SCRIPTS/queue_worker.py" --worker-id "$id" "$@" \
      >> "$LOGDIR/$id.log" 2>&1 &
    # setsid forks and exits, so $! is the pid of the short-lived setsid
    # wrapper, not the worker -- recording it leaves a dead pid in the
    # pidfile and `stop`/`live_workers` then miss the worker entirely.
    # Resolve the real pid by its unique --worker-id instead.
    sleep 1
    real_pid=$(pgrep -f "queue_worker.py --worker-id $id " | head -1)
    echo "${real_pid:-0}" > "$PIDDIR/$id.pid"
    echo "  started $id (pid ${real_pid:-?}) [$*] -> $LOGDIR/$id.log"
    started=$((started+1))
    sleep 2   # stagger: avoids a thundering herd on the SQLite claim lock
  done
}

start() {
  local aim="${1:-$DEFAULT_AIM}" gen="${2:-$DEFAULT_GEN}" atk="${3:-$DEFAULT_ATK}"
  echo "workers live: $(live_workers)  (target: $aim AIM + $gen generate + $atk attack)"

  # Reclaim anything orphaned by a previous crash before adding capacity.
  "$PYBIN" "$SCRIPTS/rerun_queue.py" reap --stale-minutes 180

  # --max-attempts 1 for AIM: its failure mode is a deterministic 6 h timeout,
  # not a flake, so retrying only burns another 6 h to learn the same thing.
  launch_pool m "$aim" --kinds generate --sdg-bases AIM --max-attempts 1
  launch_pool h "$gen" --kinds generate --exclude-sdg-bases AIM
  launch_pool a "$atk" --kinds attack
  echo "workers now live: $(live_workers)"
}

stop() {
  local n=0
  for f in "$PIDDIR"/*.pid; do
    [ -e "$f" ] || continue
    local pid; pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      # setsid made each worker its own process-group leader, so signal the
      # whole group: that reaches the job's child process too.
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      n=$((n+1))
    fi
    rm -f "$f"
  done
  echo "sent SIGTERM to $n worker(s); they exit after the current job"
  echo "run '$0 status' to watch them drain, then 'rerun_queue.py reap' if any job is stuck"
}

case "${1:-status}" in
  start)   start "${2:-$DEFAULT_AIM}" "${3:-$DEFAULT_GEN}" "${4:-$DEFAULT_ATK}" ;;
  stop)    stop ;;
  restart) stop; sleep 5; start "${2:-$DEFAULT_AIM}" "${3:-$DEFAULT_GEN}" "${4:-$DEFAULT_ATK}" ;;
  status)
    echo "workers live: $(live_workers)"
    "$PYBIN" "$SCRIPTS/rerun_queue.py" status
    ;;
  tail)    tail -n 20 -F "$LOGDIR"/*.log ;;
  *) echo "usage: $0 {start [GEN] [ATK]|stop|restart [GEN] [ATK]|status|tail}"; exit 1 ;;
esac
