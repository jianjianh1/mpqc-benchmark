#!/bin/bash
#
# Launch MPQC across nodes via mpirun + apptainer exec.
#
# Wraps the `sudo mpirun ... apptainer exec ... mpqc -i ...` template
# that produced traces/water_14_tns/. Defaults match the CloudLab
# multi-node setup; flags are provided to retarget to other clusters.
#
# Usage:
#   bin/run-mpqc-mpirun.sh SIF INPUT_JSON [OPTIONS] [-- EXTRA_MPIRUN_ARGS]
#
# Required:
#   SIF                  path to the MPQC SIF (built by build-mpqc-sif.sh)
#   INPUT_JSON           path to MPQC's input JSON
#
# Options:
#   -n, --np N           MPI ranks (default: 8)
#   -H, --hostfile FILE  mpirun hostfile (default: $PWD/hosts.txt)
#   -b, --backend NAME   MADNESS task backend hint for env (default: PaRSEC)
#   --bind PATH[:DST]    extra apptainer --bind (default: /proj if exists)
#   --tcp-include CIDR   OpenMPI btl/oob TCP include (default: 10.10.1.0/24)
#   --threads N          MAD_NUM_THREADS (default: 12)
#   --no-sudo            skip sudo wrapper (root or rootless setup)
#   --log FILE           tee mpirun's stdout/stderr to FILE
#   -h, --help           print this help and exit
#
# All flags after `--` are passed verbatim to mpirun.
#
# Examples:
#   sudo bin/run-mpqc-mpirun.sh \
#     /proj/.../apptainer/mpqc-fork-fixed-v16.sif \
#     /proj/.../water_14-post-mpi.json \
#     -n 16 -H /proj/.../hosts.txt
#
# See bin/REPRODUCING.md for the end-to-end recipe and prereqs.

set -euo pipefail

# Defaults
NP=8
HOSTFILE="$PWD/hosts.txt"
BACKEND="PaRSEC"
BIND_PATH=""
TCP_INCLUDE="10.10.1.0/24"
THREADS=12
USE_SUDO=true
LOG_FILE=""
EXTRA_MPIRUN_ARGS=()
SIF=""
INPUT=""

usage() { sed -n '3,/^set -e/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \?//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--np)         NP="$2"; shift 2 ;;
    -H|--hostfile)   HOSTFILE="$2"; shift 2 ;;
    -b|--backend)    BACKEND="$2"; shift 2 ;;
    --bind)          BIND_PATH="$2"; shift 2 ;;
    --tcp-include)   TCP_INCLUDE="$2"; shift 2 ;;
    --threads)       THREADS="$2"; shift 2 ;;
    --no-sudo)       USE_SUDO=false; shift ;;
    --log)           LOG_FILE="$2"; shift 2 ;;
    -h|--help)       usage 0 ;;
    --)              shift; EXTRA_MPIRUN_ARGS=("$@"); break ;;
    -*)              echo "unknown option: $1" >&2; usage 1 ;;
    *)
      if   [[ -z "$SIF"   ]]; then SIF="$1"
      elif [[ -z "$INPUT" ]]; then INPUT="$1"
      else echo "extra positional arg: $1" >&2; usage 1
      fi
      shift
      ;;
  esac
done

[[ -n "$SIF" && -n "$INPUT" ]] || { echo "ERROR: SIF and INPUT_JSON required" >&2; usage 1; }
[[ -f "$SIF"   ]] || { echo "ERROR: SIF not found: $SIF" >&2; exit 1; }
[[ -r "$INPUT" ]] || { echo "ERROR: INPUT JSON not readable: $INPUT" >&2; exit 1; }
# Hostfile only needed for multi-rank; mpirun -n 1 runs locally without one.
USE_HOSTFILE=$([[ "$NP" -gt 1 ]] && echo true || echo false)
if $USE_HOSTFILE; then
  [[ -r "$HOSTFILE" ]] || { echo "ERROR: hostfile not readable: $HOSTFILE" >&2; exit 1; }
fi

# Default --bind: include /proj if it exists (CloudLab) so the SIF can
# read shared inputs and write outputs.
if [[ -z "$BIND_PATH" && -d /proj ]]; then
  BIND_PATH="/proj"
fi

SUDO_PREFIX=()
if $USE_SUDO; then
  SUDO_PREFIX=( sudo )
fi

MPIRUN_FLAGS=( --allow-run-as-root )
if $USE_HOSTFILE; then
  MPIRUN_FLAGS+=( -hostfile "$HOSTFILE" --map-by node )
fi
MPIRUN_FLAGS+=(
  -n "$NP"
  --mca btl_tcp_if_include "$TCP_INCLUDE"
  --mca oob_tcp_if_include "$TCP_INCLUDE"
  -x "MAD_NUM_THREADS=$THREADS"
  -x "OMP_NUM_THREADS=1"
)
# Case-insensitive backend match: tolerate `PARSEC`, `parsec`, `Parsec`, etc.
# without silently skipping the PARSEC MCA env (which would hang the run).
case "${BACKEND,,}" in
  parsec)
    MPIRUN_FLAGS+=(
      -x "PARSEC_MCA_bind_threads=0"
      -x "PARSEC_MCA_runtime_num_cores=8"
    )
    ;;
  pthreads)
    : ;;  # no extra env
  *)
    echo "WARNING: unknown --backend '$BACKEND'; not setting any backend env" >&2
    ;;
esac

APPTAINER_FLAGS=( exec )
if [[ -n "$BIND_PATH" ]]; then
  APPTAINER_FLAGS+=( --bind "$BIND_PATH" )
fi

# Echo the full command line (including EXTRA_MPIRUN_ARGS) for logs / debugging.
# The echo MUST match the actual `run()` invocation below so users can copy-paste
# the echoed line to reproduce.
echo "+ ${SUDO_PREFIX[*]} mpirun ${MPIRUN_FLAGS[*]} ${EXTRA_MPIRUN_ARGS[*]} apptainer ${APPTAINER_FLAGS[*]} $SIF /home/mpqc/install/bin/mpqc -i $INPUT"

run() {
  "${SUDO_PREFIX[@]}" mpirun "${MPIRUN_FLAGS[@]}" "${EXTRA_MPIRUN_ARGS[@]}" \
    apptainer "${APPTAINER_FLAGS[@]}" "$SIF" \
    /home/mpqc/install/bin/mpqc -i "$INPUT"
}

# `run` is backgrounded (via `&`) rather than piped so $! is mpirun's own
# pid (or sudo's, wrapping it) instead of tee's. That lets cleanup() below
# actually reach mpirun: a caller wrapping this script in `timeout` only
# signals this wrapper process, and a plain `run 2>&1 | tee "$LOG_FILE"`
# pipeline leaves sudo/mpirun (and any remote ranks it spawned) running as
# an orphan when the wrapper is killed.
cleanup() {
  local sig="$1"
  local pid="${RUN_PID:-}"
  if [[ -z "$pid" ]]; then
    # A signal landing between `run &` and the `RUN_PID=$!` assignment below
    # would otherwise leave this trap with nothing to kill. bash populates
    # its job table synchronously when a job is backgrounded — before
    # control returns to the script — so `jobs -p` already has the PID even
    # in that narrow window, unlike the `RUN_PID` shell variable.
    pid="$(jobs -p | head -1)"
  fi
  echo "run-mpqc-mpirun.sh: received $sig, terminating mpirun (pid ${pid:-?})" >&2
  if [[ -n "$pid" ]]; then
    # `kill -TERM -"$pid"` (negative PID) signals the WHOLE process group,
    # not just $pid's immediate children — `pkill -P` only reaches one
    # generation down, so a deeper descendant (e.g. the actual mpqc binary
    # under `apptainer exec`) would survive as an orphan. `run &` backgrounds
    # its own process group (bash job control assigns a new pgid to a
    # backgrounded job even without interactive monitor mode), so -"$pid"
    # here is that whole tree's pgid.
    if $USE_SUDO; then
      sudo kill -TERM -"$pid" 2>/dev/null || true
      sudo kill -TERM   "$pid" 2>/dev/null || true
    else
      kill -TERM -"$pid" 2>/dev/null || true
      kill -TERM   "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  fi
}
trap 'cleanup SIGTERM; exit 143' TERM
trap 'cleanup SIGINT; exit 130' INT

if [[ -n "$LOG_FILE" ]]; then
  run > >(tee "$LOG_FILE") 2>&1 &
else
  run &
fi
RUN_PID=$!
# set +e: under this script's `set -euo pipefail` (see top), `wait`
# returning mpirun's own non-zero exit status would trip errexit and abort
# the script right here, before STATUS=$? (or the trap teardown after it)
# ever runs.
set +e
wait "$RUN_PID"
STATUS=$?
set -e
trap - TERM INT
exit "$STATUS"
