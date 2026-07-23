#!/bin/bash
#
# Build an MPQC SIF directly via Apptainer (no docker daemon).
#
# Reads the Apptainer recipe at `bin/mpqc.def` and substitutes the
# `{{ VAR }}` placeholders from CLI flags + env. Tokens are passed
# through env into the apptainer process arg list — they never land
# in the repo or on disk outside apptainer's own build tempdir.
#
# Usage:
#   bin/build-mpqc-sif.sh [OPTIONS] [GIT_REF] [TAG]
#
# Options:
#   -o, --output PATH     output SIF path (default: ./mpqc-<tag>.sif)
#   -j, --jobs N          parallel build jobs (default: ninja's own)
#   -u, --url URL         MPQC git URL (default: jianjianh1/mpqc-fork, the
#                         instrumented fork this repo's tooling expects)
#   -b, --backend NAME    MADNESS task backend (default: PaRSEC)
#   --tmpdir DIR          apptainer build scratch dir (default: /tmp)
#   --keep-cache          don't pass --disable-cache to apptainer
#   --dry-run             print the apptainer build command and exit
#   -h, --help            print this help and exit
#
# Args:
#   GIT_REF   git ref (branch / tag / SHA) to build (default: 1ef4e722f4,
#             the fork commit with per-residual SumInplace timing + a
#             whole-residual checksum line)
#   TAG       label used in default --output filename (default: latest)
#
# Env:
#   GH_USER   GitHub username for private-fork access (default: git)
#   GH_TOKEN  PAT for private-fork access. If set, basic-auth is
#             injected into the clone URL during apptainer build only.
#
# Examples:
#   # Instrumented fork's pinned commit, public clone, default backend/output.
#   bin/build-mpqc-sif.sh
#
#   # Our fork's branch, into a known location:
#   GH_USER=jianjianh1 GH_TOKEN=$YOUR_PAT \
#     bin/build-mpqc-sif.sh \
#       -u https://github.com/jianjianh1/mpqc-fork.git \
#       -o /proj/.../apptainer/repro.sif \
#       batched-tn-eval repro
#
# See bin/REPRODUCING.md for the end-to-end recipe and prereqs.

set -euo pipefail

# Defaults
DEF_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mpqc.def"
OUTPUT=""
JOBS=""
URL="https://github.com/jianjianh1/mpqc-fork.git"
BACKEND="PaRSEC"
TMPDIR_ARG="/tmp"
KEEP_CACHE=false
DRY_RUN=false
REF=""
TAG=""
SEQUANT_PATCH_DIR=""

usage() {
  cat <<'EOF'
Build an MPQC SIF directly via Apptainer (no docker daemon).

Reads the Apptainer recipe at `bin/mpqc.def` and substitutes the
`{{ VAR }}` placeholders from CLI flags + env. Tokens are passed
through env into the apptainer process arg list — they never land
in the repo or on disk outside apptainer's own build tempdir.

Usage:
  bin/build-mpqc-sif.sh [OPTIONS] [GIT_REF] [TAG]

Options:
  -o, --output PATH     output SIF path (default: ./mpqc-<tag>.sif)
  -j, --jobs N          parallel build jobs (default: ninja's own)
  -u, --url URL         MPQC git URL (default: jianjianh1/mpqc-fork, the
                        instrumented fork this repo's tooling expects)
  -b, --backend NAME    MADNESS task backend (default: PaRSEC)
  --tmpdir DIR          apptainer build scratch dir (default: /tmp)
  --sequant-patch-dir DIR  build against a locally-patched SeQuant checkout
                        (via CMake's FETCHCONTENT_SOURCE_DIR_SEQUANT)
                        instead of fetching the pinned upstream tag. DIR
                        must be under --tmpdir (that's the only host path
                        Apptainer bind-mounts into the %post sandbox).
  --keep-cache          don't pass --disable-cache to apptainer
  --dry-run             print the apptainer build command and exit
  -h, --help            print this help and exit

Args:
  GIT_REF   git ref (branch / tag / SHA) to build (default: 1ef4e722f4,
            the fork commit with per-residual SumInplace timing + a
            whole-residual checksum line)
  TAG       label used in default --output filename (default: latest)

Env:
  GH_USER   GitHub username for private-fork access (default: git)
  GH_TOKEN  PAT for private-fork access. If set, basic-auth is
            injected into the clone URL during apptainer build only.
            If unset and the URL is on github.com, falls back to
            `gh auth token` (when the gh CLI is installed + logged in).

Examples:
  # Instrumented fork's pinned commit, public clone, default backend/output.
  bin/build-mpqc-sif.sh

  # Our fork's branch, into a known location:
  GH_USER=jianjianh1 GH_TOKEN=$YOUR_PAT \
    bin/build-mpqc-sif.sh \
      -u https://github.com/jianjianh1/mpqc-fork.git \
      -o /proj/.../apptainer/repro.sif \
      batched-tn-eval repro

See bin/REPRODUCING.md for the end-to-end recipe and prereqs.
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)     OUTPUT="$2"; shift 2 ;;
    -j|--jobs)       JOBS="$2"; shift 2 ;;
    -u|--url)        URL="$2"; shift 2 ;;
    -b|--backend)    BACKEND="$2"; shift 2 ;;
    --tmpdir)        TMPDIR_ARG="$2"; shift 2 ;;
    --sequant-patch-dir) SEQUANT_PATCH_DIR="$2"; shift 2 ;;
    --keep-cache)    KEEP_CACHE=true; shift ;;
    --dry-run)       DRY_RUN=true; shift ;;
    -h|--help)       usage 0 ;;
    -*)              echo "unknown option: $1" >&2; usage 1 ;;
    *)
      if [[ -z "$REF" ]]; then REF="$1"
      elif [[ -z "$TAG" ]]; then TAG="$1"
      else echo "extra positional arg: $1" >&2; usage 1
      fi
      shift
      ;;
  esac
done

REF="${REF:-1ef4e722f4}"
TAG="${TAG:-latest}"
OUTPUT="${OUTPUT:-./mpqc-${TAG}.sif}"

# Pre-flight checks
if [[ ! -f "$DEF_FILE" ]]; then
  echo "ERROR: missing recipe $DEF_FILE" >&2
  exit 1
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found in PATH" >&2
  exit 1
fi
APPTAINER_VER="$(apptainer --version | awk '{print $NF}')"
echo "apptainer version: $APPTAINER_VER"

# Disk-headroom advisory (the actual minimum varies, but the libint2
# unity build + MADNESS + TA peak at ~15-20 GB in $TMPDIR_ARG).
# Use `df --output=avail` (coreutils ≥ 8.21) to avoid the awk-on-NR==2
# trap, which breaks when the device name is long and df wraps the
# header onto two lines.
if avail_kb=$(df -kP --output=avail "$TMPDIR_ARG" 2>/dev/null | tail -n 1 | tr -d '[:space:]'); then
  if [[ "$avail_kb" =~ ^[0-9]+$ ]]; then
    avail_gb=$(( avail_kb / 1024 / 1024 ))
    if (( avail_gb < 20 )); then
      echo "WARNING: only ${avail_gb} GB free in --tmpdir=$TMPDIR_ARG;"
      echo "         apptainer build typically needs ~20 GB scratch."
    fi
  fi
fi

# Resolve token + clone-url defaults. GH_USER defaults to a sensible
# value depending on whether the token looks like a GitHub OAuth PAT.
GH_USER="${GH_USER:-}"
GH_TOKEN="${GH_TOKEN:-}"
# If the URL is on github.com and no GH_TOKEN was given, fall back to
# `gh auth token` (the GitHub CLI's stored PAT) — saves the user from
# having to extract it manually. Silently no-op if gh isn't installed
# or isn't logged in.
if [[ -z "$GH_TOKEN" && "$URL" == https://github.com/* ]]; then
  if command -v gh >/dev/null 2>&1 && gh auth token >/dev/null 2>&1; then
    GH_TOKEN="$(gh auth token 2>/dev/null)"
    if [[ -n "$GH_TOKEN" ]]; then
      echo "GH_TOKEN sourced from \`gh auth token\` (set GH_TOKEN explicitly to override)"
    fi
  fi
fi
if [[ -n "$GH_TOKEN" && -z "$GH_USER" ]]; then
  # GitHub accepts any non-empty username with a PAT in basic-auth
  # form (`https://USER:PAT@github.com/...`). The conventional
  # placeholder is `git`.
  GH_USER="git"
fi

# Apptainer's `{{ VAR }}` substitution is textual: the build-arg value
# is spliced into the rendered %post script as-is. The recipe stores
# each value in a single-quoted shell variable (`_FOO='{{ FOO }}'`),
# so a single quote in the value would close the string early and
# inject arbitrary shell. None of our intended inputs (GitHub usernames,
# PATs, "PaRSEC"/"Pthreads", positive integers, https:// URLs) contain
# a single quote, so we reject any value that does — caught here instead
# of at apptainer parse time, which would error with a confusing token.
for var in URL REF BACKEND JOBS GH_USER GH_TOKEN SEQUANT_PATCH_DIR; do
  val="${!var:-}"
  if [[ "$val" == *"'"* ]]; then
    echo "ERROR: $var value contains a single quote, which mpqc.def cannot safely substitute" >&2
    exit 1
  fi
done

# SEQUANT_PATCH_DIR must be visible inside the %post sandbox, and the
# only host path Apptainer bind-mounts there is --tmpdir — catch a
# mistake here with a clear message rather than mpqc.def's deeper,
# less obvious "not visible inside the build sandbox" error.
if [[ -n "$SEQUANT_PATCH_DIR" ]]; then
  [[ -d "$SEQUANT_PATCH_DIR" ]] || { echo "ERROR: --sequant-patch-dir not found: $SEQUANT_PATCH_DIR" >&2; exit 1; }
  # Resolve both to real paths before checking containment — a lexical
  # glob match on the raw strings can be fooled by a `..`-containing value
  # that lexically starts with "$TMPDIR_ARG/" but actually resolves outside
  # the tree Apptainer bind-mounts (e.g. "$TMPDIR_ARG/../etc" passes a
  # `case "$SEQUANT_PATCH_DIR" in "$TMPDIR_ARG"/*)` match and `[[ -d ]]`
  # equally well, while pointing somewhere else entirely).
  resolved_patch_dir="$(realpath "$SEQUANT_PATCH_DIR")"
  resolved_tmpdir="$(realpath "$TMPDIR_ARG")"
  case "$resolved_patch_dir" in
    "$resolved_tmpdir"/*) ;;
    *)
      echo "ERROR: --sequant-patch-dir ($SEQUANT_PATCH_DIR, resolves to $resolved_patch_dir) must be under --tmpdir ($TMPDIR_ARG, resolves to $resolved_tmpdir) — that's the only host path Apptainer bind-mounts into the build sandbox" >&2
      exit 1
      ;;
  esac
fi

# Build the apptainer build-arg list. Token never echoed.
BUILD_ARGS=(
  --build-arg "MPQC_GIT_URL=${URL}"
  --build-arg "MPQC_GIT_REF=${REF}"
  --build-arg "MADNESS_TASK_BACKEND=${BACKEND}"
  --build-arg "BUILD_JOBS=${JOBS}"
  --build-arg "GH_USER=${GH_USER}"
)
if [[ -n "$GH_TOKEN" ]]; then
  BUILD_ARGS+=( --build-arg "GH_TOKEN=${GH_TOKEN}" )
fi
if [[ -n "$SEQUANT_PATCH_DIR" ]]; then
  BUILD_ARGS+=( --build-arg "SEQUANT_PATCH_DIR=${SEQUANT_PATCH_DIR}" )
fi

APPTAINER_FLAGS=( --tmpdir="$TMPDIR_ARG" )
if ! $KEEP_CACHE; then
  APPTAINER_FLAGS+=( --disable-cache )
fi

# Echo a redacted command line for the log. The `[@]/PAT/REPL` form
# applies the substitution to each array element independently — only
# the `GH_TOKEN=...` element matches, the rest pass through.
ECHO_ARGS=( "${BUILD_ARGS[@]/#GH_TOKEN=*/GH_TOKEN=***}" )
echo "+ sudo apptainer build ${APPTAINER_FLAGS[*]} ${ECHO_ARGS[*]} $OUTPUT $DEF_FILE"

if $DRY_RUN; then
  echo "(dry-run: not executing)"
  exit 0
fi

# Run.  Note: keep `sudo` outside the apptainer args so the env vars
# carry through correctly.
sudo apptainer build "${APPTAINER_FLAGS[@]}" \
                     "${BUILD_ARGS[@]}" \
                     "$OUTPUT" "$DEF_FILE"

echo
echo "SIF built: $OUTPUT"
echo "Quick sanity check:"
sudo apptainer exec "$OUTPUT" /home/mpqc/install/bin/mpqc --version 2>&1 \
  | head -3 || true
