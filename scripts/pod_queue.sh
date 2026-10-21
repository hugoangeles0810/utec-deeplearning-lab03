#!/usr/bin/env bash
# Runs on the pod: trains and then evaluates on test every experiment of a list,
# PARALLEL at a time. Each run leaves runs/<name>/train.log and a DONE or FAILED
# marker; runs already DONE are skipped, so relaunching resumes the queue.
#
# Usage: scripts/pod_queue.sh [EXPERIMENTS] [PARALLEL]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"

EXPERIMENTS=${1:-scripts/experiments.txt}
PARALLEL=${2:-1}
export WORKERS=${WORKERS:-4}

now() {
    date +%Y-%m-%dT%H:%M:%S
}
export -f now

run_one() {
    local name=$1
    shift
    local out=runs/$name
    if [[ -f $out/DONE ]]; then
        echo "$(now) $name: ya terminado, se omite"
        return 0
    fi
    mkdir -p "$out"
    rm -f "$out/FAILED"
    echo "$(now) $name: inicio $*"
    if uv run --frozen lab03 train --out "$out" --workers "$WORKERS" "$@" \
        >"$out/train.log" 2>&1 &&
        uv run --frozen lab03 evaluate "$out/best.pt" --workers "$WORKERS" \
            >>"$out/train.log" 2>&1; then
        touch "$out/DONE"
        echo "$(now) $name: listo"
    else
        touch "$out/FAILED"
        echo "$(now) $name: FALLÓ, ver $out/train.log"
    fi
}
export -f run_one

mkdir -p runs
rm -f runs/QUEUE_DONE
echo "$(now) cola: $EXPERIMENTS, $PARALLEL en paralelo"
# Skip comments and blank lines; each line is "<name> [flags...]".
grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$EXPERIMENTS" |
    xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _
touch runs/QUEUE_DONE
echo "$(now) cola: terminada"
