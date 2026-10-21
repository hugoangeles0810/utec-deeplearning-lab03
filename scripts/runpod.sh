#!/usr/bin/env bash
# Entrena los experimentos de scripts/experiments.txt en un pod GPU de RunPod y
# trae los resultados a runs/.
#
# Uso: scripts/runpod.sh <comando>
#   all [N]       up + push + start + wait + pull + down, N runs a la vez
#   up            crea el pod (muestra el precio de la GPU y pide confirmación)
#   push          instala rsync y uv en el pod, sube código y datos, uv sync
#   start [N]     lanza $EXPERIMENTS en el pod, N runs a la vez (por defecto 2)
#   status        progreso de cada run y uso de la GPU
#   logs RUN      sigue runs/RUN/train.log
#   wait          espera a que termine la cola y luego hace pull
#   pull [--all]  trae runs/ del pod (sin last.pt, salvo con --all)
#   ssh           abre una shell en el pod
#   down          borra el pod (pide confirmación; YES=1 la omite)
#
# Requisitos: runpodctl >= 2.9 con credenciales (RUNPOD_API_KEY o
# `runpodctl doctor`, que además registra la clave SSH), rsync y python3.
# Configurable por variables de entorno: EXPERIMENTS, DATA_SOURCE, GPU_ID, IMAGE,
# CLOUD_TYPE, VOLUME_GB, PARALLEL, POLL_SECONDS, SSH_KEY, POD_WORKSPACE.
#
# Datos: DATA_SOURCE=local (por defecto) sube data/*.h5 desde esta máquina;
# DATA_SOURCE=drive hace que el pod los descargue de Google Drive con gdown, y si
# eso falla sube los locales.
#
# Prueba corta antes de lo largo (verifica GPU, entorno y tiempos):
#   EXPERIMENTS=scripts/experiments_smoke.txt scripts/runpod.sh all 1
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STATE=$ROOT/.runpod
POD_WORKSPACE=${POD_WORKSPACE:-/workspace} # the pod volume
REMOTE=$POD_WORKSPACE/lab03

GPU_ID=${GPU_ID:-NVIDIA GeForce RTX 4090}
# Only provides sshd: the environment comes from uv.lock (uv sync --frozen).
IMAGE=${IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}
# torch 2.14.1 in uv.lock ships CUDA 13 wheels: the host driver must support it.
MIN_CUDA=${MIN_CUDA:-13.0}
CLOUD_TYPE=${CLOUD_TYPE:-SECURE} # --wait needs a public port 22
VOLUME_GB=${VOLUME_GB:-50}
PARALLEL=${PARALLEL:-2}
EXPERIMENTS=${EXPERIMENTS:-scripts/experiments.txt} # relative to the repo
POLL_SECONDS=${POLL_SECONDS:-300}
DATA_SOURCE=${DATA_SOURCE:-local} # local: rsync from data/; drive: gdown on the pod
DRIVE_FOLDER=${DRIVE_FOLDER:-https://drive.google.com/drive/folders/1-Gkgt1HDhsD5Kw2JeBsB_C9kgadGrXfY}
GDOWN_VERSION=6.4.2

# Prepended to every remote script: uv and its caches live on the volume.
# shellcheck disable=SC2016 # $HOME and $PATH expand on the pod
REMOTE_ENV='export PATH="$HOME/.local/bin:$PATH"'"
export UV_CACHE_DIR=$POD_WORKSPACE/.uv-cache UV_PYTHON_INSTALL_DIR=$POD_WORKSPACE/.uv-python"

die() {
    echo "error: $*" >&2
    exit 1
}

need() {
    command -v "$1" >/dev/null || die "falta '$1': $2"
}

confirm() {
    [[ ${YES:-} == 1 ]] && return 0
    local answer
    read -r -p "$1 [y/N] " answer
    [[ $answer == [yY]* ]]
}

pod_id() {
    [[ -f $STATE/pod_id ]] || die "no hay pod: corre 'scripts/runpod.sh up'"
    cat "$STATE/pod_id"
}

# Prints field $1 of the JSON object on stdin, or nothing.
json_field() {
    python3 -c '
import json, sys
text = sys.stdin.read()
for chunk in [text, *text.splitlines()]:
    try:
        data = json.loads(chunk)
    except ValueError:
        continue
    if isinstance(data, dict) and sys.argv[1] in data:
        print(data[sys.argv[1]])
        break
' "$1"
}

# Sets SSH (command array) and HOST from `runpodctl ssh info`.
connect() {
    local id info ip port key
    id=$(pod_id)
    info=$(runpodctl ssh info "$id") || die "runpodctl ssh info $id falló"
    read -r ip port key < <(
        python3 -c '
import json, sys
d = json.load(sys.stdin)
if "error" in d:
    sys.exit("pod no conectable: " + str(d["error"]))
print(d["ip"], d["port"], (d.get("ssh_key") or {}).get("path", ""))
' <<<"$info"
    ) || die "no se pudo leer la conexión SSH del pod $id"
    key=${SSH_KEY:-$key}
    SSH=(ssh -p "$port" -o StrictHostKeyChecking=accept-new
        -o UserKnownHostsFile="$STATE/known_hosts" -o ServerAliveInterval=30)
    if [[ -n $key ]]; then
        SSH+=(-i "$key")
    fi
    HOST=root@$ip
}

# Runs the script on stdin on the pod, after REMOTE_ENV.
remote_sh() {
    { echo "$REMOTE_ENV"; echo "REMOTE=$REMOTE"; cat; } | "${SSH[@]}" "$HOST" bash -s
}

# -a without -o/-g: the pod volume does not allow chown.
push_rsync() {
    rsync -rlptz -e "${SSH[*]}" "$@"
}

cmd_up() {
    need runpodctl "brew install runpod/runpodctl/runpodctl"
    [[ ! -f $STATE/pod_id ]] ||
        die "ya hay un pod ($(cat "$STATE/pod_id")): 'down' antes de crear otro"
    runpodctl user >/dev/null ||
        die "runpodctl sin credenciales: export RUNPOD_API_KEY=... o runpodctl doctor"

    echo "GPU: $GPU_ID · $CLOUD_TYPE · CUDA >= $MIN_CUDA · volumen ${VOLUME_GB} GB"
    echo "Imagen: $IMAGE"
    # Best effort: print the catalog entry of the GPU, which includes its price.
    runpodctl gpu list | python3 -c '
import json, sys
def walk(node):
    if isinstance(node, dict):
        if sys.argv[1] in node.values():
            print(json.dumps(node, indent=2))
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)
walk(json.load(sys.stdin))
' "$GPU_ID" || true
    confirm "¿Crear el pod? Se cobra por segundo hasta 'scripts/runpod.sh down'." ||
        exit 1

    mkdir -p "$STATE"
    local out id
    echo "Creando el pod y esperando a que responda SSH (hasta 20 min)..."
    if ! out=$(runpodctl pod create --name lab03-train --image "$IMAGE" \
        --gpu-id "$GPU_ID" --min-cuda-version "$MIN_CUDA" \
        --cloud-type "$CLOUD_TYPE" --ports 22/tcp \
        --container-disk-in-gb 30 --volume-in-gb "$VOLUME_GB" \
        --volume-mount-path "$POD_WORKSPACE" --wait --wait-timeout 20m \
        2>"$STATE/create.err"); then
        # On timeout the pod is kept and the error carries its id.
        id=$(json_field id <"$STATE/create.err")
        if [[ -n $id ]]; then
            echo "$id" >"$STATE/pod_id"
            echo "El pod $id existe pero no respondió: 'status' o 'down'." >&2
        fi
        die "$(cat "$STATE/create.err")"
    fi
    id=$(json_field id <<<"$out")
    [[ -n $id ]] || die "no se encontró el id del pod en: $out"
    echo "$id" >"$STATE/pod_id"
    echo "Pod $id listo."
}

push_data() {
    case $DATA_SOURCE in
    local)
        local file
        for file in training validation testing; do
            [[ -f $ROOT/data/$file.h5 ]] ||
                die "falta data/$file.h5: descárgalo (data/README.md) o usa DATA_SOURCE=drive"
        done
        echo "Subiendo datos (unos 700 MB la primera vez)..."
        push_rsync --progress "$ROOT"/data/*.h5 "$HOST:$REMOTE/data/"
        ;;
    drive)
        echo "Descargando los datos de Google Drive en el pod..."
        # gdown writes into .drive first, so a file in data/ is always complete.
        if remote_sh <<EOF; then
set -e
cd "\$REMOTE/data"
if [ -f training.h5 ] && [ -f validation.h5 ] && [ -f testing.h5 ]; then
    echo "Los datos ya están en el pod."
    exit 0
fi
rm -rf .drive
uvx --from gdown==$GDOWN_VERSION gdown --folder "$DRIVE_FOLDER" -O .drive
mv .drive/*.h5 .
rm -rf .drive
EOF
            return
        fi
        echo "Falló la descarga de Drive (¿cuota o permisos?): subo los datos locales." >&2
        DATA_SOURCE=local push_data
        ;;
    *)
        die "DATA_SOURCE debe ser local o drive, no '$DATA_SOURCE'"
        ;;
    esac
}

cmd_push() {
    connect
    local uv_version
    uv_version=$(uv --version 2>/dev/null | awk '{print $2}') || true
    echo "Instalando rsync, procps y uv ${uv_version:-(última)} en el pod..."
    remote_sh <<EOF
set -e
if ! command -v rsync >/dev/null || ! command -v pgrep >/dev/null; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync procps >/dev/null
fi
command -v uv >/dev/null ||
    curl -LsSf https://astral.sh/uv/${uv_version:+$uv_version/}install.sh | sh
mkdir -p \$REMOTE/data
EOF
    echo "Subiendo código..."
    push_rsync --delete --exclude __pycache__ \
        "$ROOT/pyproject.toml" "$ROOT/uv.lock" "$ROOT/.python-version" \
        "$ROOT/README.md" "$ROOT/src" "$ROOT/scripts" "$HOST:$REMOTE/"
    push_data
    echo "Instalando dependencias con uv sync..."
    remote_sh <<'EOF'
set -e
cd "$REMOTE"
uv sync --frozen --no-dev
uv run --frozen python -c '
import torch
assert torch.cuda.is_available(), "torch no ve la GPU: revisa la versión de CUDA"
print("torch", torch.__version__, "·", torch.cuda.get_device_name())
'
uv run --frozen python -c '
import h5py
for split, key in [("training", "images"), ("validation", "query/images"),
                   ("testing", "query/images")]:
    with h5py.File(f"data/{split}.h5", "r") as file:
        print(f"data/{split}.h5: {key} {file[key].shape}")
'
EOF
}

cmd_start() {
    local parallel=${1:-$PARALLEL}
    connect
    remote_sh <<EOF
cd "\$REMOTE"
if pgrep -f pod_queue.sh >/dev/null; then
    echo "La cola ya está corriendo." >&2
    exit 1
fi
mkdir -p runs
setsid nohup bash scripts/pod_queue.sh "$EXPERIMENTS" $parallel \
    >>runs/queue.log 2>&1 </dev/null &
echo "Cola $EXPERIMENTS lanzada ($parallel en paralelo). Progreso: scripts/runpod.sh status"
EOF
}

# Prints done, running or stopped.
queue_state() {
    remote_sh <<'EOF'
if [ -f "$REMOTE/runs/QUEUE_DONE" ]; then
    echo done
elif pgrep -f pod_queue.sh >/dev/null; then
    echo running
else
    echo stopped
fi
EOF
}

cmd_status() {
    connect
    echo "Cola: $(queue_state)"
    remote_sh <<'EOF'
cd "$REMOTE/runs" 2>/dev/null || { echo "Todavía no hay runs."; exit 0; }
for dir in */; do
    run=${dir%/}
    [ -d "$run" ] || continue
    state="en curso"
    [ -f "$run/DONE" ] && state=listo
    [ -f "$run/FAILED" ] && state=error
    epoch=$(grep '^epoch' "$run/train.log" 2>/dev/null | tail -n 1 | cut -d' ' -f2)
    val=$(grep 'val query' "$run/train.log" 2>/dev/null | tail -n 1 | sed 's/^ *val //')
    printf '%-14s %-9s época %-6s %s\n' "$run" "$state" "${epoch:--}" "$val"
done
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader | sed 's/^/GPU: /'
EOF
}

cmd_logs() {
    [[ -n ${1:-} ]] || die "uso: scripts/runpod.sh logs RUN"
    connect
    "${SSH[@]}" "$HOST" "tail -n 50 -f $REMOTE/runs/$1/train.log"
}

cmd_pull() {
    connect
    local exclude=(--exclude last.pt)
    if [[ ${1:-} == --all ]]; then
        exclude=()
    fi
    mkdir -p "$ROOT/runs"
    rsync -rlptz --progress -e "${SSH[*]}" ${exclude[@]+"${exclude[@]}"} \
        "$HOST:$REMOTE/runs/" "$ROOT/runs/"
    echo "Resultados en runs/. Siguiente paso: notebooks/analysis.ipynb"
}

cmd_wait() {
    connect
    echo "Esperando la cola (cada ${POLL_SECONDS}s). Ctrl-C deja el pod corriendo."
    local state
    while true; do
        # A dropped connection just means "try again later".
        state=$(queue_state 2>/dev/null) || state=unreachable
        case $state in
        done) break ;;
        stopped) die "la cola se detuvo sin terminar: revisa 'status' y runs/queue.log" ;;
        esac
        echo "$(date +%H:%M) $state"
        sleep "$POLL_SECONDS"
    done
    cmd_status
    cmd_pull
}

cmd_ssh() {
    connect
    exec "${SSH[@]}" -t "$HOST" "cd $REMOTE; exec bash -l"
}

cmd_down() {
    local id
    id=$(pod_id)
    confirm "¿Borrar el pod $id? Se pierde lo que no hayas traído con 'pull'." ||
        exit 1
    runpodctl pod delete "$id" >/dev/null
    rm -f "$STATE/pod_id" "$STATE/known_hosts"
    echo "Pod $id borrado."
}

cmd_all() {
    cmd_up
    # From here on the pod bills: never leave without saying so.
    trap 'echo "El pod $(cat "$STATE/pod_id" 2>/dev/null) sigue corriendo:" \
        "scripts/runpod.sh status | pull | down" >&2' EXIT
    cmd_push
    cmd_start "${1:-$PARALLEL}"
    cmd_wait
    trap - EXIT
    YES=1 cmd_down
}

main() {
    local command=${1:-}
    shift || true
    case $command in
    all | up | push | start | status | logs | pull | wait | ssh | down)
        "cmd_$command" "$@"
        ;;
    *)
        sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "$0"
        [[ -z $command ]] || exit 1
        ;;
    esac
}

main "$@"
