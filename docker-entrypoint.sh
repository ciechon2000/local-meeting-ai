#!/usr/bin/env bash
set -euo pipefail

requested_cache="${MODEL_CACHE:-/cache/models}"
requested_tmp="${TMP_DIR:-/tmp/local-meeting-ai}"

prepare_dir() {
  local requested="$1"
  local fallback="$2"
  local var_name="$3"

  if mkdir -p "$requested" 2>/dev/null && test -w "$requested"; then
    export "$var_name=$requested"
    return 0
  fi

  echo "WARN: $requested is not writable; using $fallback" >&2
  mkdir -p "$fallback"
  chmod 0777 "$fallback" 2>/dev/null || true
  export "$var_name=$fallback"
}

prepare_dir "$requested_cache" "/tmp/local-meeting-ai-models" MODEL_CACHE
prepare_dir "$requested_tmp" "/tmp/local-meeting-ai-runtime" TMP_DIR

export HF_HOME="${HF_HOME:-$MODEL_CACHE/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$MODEL_CACHE/transformers}"
export TORCH_HOME="${TORCH_HOME:-$MODEL_CACHE/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MODEL_CACHE/xdg}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMP_DIR"

echo "Local Meeting AI starting"
echo "MODEL_CACHE=$MODEL_CACHE"
echo "TMP_DIR=$TMP_DIR"
echo "DEVICE=${DEVICE:-auto}"

exec "$@"
