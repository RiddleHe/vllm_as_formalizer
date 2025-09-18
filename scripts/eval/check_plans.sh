#!/usr/bin/env bash

ROOT_DIR="${1:-}"
DATA_PATH="${2:-}"
OUTPUT_DIR="${3:-}"
PY_SCRIPT="${4:-check_plans.py}"
PREFIX="${5:-blocksworld_}"

if [[ -z "${ROOT_DIR}" || -z "${DATA_PATH}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Usage: $0 <root_dir_to_scan> <data_path> <output_dir> [python script]" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
for dir in "$ROOT_DIR"/${PREFIX}*; do
    [[ -d "$dir" ]] || continue

    dirname="$(basename "$dir")"
    out_json="${OUTPUT_DIR}/${dirname}.json"

    echo "Processing $dirname"
    echo "  -> $out_json"

    python3 "$PY_SCRIPT" "$dir" "$DATA_PATH" "$out_json"
done
shopt -u nullglob

echo "Done."