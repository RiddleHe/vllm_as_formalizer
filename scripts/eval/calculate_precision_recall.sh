#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-}"
OUTPUT_DIR="${2:-../../results/precision_recall}"
PY_SCRIPT="${3:-calculate_precision_recall.py}"
MAPPING_DIR="${4:-../../results/success_rates}"
RENAME_HELPERS="${5:-./rename.sh}"

source "$RENAME_HELPERS"

PREFIXES=("blocksworld_" "cooking_" "alfred_" "blocksworld-real")
DATA_PATHS=("/local-ssd/mh3897/alfred/blocksworld" "/local-ssd/mh3897/alfred/cooking" "/local-ssd/mh3897/alfred/alfred_137" "/local-ssd/mh3897/alfred/blocksworld-real")
MODEL_NAMES=("gpt-4.1" "qwen")

if [[ -z "${ROOT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Usage: $0 <root_dir_to_scan> <output_dir> [python script]" >&2
    exit 1
fi

if (( ${#PREFIXES[@]} != ${#DATA_PATHS[@]} )); then
    echo "Error: PREFIXES and DATA_PATHS must have same length." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

for i in "${!PREFIXES[@]}"; do
    prefix="${PREFIXES[$i]}"
    data_path="${DATA_PATHS[$i]}"
    dataset="${prefix%_}"

    echo "=== Processing prefix '$prefix' with data '$data_path' ==="

    shopt -s nullglob
    for dir in "$ROOT_DIR"/${prefix}*; do
        [[ -d "$dir" ]] || continue

        dirname="$(basename "$dir")"

        match=false
        for model in "${MODEL_NAMES[@]}"; do
            if [[ "$dirname" == *"$model"* ]]; then
                match=true
                break
            fi
        done
        [[ "$match" == true ]] || continue

        model_name="$(deduce_model "$dirname")"
        run_tag="$(deduce_run "$dirname")"
        pipeline_tag="$(deduce_pipeline "$dirname")"
        mapping_json="${MAPPING_DIR}/${model_name}-${dataset}-${pipeline_tag}-${run_tag}.json"

        if [[ ! -f "$mapping_json" ]]; then
            echo "Warning: mapping not found, skipping $dirname"
            continue
        fi

        out_json="${OUTPUT_DIR}/${dirname}.json"

        echo "Processing $dirname"
        echo "  -> $out_json"

        python3 "$PY_SCRIPT" "$dir" "$data_path" "$out_json"
    done
    shopt -u nullglob
done

echo "Done."