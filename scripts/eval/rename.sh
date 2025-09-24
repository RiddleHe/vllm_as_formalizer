#!/usr/bin/env bash

lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

deduce_model() {
    local path_lc="$(lower "$1")"
    if [[ "$path_lc" == *"gpt-4.1"* ]]; then
        echo "gpt-4.1"
    elif [[ "$path_lc" == *"qwen"* ]]; then
        echo "qwenvl"
    else
        echo "unknown"
    fi
}

deduce_run() {
    local name_lc="$(lower "$1")"
    if [[ "$name_lc" =~ run[_-]?([0-9]+) ]]; then
        echo "run-${BASH_REMATCH[1]}"
    else
        echo "unknown"
    fi
}

deduce_pipeline() {
    local name_lc="$(lower "$1")"
    if [[ "$name_lc" == *"vila"* ]]; then
        echo "direct-plan"
    elif [[ "$name_lc" == *"no_batch"* ]]; then
        echo "scene-graph-multi-step-no-batch"
    elif [[ "$name_lc" == *"multi_step"* ]]; then
        echo "scene-graph-multi-step-batch"
    elif [[ "$name_lc" == *"scene_graph_hard"* ]]; then
        echo "scene-graph"
    elif [[ "$name_lc" == *"caption"* ]]; then
        echo "caption"
    elif [[ "$name_lc" == *"direct_pddl"* ]]; then
        echo "direct-pddl"
    else
        echo "unknown"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then

    set -euo pipefail

    data_dir="${1:-../../results}"
    prefix="${2:-blocksworld_}"

    dataset="${prefix%_}"

    while IFS= read -r -d '' file; do
    bn="$(basename "$file")"
    dirn="$(dirname "$file")"

    if [[ "$bn" != "${prefix}"* ]]; then
        continue
    fi

    model_name="$(deduce_model "$bn")"
    n_run="$(deduce_run "$bn")"
    pipeline="$(deduce_pipeline "$bn")"

    new_name="${model_name}-${dataset}-small-${pipeline}-${n_run}.json"
    target_path="${dirn}/${new_name}"

    echo "Renaming:"
    echo "  $bn"
    echo "    -> $(basename "$target_path")"
    
    if [[ -e "$target_path" ]]; then
        echo "  [skip] target exists: $target_path"
    else
        mv -- "$file" "$target_path"
    fi

    done < <(find "$data_dir" -maxdepth 1 -type f -name '*.json' -print0)

    echo "Done."
fi