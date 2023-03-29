#!/usr/bin/env bash

set -eo pipefail

PREDS_FOLDER=$1

for filename in ${PREDS_FOLDER}*/*-step-0.jsonl; do
    base_filename=$(basename -- "$filename")
    relation=${base_filename%-step-0\.jsonl}
    echo "Evaluating for relation ${relation}"
    python -m pararel.consistency.encode_consistency_probe_from_file \
       --lm atlas-base \
       --data_file "$filename" \
       --graph "data/pattern_data/graphs/${relation}.graph" \
       --wandb
done