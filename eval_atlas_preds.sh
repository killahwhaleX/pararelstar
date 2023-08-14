#!/usr/bin/env bash

set -eo pipefail

PREDS_FOLDER=$1
LM_NAME=$2
ADDITIONAL_ARGS=$3 # e.g. "--retriever_statistics"

module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

for filename in ${PREDS_FOLDER}*/*-step-*.jsonl; do
    base_filename=$(basename -- "$filename")
    relation=${base_filename%-step-*\.jsonl}
    echo "Evaluating for relation ${relation}"
    echo "Filename ${filename}"
    python -m pararel.consistency.encode_consistency_probe_from_file \
       --lm $LM_NAME \
       --data_file "$filename" \
       --graph "data/pattern_data/graphs/${relation}.graph" \
       --wandb \
       --options_folder "data/all_n1_atlas_no_space" \
       $ADDITIONAL_ARGS
done