#!/usr/bin/env bash

set -eo pipefail

PREDS_FOLDER=$1
LM_NAME=$2
R_EMBS_FOLDER=$3 # folder with corresponding retrieval embeddings

module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

for filename in ${PREDS_FOLDER}*/*-step-*.jsonl; do
    base_filename=$(basename -- "$filename")
    relation=${base_filename%-step-*\.jsonl}
    echo "Evaluating for relation ${relation}"
    echo "Filename ${filename}"
    emb_file=(${R_EMBS_FOLDER}/${relation}-*/*.pt)
    emb_file=${emb_file%.*}
    echo "Calculating retrieval consistency statistics from ${emb_file}"
    python -m pararel.consistency.encode_consistency_probe_from_file \
       --lm $LM_NAME \
       --data_file "$filename" \
       --graph "data/pattern_data/graphs/${relation}.graph" \
       --wandb \
       --retriever_embeddings_filename "$emb_file" \
       --options_folder "data/all_n1_atlas" \
       --retriever_statistics
done