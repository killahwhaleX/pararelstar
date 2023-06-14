#!/usr/bin/env bash

set -eo pipefail

PREDS_FOLDER=$1
R_EMBS_FOLDER=$2 # folder with corresponding retrieval embeddings

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
    python -m pararel.consistency.compute_random_retrieval_consistency \
       --num_samples 1000 \
       --data_file "$filename" \
       --retriever_embeddings_filename "$emb_file" 
done