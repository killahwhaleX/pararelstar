# README

See [`old_README.md`](old_README.md) for the original README file for ParaRel.


## Environment
module load PyTorch/1.9.0-fosscuda-2020b

## To get it running
Install all requirements as specified in `requirements.txt`. Also, run the following:

```bash
pip install transformers==4.26.1 -U 
pip install wandb==0.13.3 -U
pip install networkx
```

Remove the `entity` value from the `wandb.init()` on line 51 in `encode_consistency_probe.py` (it seems to try to write to some project that we don't have access to).

Set up a wandb account (you will be prompted to submit the information for this at the first run of the code).

## Run the first test case
As specified in `old_README.md`.

```bash
python -m pararel.consistency.encode_consistency_probe \
       --data_file data/trex_lms_vocab/P106.jsonl \
       --lm bert-base-cased \
       --graph data/pattern_data/graphs/P106.graph \
       --gpu 0 \
       --wandb \
       --use_targets
```

## Debug the first test case
Can only be done from the local Alvis node! (i.e. not in a job due to port problems)

```bash
CUDA_VISIBLE_DEVICES=2, python -m debugpy --wait-for-client --listen 5678 -m pararel.consistency.encode_consistency_probe \
       --data_file data/trex_lms_vocab/P106.jsonl \
       --lm bert-base-cased \
       --graph data/pattern_data/graphs/P106.graph \
       --gpu 0 \
       --wandb \
       --use_targets
```

## Remove duplicates from the ParaRel data

The T-Rex dataset onto which ParaRel is based contains duplicated entries of two different categories, 1) several entries of the same subject but with different object alternatives (N-M relations) and 2) exact duplicates of the same subject and object entries resulting from a loss of granularity when processing the LAMA data.

A [presentation](https://docs.google.com/presentation/d/1fYYA5G3L9qj4IAMLnzlzgrrZrCEF1238EM1ROqVkNYo/edit#slide=id.g20a158dfaae_0_0) on this.

### Examples of duplicates to be removed

| obj_label | relation | relation_name | sub_label | uuid |
| --------- | -------- | ------------- | --------- | ---- |
| Bern | P937	| worked-in | Albert Einstein | e4f33b6d-6cda-4a73-9bd3-5668f884fe0d |
| Berlin | P937 | worked-in	| Albert Einstein | dd080e5d-6e84-46a9-8e2d-33edc11cf03f |
|Berlin | P937 | worked-in | Carl Philipp Emanuel Bach | e63ed2f9-ab68-43e9-a0b8-2a2ab1616c37 |
| Hamburg | P937 | worked-in | Carl Philipp Emanuel Bach | 841751bd-aefc-4e79-a3c5-b90c94336a05 |

### Fix

To fix this we: 
1. remove all entries with subjects that appear more than once in the data for each relation. This amounts to max 10% of the entries of the data for each relation.
2. remove relation P37 "official-language" for which 280 out of 900 entries are duplicates.

This is done by running the [filter_trex_data.py](pararel/consistency/filter_trex_data.py) script as follows:
```bash
python -m pararel.consistency.filter_trex_data --data_path <original-ParaRel-trex_lms_vocab-path> --save_data_path <path>
```

### Data statistics before and after the processing

We can analyze the original and the deduplicated T-REx dataset using [investigate_pararel.ipynb](investigate_pararel.ipynb). See the [presentation](https://docs.google.com/presentation/d/1fYYA5G3L9qj4IAMLnzlzgrrZrCEF1238EM1ROqVkNYo/edit#slide=id.g20a158dfaae_0_0) for the results.

## Save ParaRel prompts to a file

While standing in the root of the pararel folder, run:

```bash
module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

python3 -m pararel.consistency.generate_data \
       --folder_name "all_n1_atlas_no_space" \
       --data_path "data" \
       --relations_given "P937,P1412" \
       --format atlas
```

For debug, add
```bash
CUDA_VISIBLE_DEVICES=2, python -m debugpy --wait-for-client --listen 5678 -m pararel.consistency.generate_data \
       ...
```

## Create training data for Atlas

### Train prompts
'official-language' (P37), 'named-after' (P138), 'original-network' (P449).

       In the paper the authors say that they use 'original-language', while in their code it seems as though they use 'official-language'.

Generated results from atlas can be found under `/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-XX`.

### Test prompts
The rest of the N-1 prompts used by ParaRel.

## Evaluate Atlas

Based on predictions from Atlas. 

```bash
python -m pararel.consistency.encode_consistency_probe_from_file \
       --lm atlas-base \
       --data_file "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-P17-base-2017-901099/P17-step-0.jsonl" \
       --graph "data/pattern_data/graphs/P17.graph" \
       --wandb
```

With debug:
```bash
python -m debugpy --wait-for-client --listen 5678 -m pararel.consistency.encode_consistency_probe_from_file \
       --lm atlas-base \
       --data_file "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3/P138-base-2017-1115926/P138-step-0.jsonl" \
       --graph "data/pattern_data/graphs/P138.graph" \
       --wandb \
       --retriever_statistics
```

Generate ParaRel eval results for all Atlas prediction files in a go:
```bash
./eval_atlas_preds.sh <prefix-of-files-with-predictions, e.g. /cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval->
```

### Atlas results locations
In the folder `/mimer/NOBACKUP/groups/snic2021-23-309/project-data/atlas/experiments`.

Atlas-large: `pararel-eval-zero-shot-large`

Atlas-base: `pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3`

Atlas-base-closed-book: `pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3-closed-book`

T5: `pararel-eval-baseline-t5-no-space-likelihood-no-eos-with-3`


## Save ParaRel prompts with Atlas passage retrievals to a file

While standing in the root of the pararel folder, run:

```bash
module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

python3 -m pararel.consistency.generate_data \
       --folder_name "all_n1_atlas_no_space_w_retrieval" \
       --data_path "data" \
       --format atlas \
       --atlas_data_path "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3"
```

## Save ParaRel prompts with random passage retrievals to a file

While standing in the root of the pararel folder, run:

```bash
module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

python3 -m pararel.consistency.generate_data \
       --folder_name "all_n1_atlas_no_space_w_retrieval_random" \
       --relations_given "P937,P1412" \
       --data_path "data" \
       --format atlas \
       --random_passages_data_paths "/cephyr/users/lovhag/Alvis/projects/atlas/data/corpora/wiki/enwiki-dec2017/text-list-100-sec.jsonl /cephyr/users/lovhag/Alvis/projects/atlas/data/corpora/wiki/enwiki-dec2017/infobox.jsonl"
```