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

## Save ParaRel prompts to a file

While standing in the root of the pararel folder, run:

```bash
module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

python3 -m pararel.consistency.generate_data \
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
       --data_file "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-P17-base-2017-901099/P17-step-0.jsonl" \
       --graph "data/pattern_data/graphs/P17.graph" \
       --wandb
```

All files in a go:
```bash
./eval_atlas_preds.sh /cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-
```