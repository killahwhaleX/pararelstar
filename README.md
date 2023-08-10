# Updated ParaRel for evaluating consistency of Atlas and LLaMA

This repo describes how we create an improved version of ParaRel and how we evaluate the consistency of Atlas and LLaMA on this improved ParaRel. 

See the [old_README.md](old_README.md) for the original README file for the ParaRel repo.

## Setup 

### Environment
Assumes that you are using a HPC with pre-installed modules. 

Create a virtual environment `venv` in the working directory and run the following commands:

```bash
module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate
pip install -r requirements.txt
pip install transformers==4.26.1 -U 
pip install wandb==0.13.3 -U
pip install networkx
```

Set up a wandb account (you will be prompted to submit the information for this at the first run of the code).

## Run the first test case
As specified in [old_README.md](old_README.md):

```bash
python -m pararel.consistency.encode_consistency_probe \
       --data_file data/trex_lms_vocab/P106.jsonl \
       --lm bert-base-cased \
       --graph data/pattern_data/graphs/P106.graph \
       --gpu 0 \
       --wandb \
       --use_targets
```

## Debug example
Can only be done from a local HPC node, i.e. not in a job due to lack of port visibility.

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

The T-Rex dataset onto which ParaRel is based contains duplicated entries of two different categories, 1) several entries of the same subject but with different object alternatives (N-M relations) and 2) exact duplicates of the same subject and object entries resulting from a loss of granularity when processing the LAMA data. See a [presentation](https://docs.google.com/presentation/d/1fYYA5G3L9qj4IAMLnzlzgrrZrCEF1238EM1ROqVkNYo/edit#slide=id.g20a158dfaae_0_0) on this. [investigate_pararel.ipynb](investigate_pararel.ipynb) is used to count the number of duplicates.

### Examples of duplicates

| obj_label | relation | relation_name | sub_label | uuid |
| --------- | -------- | ------------- | --------- | ---- |
| Bern | P937	| worked-in | Albert Einstein | e4f33b6d-6cda-4a73-9bd3-5668f884fe0d |
| Berlin | P937 | worked-in	| Albert Einstein | dd080e5d-6e84-46a9-8e2d-33edc11cf03f |
|Berlin | P937 | worked-in | Carl Philipp Emanuel Bach | e63ed2f9-ab68-43e9-a0b8-2a2ab1616c37 |
| Hamburg | P937 | worked-in | Carl Philipp Emanuel Bach | 841751bd-aefc-4e79-a3c5-b90c94336a05 |

### Remove duplicates

We perform the following steps to remove duplicates from the ParaRel data:
1. remove all entries with subjects that appear more than once in the data for each relation. This amounts to max 10% of the entries of the data for each relation.
2. remove relation P37 "official-language" for which 280 out of 900 entries are duplicates.

This is done by running the [filter_trex_data.py](pararel/consistency/filter_trex_data.py) script as follows:
```bash
python -m pararel.consistency.filter_trex_data --data_path <original-ParaRel-trex_lms_vocab-path> --save_data_path <path>
```

> We use this deduplicated dataset for the remainder of the analysis described below by replacing `data/trex_lms_vocab` with the deduplicated version.

### Data statistics before and after the processing

We analyze the original and the deduplicated T-REx dataset using [pararel_duplicates/investigate_pararel.ipynb](pararel_duplicates/investigate_pararel.ipynb). See the [presentation](https://docs.google.com/presentation/d/1fYYA5G3L9qj4IAMLnzlzgrrZrCEF1238EM1ROqVkNYo/edit#slide=id.g20a158dfaae_0_0) for the results.

## Evaluate models external to ParaRel

This repo supports evaluation of e.g. BERT-base using [pararel/consistency/encode_consistency_probe.py](pararel/consistency/encode_consistency_probe.py). For evaluation of external models such as Atlas, T5 and LLaMA we need to use another approach.

The process for evaluating models external to the ParaRel repository builds on three steps. First, generate and save ParaRel prompts to files by relation. Second, evaluate the model on these prompts and store the predictions. Third, go back to the ParaRel repo and process the generated predictions using functions to get the final ParaRel results.

### 1. Save ParaRel prompts to a file

Recall that the ParaRel evaluation relies on measuring model predictions across query paraphrases related to the same fact triple consisting of a subject, relation and object. Examples of two paraphrases for the relation P17 are "[subject] is located in [object]." and "[subject], which is located in [object].", where the subject and object for example could be "L'Escala" and "Spain" respectively.

In this section we generate and save the queries for all ParaRel relations built on the corresponding paraphrase templates and fact triplets. Different models require different cloze-style query formats why we also take regard to this when generating the queries.

While standing in the root of the ParaRel folder, run:

```bash
module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

python3 -m pararel.consistency.generate_data \
       --folder_name "all_n1_atlas_no_space" \
       --data_path "data" \
       --relations_given "P937,P1412" \
       --format atlas
```

The script will then base the generated queries on the files located under `data/trex_lms_vocab` and `data/pattern_data`. Remove the `--relations_given` argument to generate prompts for all relations.

For debug, add
```bash
CUDA_VISIBLE_DEVICES=2, python -m debugpy --wait-for-client --listen 5678 -m pararel.consistency.generate_data \
       ...
```

### 2. Generate model predictions

- To get the Atlas and T5 predictions we use the [this repo](https://github.com/lovhag/atlas).
- To get the LLaMA predictions we use [this repo](https://github.com/TobiasNorlund/llama/tree/main).

**Atlas results locations**

In the folder `/mimer/NOBACKUP/groups/snic2021-23-309/project-data/atlas/experiments`.

Atlas-large: `pararel-eval-zero-shot-large`

Atlas-base: `pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3`

Atlas-base-closed-book: `pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3-closed-book`

T5: `pararel-eval-baseline-t5-no-space-likelihood-no-eos-with-3`

### 3. Process model predictions

Use [pararel/consistency/encode_consistency_probe_from_file.py](pararel/consistency/encode_consistency_probe_from_file.py) to process the generated model predictions and get the ParaRel metrics. These will be saved to your wandb project, after which you can download the full results from the wandb webpage.

For Atlas evaluations, you also have the option of adding the `--retriever_statistics` flag which means that more metrics related to retriever consistency will be added to the computations. This also means that the evaluations will take a longer time to run, so only use this if necessary.

#### Atlas example

Using the predictions from Atlas, we can generate results for e.g. relation P17 as follows.

```bash
python -m pararel.consistency.encode_consistency_probe_from_file \
       --lm atlas-base \
       --data_file "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-P17-base-2017-901099/P17-step-0.jsonl" \
       --graph "data/pattern_data/graphs/P17.graph" \
       --wandb
```

#### Atlas example with debug

```bash
python -m debugpy --wait-for-client --listen 5678 -m pararel.consistency.encode_consistency_probe_from_file \
       --lm test-atlas-base \
       --data_file "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3/P140-base-2017-1115950/P140-step-0.jsonl" \
       --graph "data/pattern_data/graphs/P140.graph" \
       --wandb \
       --retriever_statistics \
       --options_folder "data/all_n1_atlas"
```

```bash
python -m debugpy --wait-for-client --listen 5678 -m pararel.consistency.encode_consistency_probe_from_file \
       --lm test-atlas-base \
       --data_file "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3/P140-base-2017-1115950/P140-step-0.jsonl" \
       --graph "data/pattern_data/graphs/P140.graph" \
       --wandb \
       --retriever_statistics \
       --retriever_embeddings_filename /cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-compute-r-embeddings-base/P140-2017-1144078/P140-step-0-r-embedding \
       --options_folder "data/all_n1_atlas"
```

#### Atlas example generating ParaRel evaluation results for all ParaRel relations in a go
```bash
./eval_atlas_preds.sh <prefix-of-files-with-predictions, e.g. /cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval->
```

## Generate ParaRel result plots

First, download the ParaRel results you wish to plot as a csv file from the wandb project. Then use [generate_result_plots.ipynb](generate_result_plots.ipynb) to generate result tables and plots based on that data. 

## Add quality annotations to ParaRel metrics

We notice that several queries in the ParaRel dataset suffer from different issues related to:
- subject-object similarity such as "Nokia N9 was produced by Nokia", 
- unidiomatic language usage with respect to 
       - object such as "Solar Mass is named after _sun_",
       - template such as "Anne Redpath died in Edinburgh" vs. "Anne Redpath died at Edinburgh", and 
- semantic overlap in the restricted answer candidate set such as allowing both the answers "Glasgow" and "Scotland" for relation P19 "born-in".

These issues reside on different levels in the ParaRel dataset, e.g. semantic overlap in the restricted candidate set for a relation impacts all evaluations for that relation while unidiomatic language usage with respect to an object only impact queries related to that object for the given relation. We manually annotate potential issues on a relation, object and template level respectively covering the following issues:
- relation level
       - subject-object similarity, mark a relation as problematic if this issue frequently occurs for the ParaRel queries.
       - semantic overlap in the restricted candidate set, mark a relation as problematic if answer candidates overlap semantically.
- template level
       - unidiomatic language usage with respect to template, mark a template as problematic if it can be expected to work poorly with several of the subject-object pairs for the given relation.
- object level
       - unidiomatic language usage with respect to object, mark an object as problematic if it can be expected to work poorly with several of the templates for the given relation.

TODO: #Denitsa add more info on this and the corresponding data files with code for getting Figure 3 in the submission.

# Atlas specific evaluations

For the Atlas model we run additional ParaRel evaluations that also take regard to the retriever component of the model.

## Validate the Atlas answer decoding

A maximum likelihood based decoding is used to generate Atlas predictions from a restricted candidate set to the cloze-style queries of ParaRel, as described in the [Atlas repo](https://github.com/lovhag/atlas). To ascertain that it works as expected, we compare freely generated Atlas answers to the constrained generations by manual inspection using [investigate_atlas_predictions.ipynb](investigate_atlas_predictions.ipynb).

## Fixed retrieved passages for Atlas

We experiment with three different settings for fixing the retrieved passages and overriding the Atlas retriever. For this, we first need to generate ParaRel queries complemented with corresponding fixed retrieved passages.

### 1. Relevant and consistent
Save ParaRel prompts with relevant fixed retrieved passages to a file by running:

```bash
module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

python3 -m pararel.consistency.generate_data \
       --folder_name "all_n1_atlas_no_space_w_retrieval" \
       --data_path "data" \
       --format atlas \
       --atlas_data_path "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3"
```

This script requires that you have previously generated Atlas predictions for the data at hand in order to get reasonable retrieved passages.

### 2. Not relevant but cohesive and consistent

Save ParaRel prompts with a random passage retrieval of 20 passages from Atlas to a file:

```bash
module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

python3 -m pararel.consistency.generate_data \
       --folder_name "all_n1_atlas_no_space_w_retrieval_semi_random" \
       --relations_given "P937,P1412" \
       --data_path "data" \
       --format atlas \
       --atlas_data_path "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3" \
       --random_atlas_retrieval
```

In this setting, the 20 retrieved passages all align, compared to the setting below for which all 20 passages could be retrieved for 20 different fact triplets.

### 3. Not relevant or cohesive but consistent 

Save ParaRel prompts with completely random retrieved passages to a file:

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

## Atlas retriever consistency

We also wish to investigate to what extent the passages retrieved by the Atlas retriever change depending on query paraphrase. Initially, we use [investigate_atlas_predictions.ipynb](investigate_atlas_predictions.ipynb) to manually inspect the retrieved passages, before moving over to the more automated measurements described below.

### Evaluate Atlas ParaRel performance with regard to retriever consistency measured using an embeddings based metric

We add different metrics to the ParaRel evaluation to get more fine-grained results. In this setting we also analyze the retriever consistency by generating retriever passage embeddings and comparing those against each other.

To add this functionality to the ParaRel evaluation add the `--retriever_embeddings_filename` argument pointing to the location of the embeddings and add the `--retriever_statistics` flag to indicate that retriever statistics should be run.

``` bash
./eval_atlas_preds_w_r_emb_sim.sh /cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-zero-shot-large/ atlas-large /cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-compute-r-embeddings-large
```

### Investigate retriever consistency metrics for random passage-subject pairs across relations

To complement the retriever consistency evaluations on passages retrieved by the Atlas retriever we compare against a random baseline by measuring the embedding similarity for random non-related passage sets. 

#### Code example for one relation

We use the `compute_random_retrieval_consistency.py` script to measure retrieval consistency metrics for random passage sets, as examplified for relation P361 below.

```bash
module load PyTorch/1.9.0-fosscuda-2020b
source venv/bin/activate

python -m debugpy --wait-for-client --listen 5678 -m pararel.consistency.compute_random_retrieval_consistency \
       --num_samples 1000 \
       --data_file "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3/P361-base-2017-1115954/P361-step-0.jsonl" \
       --retriever_embeddings_filename "/cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-compute-r-embeddings-base/P361-2017-1144082/P361-step-0-r-embedding" \
```

#### Code example for all relations

To compute the consistency metrics for all relations in a go, use the script as seen below.

```bash
./compute_random_consistencies.sh /cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3/ /cephyr/users/lovhag/Alvis/projects/atlas/data/experiments/pararel-compute-r-embeddings-base
```