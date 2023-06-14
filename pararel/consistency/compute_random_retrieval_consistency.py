import argparse
import json
from collections import Counter
from collections import defaultdict
from typing import List, Dict

import numpy as np
import wandb
import pandas as pd
import torch
import random
from scipy.stats import pearsonr

from pararel.consistency.utils import read_jsonl_file
from pararel.consistency.encode_consistency_probe import filter_a_an_vowel_mismatch, get_r_embeddings_similarity
from pararel.consistency.encode_consistency_probe_from_file import read_atlas_results

def log_wandb(args, sample_across):
    pattern = args.data_file.split('/')[-1].split('.')[0].split('-')[0]
    tags_list = [pattern]

    config = dict(
        pattern=pattern,
        sample_across=sample_across
    )

    wandb.init(
        name=f'{pattern}_random_check_sampled_across_{sample_across}',
        project="r-consistency",
        tags=tags_list,
        config=config,
    )

def get_r_consistency_id_performance(psgs_ids, psgs_ids_to_compare):
    return len(set(psgs_ids) & set(psgs_ids_to_compare))/len(psgs_ids)

def get_r_consistency_title_performance(psgs_titles, psgs_titles_to_compare):
    overlapping_titles = set(psgs_titles) & set(psgs_titles_to_compare)
    num_overlap_titles = sum([psgs_titles.count(title)+psgs_titles_to_compare.count(title) for title in overlapping_titles])
    return num_overlap_titles/(len(psgs_titles)+len(psgs_titles_to_compare))

def sample_without_replacement(curr_val, available_vals):
    curr_ix = available_vals.index(curr_val)
    other_val = random.sample(available_vals[:curr_ix]+available_vals[curr_ix+1:], 1)[0]
    return other_val

def get_consistency_correlations(consistency_performance_1, consistency_performance_2):
    p_thresh = 0.05
        
    statistic, pvalue = pearsonr(consistency_performance_1, consistency_performance_2)
    # return None if we have no statistical significance
    return statistic if pvalue < p_thresh else None

def analyze_retriever_consistency_results_across_samples(sample_across, num_samples, lm_results: Dict, retriever_id_results: Dict = {}, retriever_title_results: Dict = {}, r_embeddings_lookup=None, r_embeddings=None) -> None:
    if sample_across=="subject":
        get_other_pattern = lambda curr_pattern, available_patterns: curr_pattern
        get_other_subj = sample_without_replacement
    elif sample_across=="pattern":
        get_other_pattern = sample_without_replacement
        get_other_subj = lambda curr_subj, available_subjects: curr_subj
    elif sample_across=="all":
        get_other_pattern = sample_without_replacement
        get_other_subj = sample_without_replacement
    else:
        raise ValueError(f"sample_across must be either 'subject', 'pattern' or 'all', got {sample_across}")
    
    available_patterns = list(lm_results.keys())
    available_subjects = list(lm_results[available_patterns[0]].keys())
    
    r_consistency_id_performance = []
    r_consistency_title_performance = []
    r_embeddings_similarity = []
    
    sample_iter = 0
    while sample_iter < num_samples:
        pattern = random.sample(available_patterns, 1)[0]
        subj = random.sample(available_subjects, 1)[0]
        other_subj = get_other_subj(subj, available_subjects)
        other_pattern = get_other_pattern(pattern, available_patterns)
        
        (pred, gold_obj) = lm_results[pattern][subj]
        if filter_a_an_vowel_mismatch(pattern, gold_obj):
            continue
        (pred, gold_obj) = lm_results[other_pattern][other_subj]
        if filter_a_an_vowel_mismatch(pattern, gold_obj):
            continue
                    
        psgs_ids = retriever_id_results[pattern][subj]
        psgs_ids_to_compare = retriever_id_results[other_pattern][other_subj]
        r_consistency_id_performance.append(get_r_consistency_id_performance(psgs_ids, psgs_ids_to_compare))

        psgs_titles = retriever_title_results[pattern][subj]
        psgs_titles_to_compare = retriever_title_results[other_pattern][other_subj]        
        r_consistency_title_performance.append(get_r_consistency_title_performance(psgs_titles, psgs_titles_to_compare))

        r_embeddings_ix = r_embeddings_lookup[(r_embeddings_lookup.pattern==pattern) & (r_embeddings_lookup.sub_label==subj)].iloc[0].name
        r_embeddings_ix_to_compare = r_embeddings_lookup[(r_embeddings_lookup.pattern==other_pattern) & (r_embeddings_lookup.sub_label==other_subj)].iloc[0].name
        r_embeddings_similarity.append(get_r_embeddings_similarity(r_embeddings[r_embeddings_ix], r_embeddings[r_embeddings_ix_to_compare]))
        
        sample_iter += 1
        
    wandb.run.summary['retriever_id_consistency'] = sum(r_consistency_id_performance)/len(r_consistency_id_performance)
    wandb.run.summary['retriever_id_consistency_std'] = np.std(r_consistency_id_performance)
    
    wandb.run.summary['retriever_title_consistency'] = sum(r_consistency_title_performance)/len(r_consistency_title_performance)
    wandb.run.summary['retriever_title_consistency_std'] = np.std(r_consistency_title_performance)
    
    wandb.run.summary['retriever_embedding_similarity_consistency'] = sum(r_embeddings_similarity)/len(r_embeddings_similarity)
    wandb.run.summary['retriever_embedding_similarity_consistency_std'] = np.std(r_embeddings_similarity)
    
    wandb.run.summary['corr_retriever_id_title_consistency'] = get_consistency_correlations(r_consistency_id_performance, r_consistency_title_performance)
    wandb.run.summary['corr_retriever_emb_title_consistency'] = get_consistency_correlations(r_embeddings_similarity, r_consistency_title_performance)
    wandb.run.summary['corr_retriever_emb_id_consistency'] = get_consistency_correlations(r_embeddings_similarity, r_consistency_id_performance)

    wandb.run.summary['total'] = num_samples

def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--num_samples", type=int, help="random samples to check per relation", default=1000)
    parse.add_argument("--data_file", type=str, help="results file", default="/mimer/NOBACKUP/groups/dsaynova/SKR/ERNIE_Zhang/ERNIE/code/pararel_predictions/P449.jsonl")
    parse.add_argument("--retriever_embeddings_filename", type=str, default=None, help="Filename (without extension) to retriever embeddings of queries")
    
    args = parse.parse_args()
    
    data = read_jsonl_file(args.data_file)
    lm_results = defaultdict(dict)
    r_id_results = defaultdict(dict)
    r_title_results = defaultdict(dict)
    for dp in data:
        prompt, subj, obj, prediction, psgs_ids, psgs_titles = read_atlas_results(dp, True)
        r_id_results[prompt][subj] = psgs_ids
        r_title_results[prompt][subj] = psgs_titles
        lm_results[prompt][subj] = (prediction,obj)

    r_embeddings_lookup = None
    r_embeddings = None
    r_embeddings_lookup = pd.DataFrame(read_jsonl_file(args.retriever_embeddings_filename+".jsonl"))
    r_embeddings = torch.load(args.retriever_embeddings_filename+".pt")

    sample_across="subject"
    log_wandb(args, sample_across=sample_across)
    analyze_retriever_consistency_results_across_samples(sample_across, args.num_samples, lm_results, r_id_results, r_title_results, r_embeddings_lookup, r_embeddings)
    wandb.finish()
    
    sample_across="pattern"
    log_wandb(args, sample_across=sample_across)
    analyze_retriever_consistency_results_across_samples(sample_across, args.num_samples, lm_results, r_id_results, r_title_results, r_embeddings_lookup, r_embeddings)
    wandb.finish()
    
    sample_across="all"
    log_wandb(args, sample_across=sample_across)
    analyze_retriever_consistency_results_across_samples(sample_across, args.num_samples, lm_results, r_id_results, r_title_results, r_embeddings_lookup, r_embeddings)
    wandb.finish()
    
if __name__ == '__main__':
    main()
