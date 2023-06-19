import argparse
import json
from collections import Counter
from collections import defaultdict
from typing import List, Dict

import numpy as np
import wandb
from scipy.stats import entropy
import pandas as pd
import torch
from nltk.stem.snowball import SnowballStemmer
import os

from pararel.consistency.lm_pipeline import build_model_by_name, run_query
from pararel.consistency.utils import read_jsonl_file, read_graph

from pararel.consistency.encode_consistency_probe import log_wandb, analyze_graph, analyze_results, evaluate_lama, group_score_lama_eval, group_score_incorrect_ans_eval

stemmer = SnowballStemmer(language='english')
def get_passage_obj_freq(passages, obj):
    text = (" ").join([passage['text'].lower() for passage in passages])
    return text.count(stemmer.stem(obj.lower()))

def get_answer_freq_rank(passages, pred, gold, options):
    # returns the frequency rank of each of pred and gold
    pred_ix = options.index(pred)
    gold_ix = options.index(gold)
    
    opt_count = []
    for opt in options:
        opt_count += [get_passage_obj_freq(passages, opt)] # stemmer cannot handle e.g. 'physicist'
    return (len(options) - np.where(np.argsort(opt_count)==pred_ix)[0][0],
            len(options) - np.where(np.argsort(opt_count)==gold_ix)[0][0]) # rank 1 means most frequent

def read_ernie_results(data, r_flag = False, options=None):
    return data["prompt"], data["subject"], data["object"].lower(), data["prediction"], [0]

def read_atlas_results(data, r_flag = False, options=None):
    retrieval_id = [p["id"] for p in data["passages"]] if r_flag else [0]
    retrieval_title = [p["title"].split(":")[0] for p in data["passages"]] if r_flag else [0] #title: "Eward Burke: Early life and career", section: "Early life and career"
    r_rank_pred = None
    r_rank_gold = None
    if options is not None:
        r_rank_pred, r_rank_gold = get_answer_freq_rank(data["passages"], data["generation_by_choice"], data["answers"][0], options)
    return data["pattern"], data["sub_label"], data["answers"][0], data["generation_by_choice"], retrieval_id, retrieval_title, r_rank_pred, r_rank_gold

def read_llama_results(data, r_flag = False, options=None):
    return data["pattern"], data["sub_label"], data["answers"][0], data["generation"], [0]

def get_filtered_data(data):
    sorted_data = {i: data[i] for i in sorted(data.keys())}
    data_reduced = []
    preds_reduced = []
    for key, val in sorted_data.items():
        data_reduced.append({'sub_label': key, 'obj_label': val[1]})
        preds_reduced.append([{'score': 1, 'token_str': val[0]}])
    return data_reduced, preds_reduced
    

def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--lm", type=str, help="name of the used language model", default="ernie-zhang")
    parse.add_argument("--data_file", type=str, help="results file", default="/mimer/NOBACKUP/groups/dsaynova/SKR/ERNIE_Zhang/ERNIE/code/pararel_predictions/P449.jsonl")
    parse.add_argument("--graph", "--graph", type=str, help="graph file",
                       default="data/pattern_data/graphs/P449.graph")

    parse.add_argument("--wandb", action='store_true')
    parse.add_argument("--baseline", action='store_true', default=False)
    parse.add_argument("--wandb_flag", type=str, help="additional flag for wandb", default="")
    parse.add_argument("--retriever_statistics", action='store_true')
    parse.add_argument("--retriever_embeddings_filename", type=str, default=None, help="Filename (without extension) to retriever embeddings of queries")
    parse.add_argument("--options_folder", type=str, default=None, help="Path to folder with files listing answer options, e.g. 'P17_options.txt'")
    
    args = parse.parse_args()

    if args.wandb:
        log_wandb(args)

    model_name = args.lm

    print('Language Models: {}'.format(model_name))

    patterns_graph = read_graph(args.graph)
    
    data = read_jsonl_file(args.data_file)
    lm_results = defaultdict(dict)
    r_id_results = defaultdict(dict)
    r_title_results = defaultdict(dict)
    r_rank_pred_results = defaultdict(dict)
    r_rank_gold_results = defaultdict(dict)

    read_results_fn = None
    if "ernie" in args.lm:
        read_results_fn = read_ernie_results
    elif any(model in args.lm for model in ("atlas", "t5")):
        read_results_fn = read_atlas_results
    elif "llama" in args.lm:
        read_results_fn = read_llama_results
    else:
        ValueError("LM must be any of ERNIE, Atlas or LLaMA models.")

    # need to read the relation options if wish to get statistics related to frequence heuristic
    options = None
    if args.retriever_statistics:
        pattern = args.data_file.split('/')[-1].split('.')[0].split('-')[0]
        options_filepath = os.path.join(args.options_folder, f"{pattern}_options.txt")
        with open(options_filepath, "r") as f:
            options = [line.strip() for line in f.readlines()]

    for dp in data:
        prompt, subj, obj, prediction, psgs_ids, psgs_titles, r_rank_preds, r_rank_golds = read_results_fn(dp, args.retriever_statistics, options)
        r_id_results[prompt][subj] = psgs_ids
        r_title_results[prompt][subj] = psgs_titles
        lm_results[prompt][subj] = (prediction,obj)
        r_rank_pred_results[prompt][subj] = r_rank_preds
        r_rank_gold_results[prompt][subj] = r_rank_golds

    r_embeddings_lookup = None
    r_embeddings = None
    if args.retriever_statistics and args.retriever_embeddings_filename is not None:
        r_embeddings_lookup = pd.DataFrame(read_jsonl_file(args.retriever_embeddings_filename+".jsonl"))
        r_embeddings = torch.load(args.retriever_embeddings_filename+".pt")

    if args.retriever_statistics:
        analyze_results(lm_results, patterns_graph, r_id_results, r_title_results, r_rank_pred_results, r_rank_gold_results, r_embeddings_lookup, r_embeddings)
    else: 
        analyze_results(lm_results, patterns_graph)
    analyze_graph(patterns_graph)

    # Analyze LAMA performance
    # Load prompts
    prompts = [x.lm_pattern for x in list(patterns_graph.nodes)]
    results_dict = {}
    for prompt_id, prompt in enumerate(prompts):
        filtered_data, predictions = get_filtered_data(lm_results[prompt])
        results_dict[prompt] = {"data": filtered_data, "predictions": predictions}

    # Evaluate on LAMA
    lama_acc = evaluate_lama(prompts[0], results_dict)
    wandb.run.summary['lama_acc'] = lama_acc

    # Group Eval
    group_acc = group_score_lama_eval(results_dict)
    wandb.run.summary['lama_group_acc'] = group_acc

    group_false_acc = group_score_incorrect_ans_eval(results_dict)
    wandb.run.summary['group-unacc'] = group_false_acc


if __name__ == '__main__':
    main()
