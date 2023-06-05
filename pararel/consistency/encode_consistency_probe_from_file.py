import argparse
import json
from collections import Counter
from collections import defaultdict
from typing import List, Dict

import numpy as np
import wandb
from scipy.stats import entropy

from pararel.consistency.lm_pipeline import build_model_by_name, run_query
from pararel.consistency.utils import read_jsonl_file, read_graph

from pararel.consistency.encode_consistency_probe import log_wandb, analyze_graph, analyze_results, evaluate_lama, group_score_lama_eval, group_score_incorrect_ans_eval

def read_ernie_results(data, r_flag = False):
    return data["prompt"], data["subject"], data["object"].lower(), data["prediction"], [0]

def read_atlas_results(data, r_flag = False):
    retrieval = [p["id"] for p in data["passages"]] if r_flag else [0]
    return data["pattern"], data["sub_label"], data["answers"][0], data["generation_by_choice"], retrieval

def read_llama_results(data, r_flag = False):
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
    args = parse.parse_args()

    if args.wandb:
        log_wandb(args)

    model_name = args.lm

    print('Language Models: {}'.format(model_name))

    patterns_graph = read_graph(args.graph)
    
    data = read_jsonl_file(args.data_file)
    lm_results = defaultdict(dict)
    r_results = defaultdict(dict)

    read_results_fn = None
    if "ernie" in args.lm:
        read_results_fn = read_ernie_results
    elif any(model in args.lm for model in ("atlas", "t5")):
        read_results_fn = read_atlas_results
    elif "llama" in args.lm:
        read_results_fn = read_llama_results
    else:
        ValueError("LM must be any of ERNIE, Atlas or LLaMA models.")

    for dp in data:
        prompt, subj, obj, prediction, psgs = read_results_fn(dp, args.retriever_statistics)
        r_results[prompt][subj] = psgs
        lm_results[prompt][subj] = (prediction,obj)

    if args.retriever_statistics:
        analyze_results(lm_results, patterns_graph, r_results)
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
