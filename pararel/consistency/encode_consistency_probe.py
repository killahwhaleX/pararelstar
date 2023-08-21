import argparse
import json
from collections import Counter
from collections import defaultdict
from typing import List, Dict

import numpy as np
import wandb
from scipy.stats import entropy, pearsonr
from torch import nn

from pararel.consistency.lm_pipeline import build_model_by_name, run_query
from pararel.consistency.utils import read_jsonl_file, read_graph


def log_wandb(args):
    pattern = args.data_file.split('/')[-1].split('.')[0].split('-')[0]
    lm = args.lm
    tags_list = [pattern, 'probe']
    if args.wandb_flag: 
        tags_list.append(args.wandb_flag)

    if args.baseline:
        lm = 'majority-baseline'

    config = dict(
        pattern=pattern,
        lm=lm
    )

    if 'consistency' in lm:
        params = lm.split('consistency_')[-1].split('/')[0]
        model_args = params.split('_')

        config['loss_strategy'] = model_args[0]
        config['n_tuples'] = model_args[1]
        config['n_graphs'] = model_args[2]
        config['rels_train'] = model_args[3]
        config['origin_lm'] = model_args[4]
        config['loss'] = model_args[5]
        config['vocab'] = model_args[6]
        config['wiki'] = model_args[7]
        config['lama_train'] = model_args[8]
        config['wiki_consistent_train_ratio'] = model_args[9]
        config['consistent_loss_ratio'] = model_args[10]
        config['additional_notes'] = model_args[11]

        checkpoint = lm.split('checkpoint-')[-1].split('/')[0]
        config['checkpoint'] = checkpoint

        model_name = lm.split('/checkpoint-')[0]
        config['model_name'] = model_name

    wandb.init(
        name=f'{pattern}_consistency_probe_{lm}',
        project="consistency-latest",
        tags=tags_list,
        config=config,
    )


def read_txt_lines(in_f: str) -> List[str]:
    with open(in_f, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def get_first_object(preds, possible_objects):
    for row in preds:
        token = row['token_str']
        if token in possible_objects:
            return token
    return ''


def parse_lm_results(lm_results: Dict, possible_objects: List[str]) -> Dict:
    output_dic = defaultdict(dict)
    c = 0
    for pattern, dic in lm_results.items():
        for data, preds in zip(dic['data'], dic['predictions']):
            subj = data['sub_label']
            obj = data['obj_label']
            first_object = get_first_object(preds, possible_objects)
            output_dic[pattern][subj] = (first_object, obj)
    return output_dic


def get_node(graph, pattern):
    for node in graph.nodes:
        if node.lm_pattern == pattern:
            return node
    return None


def filter_a_an_vowel_mismatch(pattern, gold_object):
    words = pattern.split(' ')
    indices = [ind for (ind, x) in enumerate(words) if '[Y]' in x]
    assert len(indices) == 1
    y_index = indices[0]
    if y_index == 0:
        return False

    vowels = ['a', 'e', 'i', 'o', 'u']
    if words[y_index - 1] == 'a' and any([gold_object.lower().startswith(x) for x in vowels]):
        return True
    elif words[y_index - 1] == 'an' and not any([gold_object.lower().startswith(x) for x in vowels]):
        return True
    return False

def get_r_embeddings_similarity(r_embs_1, r_embs_2):
    return nn.functional.cosine_similarity(r_embs_1, r_embs_2, dim=0).item()

def get_stratified_consistency_metrics(metric, consistency_performance):
    # metric goes over the same dimensions as consistency_performance (nbr_subjects, nbr_patterns)
    raw_metric = []
    match_metric = []
    no_match_metric = []
    for subj, vals in metric.items():
        # iterate over pairwise pattern performance
        raw_metric.extend(vals)
        match_list = [i for i, j in zip(vals,consistency_performance[subj]) if j]
        no_match_list = [i for i, j in zip(vals,consistency_performance[subj]) if not j]
        match_metric.extend(match_list)
        no_match_metric.extend(no_match_list)
    results = {"mean": sum(raw_metric)/len(raw_metric),
               "std": np.std(raw_metric),
               "match_mean": sum(match_metric)/len(match_metric),
               "match_std": np.std(match_metric),
               "no_match_mean": sum(no_match_metric)/len(no_match_metric),
               "no_match_std": np.std(no_match_metric)}
    
    return results

def get_consistency_correlations(consistency_performance_1, consistency_performance_2):
    p_thresh = 0.05
    cons_1 = []
    cons_2 = []
    for subj, cons_vals in consistency_performance_1.items():
        cons_1.extend(cons_vals)
        cons_2.extend(consistency_performance_2[subj])
        
    statistic, pvalue = pearsonr(cons_1, cons_2)
    return statistic #if pvalue < p_thresh else None

def analyze_results(lm_results: Dict, patterns_graph, choice_confidences: Dict = None, retriever_id_results: Dict = {}, retriever_title_results: Dict = {}, retriever_rank_pred_results=None, retriever_rank_gold_results=None, r_embeddings_lookup=None, r_embeddings=None) -> None:
    total = 0
    points = 0

    total_syn = 0
    total_lex = 0
    total_both = 0
    total_no = 0

    points_syn = 0
    points_lex = 0
    points_both = 0
    points_no = 0

    points_by_edge = defaultdict(list)
    edges_out = defaultdict(list)

    avg_entropy = []

    consistent_subjects = defaultdict(list)
    correct_subjects_per_pattern = defaultdict(int)
    correct_patterns_per_subject = defaultdict(int)
    consistency_performance = defaultdict(list)
    k_consistency_performance = defaultdict(list)
    r_consistency_id_performance = defaultdict(list)
    r_consistency_title_performance = defaultdict(list)
    r_embeddings_similarity = defaultdict(list)
    r_avg_rank_pred = defaultdict(list)
    r_avg_rank_gold = defaultdict(list)
    min_choice_confidence = defaultdict(list)
    
    for pattern, vals in lm_results.items():
        for subj, (pred, gold_obj) in vals.items():
            if len(retriever_id_results)>0:
                psgs_ids = retriever_id_results[pattern][subj]
                psgs_titles = retriever_title_results[pattern][subj]
                r_rank_pred = retriever_rank_pred_results[pattern][subj]
                r_rank_gold = retriever_rank_gold_results[pattern][subj]
            if r_embeddings_lookup is not None:
                r_embeddings_ix = r_embeddings_lookup[(r_embeddings_lookup.pattern==pattern) & (r_embeddings_lookup.sub_label==subj)].iloc[0].name
            graph_node = get_node(patterns_graph, pattern)
            if graph_node is None:
                continue
            if filter_a_an_vowel_mismatch(pattern, gold_obj):
                continue

            correct_patterns_per_subject[subj] += int(pred == gold_obj)
            correct_subjects_per_pattern[pattern] += int(pred == gold_obj)
            consistent_subjects[subj].append(pred)
            base_pattern_success = []
            # going over all entailed patterns
            for ent_node in patterns_graph.successors(graph_node):
                if [graph_node, ent_node] not in patterns_graph.edges:
                    continue
                entailment_type = patterns_graph.edges[graph_node, ent_node]

                ent_pattern = ent_node.lm_pattern
                if filter_a_an_vowel_mismatch(ent_pattern, gold_obj):
                    continue
                if len(retriever_id_results)>0:
                    psgs_ids_to_compare = retriever_id_results[ent_pattern][subj]
                    psgs_titles_to_compare = retriever_title_results[ent_pattern][subj]
                    r_consistency_id_performance[subj].append(len(set(psgs_ids) & set(psgs_ids_to_compare))/len(psgs_ids))
                    overlapping_titles = set(psgs_titles) & set(psgs_titles_to_compare)
                    num_overlap_titles = sum([psgs_titles.count(title)+psgs_titles_to_compare.count(title) for title in overlapping_titles])
                    r_consistency_title_performance[subj].append(num_overlap_titles/(len(psgs_titles)+len(psgs_titles_to_compare)))
                    
                    # frequency based heuristics
                    r_rank_pred_to_compare = retriever_rank_pred_results[ent_pattern][subj]
                    r_avg_rank_pred[subj].append(np.mean([r_rank_pred, r_rank_pred_to_compare]))
                    
                    r_rank_gold_to_compare = retriever_rank_gold_results[ent_pattern][subj]
                    r_avg_rank_gold[subj].append(np.mean([r_rank_gold, r_rank_gold_to_compare]))
                    
                if r_embeddings_lookup is not None:
                    r_embeddings_ix_to_compare = r_embeddings_lookup[(r_embeddings_lookup.pattern==ent_pattern) & (r_embeddings_lookup.sub_label==subj)].iloc[0].name
                    r_embeddings_similarity[subj].append(get_r_embeddings_similarity(r_embeddings[r_embeddings_ix], r_embeddings[r_embeddings_ix_to_compare]))
                
                if choice_confidences is not None:
                    min_choice_confidence[subj].append(min(choice_confidences[pattern][subj], choice_confidences[ent_pattern][subj]))
                
                success = pred == lm_results[ent_pattern][subj][0]
                k_success = pred == lm_results[ent_pattern][subj][0] and pred == gold_obj
                if success:
                    points += 1
                total += 1
                base_pattern_success.append(int(success))
                consistency_performance[subj].append(success)
                k_consistency_performance[subj].append(k_success)

                points_by_edge[graph_node.lm_pattern + '_' + ent_node.lm_pattern].append(int(success))
                edges_out[graph_node.lm_pattern].append(int(success))

                if entailment_type['edge_type'].syntactic_change and not entailment_type['edge_type'].lexical_change \
                        and not entailment_type['edge_type'].determiner_change:
                    if success:
                        points_syn += 1
                    total_syn += 1
                elif entailment_type['edge_type'].lexical_change and not entailment_type['edge_type'].syntactic_change \
                        and not entailment_type['edge_type'].determiner_change:
                    if success:
                        points_lex += 1
                    total_lex += 1
                elif entailment_type['edge_type'].lexical_change and entailment_type['edge_type'].syntactic_change \
                        and not entailment_type['edge_type'].determiner_change:
                    if success:
                        points_both += 1
                    total_both += 1
                if not entailment_type['edge_type'].syntactic_change and not entailment_type['edge_type'].lexical_change \
                        and not entailment_type['edge_type'].determiner_change:
                    if success:
                        points_no += 1
                    total_no += 1
            base_success = sum(base_pattern_success) / len(base_pattern_success)
            ent = entropy([base_success, 1.0 - base_success], base=2)
            avg_entropy.append(ent) 
    if total > 0:
        print('overall', points, total, points / total)
        wandb.run.summary['consistency'] = points / total
    else:
        wandb.run.summary['consistency'] = -1
    if len(retriever_id_results)>0:
        results = get_stratified_consistency_metrics(r_consistency_id_performance, consistency_performance)
        results_str = "retriever_id_consistency"
        for key, val in results.items():
            wandb.run.summary[results_str+"_"+key] = val
        
        results = get_stratified_consistency_metrics(r_consistency_title_performance, consistency_performance)
        results_str = "retriever_title_consistency"
        for key, val in results.items():
            wandb.run.summary[results_str+"_"+key] = val
            
        wandb.run.summary['corr_consistency_retriever_id_consistency'] = get_consistency_correlations(consistency_performance, r_consistency_id_performance)       
        wandb.run.summary['corr_consistency_retriever_title_consistency'] = get_consistency_correlations(consistency_performance, r_consistency_title_performance)
        wandb.run.summary['corr_retriever_id_title_consistency'] = get_consistency_correlations(r_consistency_id_performance, r_consistency_title_performance)
        
        results = get_stratified_consistency_metrics(r_avg_rank_pred, consistency_performance)
        results_str = "retriever_rank_pred"
        for key, val in results.items():
            wandb.run.summary[results_str+"_"+key] = val
        
        results = get_stratified_consistency_metrics(r_avg_rank_gold, consistency_performance)
        results_str = "retriever_rank_gold"
        for key, val in results.items():
            wandb.run.summary[results_str+"_"+key] = val
            
        wandb.run.summary['corr_consistency_retriever_rank_pred'] = get_consistency_correlations(r_avg_rank_pred, consistency_performance)       
        wandb.run.summary['corr_consistency_retriever_rank_gold'] = get_consistency_correlations(r_avg_rank_gold, consistency_performance)       
        wandb.run.summary['corr_retriever_rank_pred_gold'] = get_consistency_correlations(r_avg_rank_gold, r_avg_rank_pred)       
                          
    if r_embeddings_lookup is not None:
        results = get_stratified_consistency_metrics(r_embeddings_similarity, consistency_performance)
        results_str = "retriever_embedding_similarity_consistency"
        for key, val in results.items():
            wandb.run.summary[results_str+"_"+key] = val
        
        wandb.run.summary['corr_consistency_retriever_emb_consistency'] = get_consistency_correlations(consistency_performance, r_embeddings_similarity)
        wandb.run.summary['corr_retriever_emb_title_consistency'] = get_consistency_correlations(r_embeddings_similarity, r_consistency_title_performance)
        wandb.run.summary['corr_retriever_emb_id_consistency'] = get_consistency_correlations(r_embeddings_similarity, r_consistency_id_performance)

    if choice_confidences is not None:
        results = get_stratified_consistency_metrics(min_choice_confidence, consistency_performance)
        results_str = "choice_confidence"
        for key, val in results.items():
            wandb.run.summary[results_str+"_"+key] = val
            
    if total_syn > 0:
        wandb.run.summary['syntactic_consistency'] = points_syn / total_syn
        print('syntactic', points_syn, total_syn, points_syn / total_syn)
    else:
        wandb.run.summary['syntactic_consistency'] = -1
    if total_lex > 0:
        wandb.run.summary['lexical_consistency'] = points_lex / total_lex
        print('lexical', points_lex, total_lex, points_lex / total_lex)
    else:
        wandb.run.summary['lexical_consistency'] = -1
    if total_no > 0:
        wandb.run.summary['no_change_consistency'] = points_no / total_no
        print('no change', points_no, total_no, points_no / total_no)
    else:
        wandb.run.summary['no_change_consistency'] = -1
    if total_both > 0:
        print('both', points_both, total_both, points_both / total_both)
        wandb.run.summary['both_consistency'] = points_both / total_both
    else:
        wandb.run.summary['both_consistency'] = -1

    avg_out_normalized = []
    out_edges_total = 0
    for k, vals in points_by_edge.items():
        eo = sum(edges_out[k.split('_')[0]]) / len(edges_out[k.split('_')[0]])
        avg_out_normalized.append(eo * (sum(vals) / len(vals)))
        out_edges_total += eo
    wandb.run.summary['avg_consistency_by_edge_out'] = sum(avg_out_normalized) / out_edges_total

    all_consistent = 0
    for subj, preds in consistent_subjects.items():
        preds_set = set(preds)
        if len(preds_set) == 1:
            all_consistent += 1
    wandb.run.summary['consistent_subjects'] = all_consistent / len(consistent_subjects)

    successful_subjects = 0
    for subj, success in correct_patterns_per_subject.items():
        if success > 0:
            successful_subjects += 1
    wandb.run.summary['successful_subjects'] = successful_subjects / len(correct_patterns_per_subject)

    successful_patterns = 0
    for pattern, success in correct_subjects_per_pattern.items():
        if success > 0:
            successful_patterns += 1
    wandb.run.summary['successful_patterns'] = successful_patterns / len(correct_subjects_per_pattern)
    success_for_knowledgable_patterns, total_for_knowledgable_patterns = 0, 0
    success_for_unknowledgable_patterns, total_for_unknowledgable_patterns = 0, 0
    for subj, success in consistency_performance.items():
        if correct_patterns_per_subject[subj] > 0:
            success_for_knowledgable_patterns += sum(success)
            total_for_knowledgable_patterns += len(success)
        else:
            success_for_unknowledgable_patterns += sum(success)
            total_for_unknowledgable_patterns += len(success)
    if total_for_knowledgable_patterns > 0:
        wandb.run.summary[
            'knowledgable_consistency'] = success_for_knowledgable_patterns / total_for_knowledgable_patterns
    else:
        wandb.run.summary['knowledgable_consistency'] = 0
    if total_for_unknowledgable_patterns > 0:
        wandb.run.summary['unknowledgable_consistency'] = success_for_unknowledgable_patterns \
                                                          / total_for_unknowledgable_patterns
    else:
        wandb.run.summary['unknowledgable_consistency'] = 0

    k_success_for_knowledgable_patterns, k_total_for_knowledgable_patterns = 0, 0
    for subj, success in k_consistency_performance.items():
        if correct_patterns_per_subject[subj] > 0:
            k_success_for_knowledgable_patterns += sum(success)
            k_total_for_knowledgable_patterns += len(success)
    if k_total_for_knowledgable_patterns > 0:
        wandb.run.summary[
            'k_knowledgable_consistency'] = k_success_for_knowledgable_patterns / k_total_for_knowledgable_patterns
    else:
       wandb.run.summary['k_knowledgable_consistency'] = 0


    wandb.run.summary['total'] = total
    wandb.run.summary['total_syn'] = total_syn
    wandb.run.summary['total_lex'] = total_lex
    wandb.run.summary['total_both'] = total_both
    wandb.run.summary['total_no'] = total_no

    wandb.run.summary['avg_entropy'] = np.average(avg_entropy)
    wandb.run.summary['std_entropy'] = np.std(avg_entropy)


def analyze_graph(patterns_graph):
    syn_edges = 0
    lex_edges = 0
    both_edges = 0

    for node in patterns_graph:
        for ent_node in patterns_graph.successors(node):
            entailment_type = patterns_graph.edges[node, ent_node]['edge_type']
            if entailment_type.syntactic_change and not entailment_type.lexical_change \
                    and not entailment_type.determiner_change:
                syn_edges += 1
            elif entailment_type.lexical_change and not entailment_type.syntactic_change \
                    and not entailment_type.determiner_change:
                lex_edges += 1
            elif entailment_type.lexical_change and entailment_type.syntactic_change \
                    and not entailment_type.determiner_change:
                both_edges += 1

    wandb.run.summary['n_patterns'] = len(patterns_graph)
    wandb.run.summary['all_edges'] = len(patterns_graph.edges)
    wandb.run.summary['syntactic_edges'] = syn_edges
    wandb.run.summary['lexical_edges'] = lex_edges
    wandb.run.summary['both_edges'] = both_edges


def evaluate_lama(pattern: str, lm_results: Dict):
    points = 0
    data, predictions = lm_results[pattern]['data'], lm_results[pattern]['predictions']
    for datum, preds in zip(data, predictions):
        subj = datum['sub_label']
        obj = datum['obj_label']
        pred_obj = preds[0]['token_str']
        if pred_obj == obj:
            points += 1
    return points / len(data)


def group_score_lama_eval(lm_results: Dict):
    patterns = list(lm_results.keys())

    points = 0
    data = lm_results[patterns[0]]['data']
    for datum_ind, datum in enumerate(data):
        obj = datum['obj_label']
        consistent_true = True
        for pattern in patterns:
            preds = lm_results[pattern]['predictions'][datum_ind]
            if preds[0]['token_str'] != obj:
                consistent_true = False
                break

        if consistent_true:
            points += 1

    return points / len(data)


def group_score_incorrect_ans_eval(lm_results: Dict):
    patterns = list(lm_results.keys())

    points = 0
    data = lm_results[patterns[0]]['data']
    for datum_ind, datum in enumerate(data):
        obj = datum['obj_label']
        answers = defaultdict(int)
        for pattern in patterns:
            preds = lm_results[pattern]['predictions'][datum_ind]
            answers[preds[0]['token_str']] += 1

        if len(answers) == 1 and list(answers.keys())[0] != obj:
            points += 1

    return points / len(data)


def create_majority_baseline(data):
    data_reduced = []
    for row in data:
        data_reduced.append({'sub_label': row['sub_label'], 'obj_label': row['obj_label']})

    objs = [x['obj_label'] for x in data]
    most_common = Counter(objs).most_common()[0][0]
    preds_reduced = []
    for _ in data:
        vals = [{'score': 1, 'token_str': most_common}]
        preds_reduced.append(vals)
    return data_reduced, preds_reduced


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--lm", type=str, help="name of the used masked language model", default="bert-base-uncased")
    parse.add_argument("--data_file", type=str, help="", default="data/trex_lms_vocab/P449.jsonl")
    parse.add_argument("-graph", "--graph", type=str, help="graph file",
                       default="data/pattern_data/graphs/P449.graph")

    parse.add_argument("--gpu", type=int, default=-1)
    parse.add_argument("--bs", type=int, default=200)
    parse.add_argument("--wandb", action='store_true')
    parse.add_argument("--no_subj", type=bool, default=False)
    parse.add_argument("--baseline", action='store_true', default=False)
    parse.add_argument("--use_targets", action='store_true', default=False, help="use the set of possible objects"
                                                                                 "from the data as the possible"
                                                                                 "candidates")
    parse.add_argument("--wandb_flag", type=str, default = None)
    args = parse.parse_args()

    if args.wandb:
        log_wandb(args)

    # Load data
    if args.no_subj:
        data = [{"sub_label": "", "obj_label": ""}]
    else:
        data = read_jsonl_file(args.data_file)

    if "uncased" in args.lm:
        for x in data:
            x["obj_label"] = x["obj_label"].lower()
    
    model_name = args.lm

    print('Language Models: {}'.format(model_name))

    model = build_model_by_name(model_name, args)

    patterns_graph = read_graph(args.graph)

    subj_obj = {}
    for row in data:
        subj_obj[row['sub_label']] = row['obj_label']

    # Load prompts
    prompts = [x.lm_pattern for x in list(patterns_graph.nodes)]

    if args.use_targets:
        all_objects = list(set([x['obj_label'] for x in data]))
        # if 'roberta' in args.lm or 'albert' in args.lm:
        if 'roberta' in args.lm:
            all_objects = [' ' + x for x in all_objects]
        elif 'albert' in args.lm:
            all_objects = [model.tokenizer.tokenize(x)[0] for x in all_objects]
    else:
        all_objects = None

    results_dict = {}

    if args.baseline:
        for prompt_id, prompt in enumerate(prompts):
            filtered_data, predictions = create_majority_baseline(data)
            results_dict[prompt] = {"data": filtered_data, "predictions": predictions}
    else:
        for prompt_id, prompt in enumerate(prompts):
            results_dict[prompt] = []
            filtered_data, predictions = run_query(model, data, prompt, all_objects, args.bs)
            results_dict[prompt] = {"data": filtered_data, "predictions": predictions}

    # Evaluate on LAMA
    lama_acc = evaluate_lama(prompts[0], results_dict)
    wandb.run.summary['lama_acc'] = lama_acc

    # Group Eval
    group_acc = group_score_lama_eval(results_dict)
    wandb.run.summary['lama_group_acc'] = group_acc

    group_false_acc = group_score_incorrect_ans_eval(results_dict)
    wandb.run.summary['group-unacc'] = group_false_acc

    lm_results = parse_lm_results(results_dict, all_objects)

    analyze_results(lm_results, patterns_graph)
    analyze_graph(patterns_graph)

    if 'models' in model_name or 'nyu' in model_name:
        model_name = model_name.replace('/', '_')
    pattern = args.data_file.split('/')[-1].split('.')[0]
    with open('data/output/predictions_lm/trex_lms_vocab/{}_{}.json'.format(pattern, model_name), 'w') as f:
        json.dump(lm_results, f)


if __name__ == '__main__':
    main()
