import argparse
import glob
import pickle
import random
import os
import json
import pandas as pd

from pararel.consistency import utils
from pararel.consistency.lm_pipeline import parse_prompt
from pararel.consistency.encode_consistency_probe import filter_a_an_vowel_mismatch

import random
random.seed(42)

def get_pararel_prompt(sample, prompt):
   return {'prompt': parse_prompt(prompt, sample["sub_label"], "[MASK]"),
                                'sub_label': sample["sub_label"], 'obj_label': sample["obj_label"]}

def get_atlas_prompt(sample, prompt):
    # remove space before mask token for Atlas
    return {'question': parse_prompt(prompt, sample["sub_label"], "<extra_id_0>").replace(" <extra_id_0>", "<extra_id_0>"),
                     'sub_label': sample["sub_label"], 'answers': [sample["obj_label"]],
                     'pattern': prompt}

def get_ernie_zhang_prompt(sample, prompt):
    return {'question': parse_prompt(prompt, sample["sub_label"], "X"),
                     'sub_label': sample["sub_label"], 'answers': [sample["obj_label"]],
                     'pattern': prompt}

POSSIBLE_FORMATS = {"pararel": get_pararel_prompt, "atlas": get_atlas_prompt, "ernie_zhang": get_ernie_zhang_prompt}

def get_atlas_passages(fact_data, atlas_data, pattern_graph):
    passage_pattern = list(pattern_graph)[0].lm_pattern
    # might have a/an mismatches in the pattern and gold object pair
    if filter_a_an_vowel_mismatch(passage_pattern, fact_data["obj_label"]):
        passage_pattern = list(pattern_graph)[1].lm_pattern
        assert not filter_a_an_vowel_mismatch(passage_pattern, fact_data["obj_label"])
    sub_label = fact_data["sub_label"]
    passages = atlas_data[(atlas_data.sub_label==sub_label) & (atlas_data.pattern==passage_pattern)].passages
    
    assert len(passages)==1, f"Only one passage retrieval should match a given pattern-sub_label pair, got {passages}"
    return passages.iloc[0], passage_pattern

def get_atlas_instance_passages(fact_data, atlas_data, pattern):
    passages = atlas_data[(atlas_data.sub_label==fact_data["sub_label"]) & (atlas_data.pattern==pattern)].passages
    assert len(passages)==1, f"Only one passage retrieval should match a given pattern-sub_label pair, got {passages}"
    return passages.iloc[0]

def get_random_atlas_passages(atlas_data):
    # randomly sample a retrieval of 20 passages
    random_row = atlas_data.sample()
    passages = random_row.passages.iloc[0]
    passage_pattern = random_row.pattern.iloc[0]
    
    return passages, passage_pattern

def get_trex_passage(fact_data, trex_data):
    evidence = trex_data[trex_data.uuid==fact_data["uuid"]].iloc[0].evidences[0]
    return {"text": evidence["masked_sentence"].replace("[MASK]", evidence["obj_surface"])}

def generate_data(args):

    lama_path = os.path.join(args.data_path, "trex_lms_vocab")
    graph_path = os.path.join(args.data_path, "pattern_data", "graphs")

    relations_given = sorted(args.relations_given.split(","))
    output_path = os.path.join(args.data_path, args.folder_name)

    assert not (args.atlas_data_path is not None and not args.random_passages_data_paths == []), "Can only load fixed passages from one source"
    if not args.random_passages_data_paths == []:
        print(f"Loading passages for random retrieval from {args.random_passages_data_paths}...")
        num_random_passages = 20
        all_passages = []
        for path in args.random_passages_data_paths:
            all_passages += utils.read_jsonl_file(path)

    if not os.path.exists(output_path):
        print("Saving data to: ", output_path)
        os.mkdir(output_path)
        
    for relation in relations_given:
        output_file = os.path.join(output_path, relation + ".jsonl")
        if os.path.exists(output_file):
            print(f"Data already exists for {output_file}. Skipping.")
            continue

        pattern_path = os.path.join(graph_path, relation+".graph")
        data = utils.read_jsonl_file(os.path.join(lama_path, relation + ".jsonl"))
        if args.atlas_data_path is not None:
            atlas_prediction_file_paths = glob.glob(os.path.join(args.atlas_data_path, relation+"-*", relation+"-*.jsonl"))
            assert len(atlas_prediction_file_paths) == 1,f"Should only have one atlas preds file, got {atlas_prediction_file_paths}"
            predictions_data = pd.DataFrame(utils.read_jsonl_file(atlas_prediction_file_paths[0]))
            
        if args.trex_data_path is not None:
            trex_file_path = glob.glob(os.path.join(args.trex_data_path, relation+".jsonl"))[0]
            trex_data = pd.DataFrame(utils.read_jsonl_file(trex_file_path))

        passages = None
        with open(pattern_path, "rb") as f:
            graph = pickle.load(f)
            f_true = open(output_file, "w")
            for i, d in enumerate(data):
                if args.atlas_data_path is not None:
                    if not args.random_atlas_retrieval:
                        passages, passages_pattern = get_atlas_passages(d, predictions_data, graph)
                    else:
                        passages, passages_pattern = get_random_atlas_passages(predictions_data)
                if args.trex_data_path is not None:
                    passages = [get_trex_passage(d, trex_data)]
                    passages_pattern = ""
                elif not args.random_passages_data_paths == []:
                    passages = random.sample(all_passages, num_random_passages)
                    passages_pattern = ""
                for node in graph.nodes():
                    pattern = node.lm_pattern
                    dict_results = POSSIBLE_FORMATS[args.format](d, pattern)
                    if args.add_instance_context:
                        if args.atlas_data_path is not None:
                            passages = get_atlas_instance_passages(d, predictions_data, pattern)
                            passages_pattern = None
                        else:
                            raise ValueError("The `--add_instance_context` is only applicable for when Atlas passages are used.")
                    if passages is not None:
                        dict_results["passages_pattern"] = passages_pattern
                        if args.add_fixed_context or args.add_instance_context:
                            dict_results["question"] = dict_results["question"] + f" context: {passages[0]['text']}"
                        else: # otherwise add the passages as a separate entry, instead of as context 
                            dict_results["passages"] = passages[:args.number_of_fixed_passages] if args.number_of_fixed_passages is not None else passages
                    f_true.write(json.dumps(dict_results))
                    f_true.write("\n")
            f_true.close()

        if args.generate_targets:
            all_objects = list(set([x['obj_label'] for x in data]))
            with open(os.path.join(output_path, "{}_options.txt".format(relation)), 'w') as f:
                for val in all_objects:
                    f.write(val+"\n")


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--folder_name", type=str, help="name of the folder to save the data to")
    parser.add_argument("--relations_given", "-r", type=str, default="P937,P1412,P127,P103,P276,P159,P140,P136,P495,P17,P361,P36,P740,P264,P407,P138,P30,P131,P176,P449,P279,P19,P101,P364,P106,P1376,P178,P413,P27,P20", help="what relations")
    parser.add_argument("--data_path", "-lama", type=str,
                        default="/mimer/NOBACKUP/groups/dsaynova/SKR/ParaRel/pararel/data/", help="pararel data path")
    parser.add_argument("--format", type=str, help="format to match for pararel queries", default="pararel")
    parser.add_argument("--generate_targets", action='store_true', default=True, help="save the set of possible objects"
                                                                                 "from the data as the possible"
                                                                                 "candidates")
    parser.add_argument("--atlas_data_path", default=None, type=str, help="path to Atlas predictions data from which to load fixed Atlas passages")
    parser.add_argument("--random_atlas_retrieval", action='store_true', default=False, help="whether to do random sampling from Atlas retrievals, or not (use the correct)")
    parser.add_argument("--random_passages_data_paths", nargs="+", default=[],
            help="list of space-separated paths to retrieval passages from which to load fixed random Atlas passages")
    
    parser.add_argument("--trex_data_path", default=None, type=str, help="path to T-REx from which to load gold support passages")
    parser.add_argument("--number_of_fixed_passages", default=None, type=int, help="Number of fixed passages to use for the data creation")
    parser.add_argument("--add_fixed_context", action='store_true', default=False, help="Whether to skip adding retrieved passages and add the first retrieved passage directly to the query, fixed across paraphrases")
    parser.add_argument("--add_instance_context", action='store_true', default=False, help="Whether to skip adding retrieved passages and add the first retrieved passage directly to the query, where each passage corresponds to the instance at hand")
    
    args = parser.parse_args()

    if args.format not in POSSIBLE_FORMATS:
        raise ValueError(f"This function does not yet have support for any other formats than {POSSIBLE_FORMATS}.")

    if args.atlas_data_path is not None and args.trex_data_path is not None:
        raise ValueError(r"It is not possible to use both Atlas passages and T-REx passages at the same time.")

    generate_data(args)


if __name__ == "__main__":
    main()
