import argparse
import glob
import pickle
import random
import os
import sys
import json
from pathlib import Path
import pandas as pd

scripts_path = Path().absolute() / ".." / ".." 
scripts_path = str(scripts_path.resolve())

#print(scripts_path)

if scripts_path not in sys.path:
    sys.path.append(scripts_path)


from pararel.consistency import utils
from pararel.consistency.lm_pipeline import parse_prompt
from pararel.consistency.encode_consistency_probe import filter_a_an_vowel_mismatch

import random
random.seed(42)

def get_pararel_prompt(sample, prompt):
   return {
            'prompt': parse_prompt(prompt, sample["sub_label"], "[MASK]"),
            'sub_label': sample["sub_label"], 
            'obj_label': sample["obj_label"], 
            'uuid': sample["uuid"],
            'rel_ix': sample["relation"] + "_" + str(sample["index"]),
            }

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

def get_random_atlas_passages(atlas_data):
    # randomly sample a retrieval of 20 passages
    random_row = atlas_data.sample()
    passages = random_row.passages.iloc[0]
    passage_pattern = random_row.pattern.iloc[0]
    
    return passages, passage_pattern

def generate_data(folder_name, relations_given, data_path, format_prompt, generate_targets, atlas_data_path, random_passages_data_paths, random_atlas_retrieval=False):

    lama_path = os.path.join(data_path, "trex_lms_vocab")
    graph_path = os.path.join(data_path, "pattern_data", "graphs")

    relations_given = sorted(relations_given.split(","))
    output_path = os.path.join(data_path, folder_name)

    assert not (atlas_data_path is not None and not random_passages_data_paths == []), "Can only load fixed passages from one source"
    if not random_passages_data_paths == []:
        print(f"Loading passages for random retrieval from {random_passages_data_paths}...")
        num_random_passages = 20
        all_passages = []
        for data_path in random_passages_data_paths:
            all_passages += utils.read_jsonl_file(data_path)

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
        if atlas_data_path is not None:
            atlas_prediction_file_paths = glob.glob(os.path.join(atlas_data_path, relation+"-*", relation+"-*.jsonl"))
            assert len(atlas_prediction_file_paths) == 1,f"Should only have one atlas preds file, got {atlas_prediction_file_paths}"
            predictions_data = pd.DataFrame(utils.read_jsonl_file(atlas_prediction_file_paths[0]))

        with open(pattern_path, "rb") as f:
            graph = pickle.load(f)
            f_true = open(output_file, "w")
            for i, d in enumerate(data):
                if atlas_data_path is not None:
                    if not random_atlas_retrieval:
                        passages, passages_pattern = get_atlas_passages(d, predictions_data, graph)
                    else:
                        passages, passages_pattern = get_random_atlas_passages(predictions_data)
                elif not random_passages_data_paths == []:
                    passages = random.sample(all_passages, num_random_passages)
                    passages_pattern = ""
                for node in graph.nodes():
                    pattern = node.lm_pattern
                    d['index'] = i
                    d['relation'] = relation
                    dict_results = POSSIBLE_FORMATS[format_prompt](d, pattern)
                    if atlas_data_path is not None or not random_passages_data_paths == []:
                        dict_results["passages_pattern"] = passages_pattern
                        dict_results["passages"] = passages
                    f_true.write(json.dumps(dict_results))
                    f_true.write("\n")
            f_true.close()

        if generate_targets:
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
    
    args = parser.parse_args()

    if args.format not in POSSIBLE_FORMATS:
        raise ValueError(f"This function does not yet have support for any other formats than {POSSIBLE_FORMATS}.")

    generate_data(args.folder_name, args.relations_given, args.data_path, args.format, args.generate_targets, args.atlas_data_path, args.random_passages_data_paths, args.random_atlas_retrieval)


if __name__ == "__main__":
    main()
