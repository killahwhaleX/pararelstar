import argparse
import glob
import pickle
import random
import os
import json

from pararel.consistency import utils
from pararel.consistency.lm_pipeline import parse_prompt


def get_pararel_prompt(sample, prompt):
   return {'prompt': parse_prompt(prompt, sample["sub_label"], "[MASK]"),
                                'sub_label': sample["sub_label"], 'obj_label': sample["obj_label"]}

def get_atlas_prompt(sample, prompt):
    return {'question': parse_prompt(prompt, sample["sub_label"], "<extra_id_0>"),
                     'sub_label': sample["sub_label"], 'answers': [sample["obj_label"]],
                     'pattern': prompt}

def get_ernie_zhang_prompt(sample, prompt):
    return {'question': parse_prompt(prompt, sample["sub_label"], "X"),
                     'sub_label': sample["sub_label"], 'answers': [sample["obj_label"]],
                     'pattern': prompt}

POSSIBLE_FORMATS = {"pararel": get_pararel_prompt, "atlas": get_atlas_prompt, "ernie_zhang": get_ernie_zhang_prompt}


def generate_data(relations_given, data_path, format_prompt, generate_targets):

    lama_path = os.path.join(data_path, "trex_lms_vocab")
    graph_path = os.path.join(data_path, "pattern_data", "graphs")

    relations_given = sorted(relations_given.split(","))
    output_path = os.path.join(data_path,"all_n1_" + format_prompt)

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

        with open(pattern_path, "rb") as f:
            graph = pickle.load(f)
            f_true = open(output_file, "w")
            for i, d in enumerate(data):
                for node in graph.nodes():
                    pattern = node.lm_pattern
                    dict_results = POSSIBLE_FORMATS[format_prompt](d, pattern)
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
    parser.add_argument("--relations_given", "-r", type=str, default="P937,P1412,P127,P103,P276,P159,P140,P136,P495,P17,P361,P36,P740,P264,P407,P138,P30,P131,P176,P449,P279,P19,P101,P364,P106,P1376,P178,P413,P27,P20", help="what relations")
    parser.add_argument("--data_path", "-lama", type=str,
                        default="/mimer/NOBACKUP/groups/dsaynova/SKR/ParaRel/pararel/data/", help="pararel data path")
    parser.add_argument("--format", type=str, help="format to match for pararel queries", default="pararel")
    parser.add_argument("--generate_targets", action='store_true', default=True, help="save the set of possible objects"
                                                                                 "from the data as the possible"
                                                                                 "candidates")
    args = parser.parse_args()

    if args.format not in POSSIBLE_FORMATS:
        raise ValueError(f"This function does not yet have support for any other formats than {POSSIBLE_FORMATS}.")

    generate_data(args.relations_given, args.data_path, args.format, args.generate_targets)


if __name__ == "__main__":
    main()
