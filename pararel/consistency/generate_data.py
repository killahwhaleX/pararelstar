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


def generate_data(num_relations, relations_given, LAMA_path, format_prompt, generate_targets):

    graph_path = "/mimer/NOBACKUP/groups/dsaynova/SKR/ParaRel/pararel/data/pattern_data/graphs/"
    relations_path = glob.glob(graph_path + "*.graph")
    output_path = "/mimer/NOBACKUP/groups/dsaynova/SKR/ParaRel/pararel/data/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    relation_path_keep = []
    if relations_given != "":
        relations_given = sorted(relations_given.split(","))
        for relation_path in relations_path:
            relation = relation_path.split("/")[-1].split(".")[0]
            if relation in relations_given:
                relation_path_keep.append(relation_path)
    if len(relation_path_keep) < num_relations:
        for relation_path in relations_path:
            if relation_path not in relation_path_keep:
                relation = relation_path.split("/")[-1].split(".")[0]
                relation_path_keep.append(relation_path)
                if len(relation_path_keep) == num_relations:
                    break
    output_path = output_path + "all_n1_" + format_prompt + "/"

    if not os.path.exists(output_path):
        print("Saving data to: ", output_path)
        os.mkdir(output_path)

        output_path_true = output_path + "train_"

        for relation_path in relation_path_keep:
    
            
            with open(relation_path, "rb") as f:
                graph = pickle.load(f)
            relation = relation_path.split("/")[-1].split(".")[0]

            f_true = open(output_path_true + relation + ".jsonl", "w")

            data = utils.read_jsonl_file(LAMA_path + relation + ".jsonl")

            if generate_targets:
                all_objects = list(set([x['obj_label'] for x in data]))
                with open(output_path+"{}_{}_options.txt".format(format_prompt, relation), 'w') as f:
                    for val in all_objects:
                        f.write(val+"\n")

            for i, d in enumerate(data):
                for node in graph.nodes():
                    pattern = node.lm_pattern

                    dict_results = POSSIBLE_FORMATS[format_prompt](d, pattern)

                    f_true.write(json.dumps(dict_results))
                    f_true.write("\n")

            f_true.close()

    else:
        print("Data already exists")


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--num_relations", "-nr", type=int, default=31, help="number of relations")
    parser.add_argument("-relations_given", "-r", type=str, default="P937,P1412,P127,P103,P276,P159,P140,P136,P495,P17,P361,P36,P740,P264,P407,P138,P30,P131,P176,P449,P279,P19,P101,P364,P106,P1376,P178,P37,P413,P27,P20", help="which relations")
    parser.add_argument("--LAMA_path", "-lama", type=str,
                        default="/mimer/NOBACKUP/groups/dsaynova/SKR/ParaRel/pararel/data/trex_lms_vocab/", help="number of tuples")
    parser.add_argument("--format", type=str, help="format to match for pararel queries", default="pararel")
    parser.add_argument("--generate_targets", action='store_true', default=True, help="save the set of possible objects"
                                                                                 "from the data as the possible"
                                                                                 "candidates")
    args = parser.parse_args()

    if args.format not in POSSIBLE_FORMATS:
        raise ValueError(f"This function does not yet have support for any other formats than {POSSIBLE_FORMATS}.")

    generate_data(args.num_relations, args.relations_given, args.LAMA_path, args.format, args.generate_targets)


if __name__ == "__main__":
    main()
