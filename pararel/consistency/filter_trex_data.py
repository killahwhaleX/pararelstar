import argparse
import os
import json
import pandas as pd

from pararel.consistency import utils

def filter_data(data):  
    data = pd.DataFrame(data)
    duplicated_sub_labels = []
    for sub_label in data.sub_label.unique():
        matching = data[data.sub_label==sub_label]
        if len(matching) > 1:
            duplicated_sub_labels.append(sub_label)
    
    duplicated_mask = data.sub_label.isin(duplicated_sub_labels)
    return data[~(duplicated_mask)]

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--relations", "-r", type=str, default="P937,P1412,P127,P103,P276,P159,P140,P136,P495,P17,P361,P36,P740,P264,P407,P138,P30,P131,P176,P449,P279,P19,P101,P364,P106,P1376,P178,P413,P27,P20", help="the relations to use")
    parser.add_argument("--data_path", type=str,
                        default="/cephyr/users/lovhag/Alvis/projects/pararel/data/trex_lms_vocab", help="pararel T-REx data path")
    parser.add_argument("--save_data_path", "-lama", type=str,
                        default="/cephyr/users/lovhag/Alvis/projects/pararel/data/trex_lms_vocab_deduplicated", help="data path to save to")

    args = parser.parse_args()
    
    os.makedirs(args.save_data_path, exist_ok=True)

    relations = sorted(args.relations.split(","))
    for relation in relations:
        print(f"Processing relation {relation}")
        data = utils.read_jsonl_file(os.path.join(args.data_path, relation + ".jsonl"))
        filtered_data = filter_data(data)
        with open(os.path.join(args.save_data_path, relation + ".jsonl"), "w") as f:
            for _, row in filtered_data.iterrows():
                f.write(json.dumps(row.to_dict()))
                f.write("\n")
        
        num_removed = len(data)-len(filtered_data)
        print(f"New data size is {len(filtered_data)} samples. Removed {num_removed} ({(num_removed/len(data))*100:.1f}%) duplicated entries.")
        print("-------------------")
        print()
        
if __name__ == "__main__":
    main()