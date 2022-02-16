import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import json


if __name__ == '__main__':
    unlabelled_loc = sys.argv[1]
    id_file_loc = sys.argv[2]
    logits_loc = sys.argv[3]
    output_file = sys.argv[4]
    task_name = sys.argv[5]

    assert task_name == 'cls' or task_name == 'nli', "Invalid task name used"

    if unlabelled_loc.endswith('.csv'):
        data = pd.read_csv(unlabelled_loc, header=None)
    else:
        data = pd.read_json(unlabelled_loc, lines=True)

    if len(data.columns) == 3:
        data.columns = ['text', 'label', 'text_pair']
    else:
        data.columns = ['text', 'label']

    with open(id_file_loc) as f:
        ids = [l.strip() for l in f]
    soft_labels = np.loadtxt(logits_loc, skiprows=1)
    data['label'] = list(soft_labels)

    press_data = defaultdict(list)
    press_labels = defaultdict(list)
    abstract_data = defaultdict(list)
    abstract_labels = defaultdict(list)
    for id_split,v in zip(ids,data.values):
        id_,split = id_split.split('_')
        if split == 'press':
            press_data[id_].append(v[0])
            press_labels[id_].append(v[1])
        else:
            abstract_data[id_].append(v[0])
            abstract_labels[id_].append(v[1])

    if task_name == 'nli':
        # Get pairs from the soft labels
        out_data = []
        with open(output_file, 'wt') as f:
            for id_ in press_data:
                curr = {}
                top_press = press_data[id_][np.argmax(np.array(press_labels[id_])[:,1])]
                top_abstract = abstract_data[id_][np.argmax(np.array(abstract_labels[id_])[:,1])]

                f.write(json.dumps({'text': top_press, 'label': -1, 'text_pair': top_abstract}) + '\n')
    else:
        out_data = []
        for id_ in press_data:
            top_press = press_data[id_][np.argmax(np.array(press_labels[id_])[:,1])]
            top_abstract = abstract_data[id_][np.argmax(np.array(abstract_labels[id_])[:,1])]
            out_data.append([top_press, '-1', 'press'])
            out_data.append([top_abstract, '-1', 'abstract'])
        df = pd.DataFrame(out_data)
        df.to_csv(output_file, index=None, header=None) 
    
