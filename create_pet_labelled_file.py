import sys
import pandas as pd
import numpy as np


if __name__ == '__main__':
    unlabelled_loc = sys.argv[1]
    logits_loc = sys.argv[2]
    output_file = sys.argv[3]

    if unlabelled_loc.endswith('.csv'):
        data = pd.read_csv(unlabelled_loc, header=None)
    else:
        data = pd.read_json(unlabelled_loc, lines=True)

    if len(data.columns) == 3:
        data.columns = ['text', 'label', 'text_pair']
    else:
        data.columns = ['text', 'label']
    soft_labels = np.loadtxt(logits_loc, skiprows=1)
    data['label'] = list(soft_labels)

    data.to_csv(output_file, index=None)
