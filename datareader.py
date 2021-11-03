from typing import AnyStr, List, Tuple, Callable
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from transformers import PreTrainedTokenizer
from ast import literal_eval
import re
from collections import defaultdict
from unidecode import unidecode
import json
import ipdb


NLI_LABELS = {
        'same': 1,
        'exaggerates': 2,
        'downplays': 0
}

def from_np_array(array_string):
    array_string = ','.join(re.sub('\[\s+', '[', array_string).split())
    return np.array(literal_eval(array_string)).astype(np.float32)


def read_jsonl_dataset(filename: AnyStr):
    with open(filename) as f:
        data = [json.loads(l) for l in f]
    return data


def text_to_batch_transformer(text: List, tokenizer: PreTrainedTokenizer, text_pair: List = None) -> Tuple[List, List]:
    """Turn a piece of text into a batch for transformer model

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :return: A list of IDs and a mask
    """
    if not isinstance(text, List):
        text = [text]
    if text_pair and not isinstance(text_pair, List):
        text = [text_pair]
    max_length = min(tokenizer.model_max_length, 4096)
    if text_pair:
        input_ids = [tokenizer.encode(t, text_pair=tp, add_special_tokens=True, max_length=max_length, truncation=True, verbose=False)
                     for t,tp in zip(text, text_pair)]
    else:
        input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=max_length, truncation=True, verbose=False) for t in text]
    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def text_to_sequence_batch_transformer(text: List, tokenizer: PreTrainedTokenizer, text_pair: List = None) -> Tuple[List, List]:
    """Turn a list of text into a sequence of sentences separated by SEP token

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :return: A list of IDs and a mask
    """
    max_length = tokenizer.model_max_length
    input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=max_length, truncation=True, verbose=False) for t in text]
    input_ids = [[id_ for i, sent in enumerate(input_ids) for j, id_ in enumerate(sent) if (i == 0 or j != 0)]]
    if text_pair is not None:
        pair_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=max_length, truncation=True, verbose=False) for t in text_pair]
        pair_ids = [id_ for i, sent in enumerate(pair_ids) for j, id_ in enumerate(sent) if (i == 0 or j != 0)]
        input_ids[0] +=  pair_ids
    input_ids[0] = input_ids[0][:max_length]

    input_ids[0][-1] = tokenizer.sep_token_id

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def collate_batch_transformer(pad_token_id: int, input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [i[0][0] for i in input_data]
    masks = [i[1][0] for i in input_data]
    labels = [i[2] for i in input_data]
    if isinstance(labels[0], List):
        if not isinstance(labels[0][0], float):
            labels = [l for sent in labels for l in sent]


    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [pad_token_id] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    outputs = (torch.tensor(input_ids), torch.tensor(masks), torch.tensor(labels))

    if len(input_data[0]) > 3:
        logits_masks = [i[3] for i in input_data]
        logits_masks = [(m + [0] * (max_length - len(m))) for m in logits_masks]
        assert (all(len(m) == max_length for m in logits_masks))
        outputs += (torch.tensor(logits_masks),)

    return outputs


def collate_batch_transformer_with_weight(pad_token_id: int, input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return collate_batch_transformer(pad_token_id, input_data) + (torch.tensor([i[3] for i in input_data]),)


class GoldSuttonDataset(Dataset):

    # Different ways to use the dataset
    CLS_PRESS = 'cls_press'
    CLS_ABSTRACT = 'cls_abstract'
    NLI = 'nli'

    def __init__(self, jsonl_file: AnyStr, tokenizer, abstract_tokenizer, tokenizer_fn: Callable = text_to_batch_transformer, mode: AnyStr = 'nli'):
        self.dataset = pd.read_json(jsonl_file, lines=True)
        self.tokenizer = tokenizer
        self.abstract_tokenizer = abstract_tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.mode = mode
        self.sentence_list_cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset.iloc[idx]
        if self.mode == GoldSuttonDataset.CLS_PRESS:
            text = unidecode(row['press_release_conclusion'])
            label = row['press_release_strength']
            input_ids, mask = self.tokenizer_fn([text], self.tokenizer)
        elif self.mode == GoldSuttonDataset.CLS_ABSTRACT:
            text = unidecode(row['abstract_conclusion'])
            label = row['abstract_strength']
            input_ids, mask = self.tokenizer_fn([text], self.abstract_tokenizer)
        elif self.mode == GoldSuttonDataset.NLI:
            text_abstract = unidecode(row['abstract_conclusion'])
            text_press_release = unidecode(row['press_release_conclusion'])
            label = NLI_LABELS[row['exaggeration_label']]
            input_ids, mask = self.tokenizer_fn([text_press_release], self.tokenizer, text_pair=[text_abstract])
        else:
            raise NotImplementedError

        return input_ids, mask, label

    def getLabels(self, indices=None):
        if indices is not None:
            data = self.dataset.iloc[indices]
        else:
            data = self.dataset

        if self.mode == GoldSuttonDataset.CLS_PRESS:
            return np.array([l for l in data['press_release_causation_primary_label']])
        elif self.mode == GoldSuttonDataset.CLS_ABSTRACT:
            return np.array([l for l in data['article_abstract_causation_primary_label']])
        elif self.mode == GoldSuttonDataset.NLI:
            return np.array([NLI_LABELS[l] for l in data['exaggeration_label']])
        else:
            raise NotImplementedError


class ClassificationDataset(Dataset):

    def __init__(self, file: AnyStr, tokenizer, tokenizer_fn: Callable = text_to_batch_transformer, soft_labels: bool = False, nli: bool = False):

        if soft_labels:
            self.dataset = pd.read_csv(file, converters={'label': from_np_array})
        elif file.endswith('.jsonl'):
            self.dataset = pd.read_json(file, lines=True)
        else:
            self.dataset = pd.read_csv(file)
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.soft_labels = soft_labels
        self.nli = nli

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset.values[idx]
        # Calls the text_to_batch function (uncomment to use all fields)
        if self.nli:
            input_ids, masks = self.tokenizer_fn([row[0]], self.tokenizer, text_pair=[row[2]])
            if isinstance(row[1], str):
                label = NLI_LABELS[row[1]]
            else:
                label = row[1]
        else:
            input_ids, masks = self.tokenizer_fn([row[0]], self.tokenizer)
            label = row[1]

        return input_ids, masks, label

    def getLabels(self, indices=None):
        if self.soft_labels:
            raise NotImplementedError

        if self.nli:
            is_str = isinstance(self.dataset.iloc[0]['label'], str)
            if indices is None:
                return np.array([NLI_LABELS[l] if is_str else l     for l in self.dataset['label']])
            else:
                return np.array([NLI_LABELS[l] if is_str else l  for l in self.dataset.iloc[indices]['label']])

        if indices is None:
            return np.array(self.dataset['label'])
        else:
            return np.array(self.dataset.iloc[indices]['label'])