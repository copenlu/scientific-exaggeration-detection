import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc
from functools import partial
from typing import Tuple, List, Callable, AnyStr
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from datareader import collate_batch_transformer
import ipdb


def distillation_loss(temperature, logits, target_logits):
    """Compute the distillation loss (KL divergence between predictions and targets) as described in the PET paper"""
    p = F.log_softmax(logits / temperature, dim=1)
    q = F.softmax(target_logits / temperature, dim=1)
    return F.kl_div(p, q, reduction='sum') * (temperature ** 2) / logits.shape[0]


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return np.sum(preds == labels).astype(np.float32) / float(labels.shape[0])


def acc_f1(preds: List, labels: List, averaging: AnyStr = 'binary') -> Tuple[float, float, float, float]:
    acc = accuracy(preds, labels)
    P, R, F1, _ = precision_recall_fscore_support(labels, preds, average=averaging)
    return acc,P,R,F1


def average_precision(labels: np.ndarray, order: np.ndarray) -> float:
    """
    Calculates the average precision of a ranked list
    :param labels: True labels of the items
    :param order: The ranking order
    :return: Average precision
    """
    j = 0
    ap = 0
    for i, v in enumerate(labels[order]):
        if v == 1:
            j += 1
            ap += j / (i + 1)
    return ap / j

def plot_label_distribution(labels: np.ndarray, logits: np.ndarray) -> matplotlib.figure.Figure:
    """ Plots the distribution of labels in the prediction

    :param labels: Gold labels
    :param logits: Logits from the model
    :return: None
    """
    predictions = np.argmax(logits, axis=-1)
    labs, counts = zip(*list(sorted(Counter(predictions).items(), key=lambda x: x[0])))

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar(labs, counts, width=0.2)
    ax.set_xticks(labs, [str(l) for l in labs])
    ax.set_ylabel('Count')
    ax.set_xlabel("Label")
    ax.set_title("Prediction distribution")
    return fig


class ClassificationEvaluator:
    """Wrapper to evaluate a model for classification tasks

    """

    def __init__(
            self,
            dataset: Dataset,
            device: torch.device,
            num_labels: int = 2,
            averaging: AnyStr = 'binary',
            pad_token_id: int = None,
            multi_gpu: bool = False,
            task_idx: int = 0,
            multi_task: bool = False,
            batch_size: int = 32,
            temperature: float = 1.0
    ):
        self.dataset = dataset
        if isinstance(dataset, Subset):
            self.all_labels = list(dataset.dataset.getLabels(dataset.indices))
        else:
            self.all_labels = dataset.getLabels()
        collator = collate_batch_transformer

        if pad_token_id is None:
            collate_fn = partial(collator, dataset.tokenizer.pad_token_id)
        else:
            collate_fn = partial(collator, pad_token_id)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn
        )
        self.device = device
        self.averaging = averaging
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.multi_gpu = multi_gpu
        self.multi_task = multi_task

        self.task_idx = task_idx
        self.distill_loss = partial(distillation_loss, temperature)

    def predict(
            self,
            model: torch.nn.Module
    ) -> Tuple:
        model.eval()
        with torch.no_grad():
            labels_all = []
            logits_all = []
            losses_all = []
            preds_all = []
            for batch in tqdm(self.dataloader, desc="Evaluation"):
                if isinstance(batch, dict):
                    batch = [batch['input_ids'], batch['labels']]
                batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids = batch[0]
                masks = batch[1]
                labels = batch[2]
                input_dict = {'input_ids': input_ids, 'attention_mask': masks}
                if self.multi_task:
                    input_dict['task_num'] = self.task_idx
                    input_dict['logits_mask'] = batch[3] if len(batch) > 3 else None
                if self.multi_gpu:
                    outputs = model(**input_dict)
                    outputs = (torch.nn.CrossEntropyLoss()(outputs[0].reshape(-1, self.num_labels), labels.reshape(-1)), outputs.logits)
                else:
                    #input_dict['labels'] = labels
                    outputs = model(**input_dict)
                    labels = labels.view(outputs[0].shape[0], -1)
                    if labels.shape[1] > 1:
                        loss = self.distill_loss(outputs[0], labels.reshape(-1, self.num_labels))
                    else:
                        loss = torch.nn.CrossEntropyLoss()(outputs[0].reshape(-1, self.num_labels), labels.reshape(-1))
                    outputs = (loss,outputs[0])


                labels_all.extend(list(labels.detach().cpu().numpy()))
                #logits_all.extend(list(outputs.logits.detach().cpu().numpy()))
                logits_all.extend(list(outputs[1].detach().cpu().numpy()))

                #losses_all.append(outputs.loss.item())
                losses_all.append(outputs[0].item())
                #preds = np.argmax(outputs.logits.detach().cpu().numpy().reshape(-1, self.num_labels), axis=-1)
                preds = np.argmax(outputs[1].detach().cpu().numpy().reshape(-1, self.num_labels), axis=-1)
                preds_all.extend([p for p in preds])

        assert len(labels_all) == len(self.all_labels)
        assert len(logits_all) == len(self.all_labels)
        assert len(preds_all) == len(self.all_labels)
        return labels_all, logits_all, losses_all, preds_all

    def roc_auc(self, model: torch.nn.Module):
        labels_all, logits_all, losses_all = self.predict(model)
        logits = np.asarray(logits_all).reshape(-1, self.num_labels)
        labels = np.asarray(labels_all).reshape(-1)
        fpr, tpr, _ = roc_curve(labels, logits[:, 1])
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def evaluate(
            self,
            model: torch.nn.Module,
            plot_callbacks: List[Callable] = [],
            return_labels_logits: bool = False
    ) -> Tuple:
        """Collect evaluation metrics on this dataset

        :param model: The pytorch model to evaluate
        :param plot_callbacks: Optional function callbacks for plotting various things
        :return: (Loss, Accuracy, Precision, Recall, F1)
        """
        labels_all, logits_all, losses_all, preds_all = self.predict(model)
        loss = sum(losses_all) / len(losses_all)
        print(preds_all)

        if len(np.array(labels_all).shape) > 1 and np.array(labels_all).shape[1] > 1:
            acc,P,R,F1 = ('NA', 'NA', 'NA', 'NA')
        else:
            acc,P,R,F1 = acc_f1(np.asarray(preds_all), np.asarray(labels_all).reshape(-1), averaging=self.averaging)
        ret_vals = (loss, acc, P, R, F1)

        # Plotting
        plots = []
        for f in plot_callbacks:
            plots.append(f(labels_all, logits_all))

        if len(plots) > 0:
            ret_vals = (loss, acc, P, R, F1), plots

        # Labels and logits
        if return_labels_logits:
            ret_vals = ret_vals + (labels_all, logits_all, preds_all,)

        return ret_vals
