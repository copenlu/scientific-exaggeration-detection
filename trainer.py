import torch
import gc
import random
import numpy as np
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from typing import AnyStr, Union, List
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import ipdb

from datareader import collate_batch_transformer
from datareader import collate_batch_transformer_with_weight
from metrics import ClassificationEvaluator
from metrics import distillation_loss
from model import AutoTransformerMultiTask
from model import AutoTransformerForSentenceSequenceModelingMultiTask


class AbstractTransformerTrainer:
    """
    An abstract class which other trainers should implement
    """
    def __init__(
            self,
            model = None,
            device = None,
            tokenizer = None
    ):

        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    def create_optimizer(self, lr: float, weight_decay: float=0.0):
        """
        Create a weighted adam optimizer with the given learning rate
        :param lr:
        :return:
        """

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return AdamW(optimizer_grouped_parameters, lr=lr)

    def save(self, model_file: AnyStr):
        """
        Saves the current model
        :return:
        """
        if not Path(model_file).parent.exists():
            Path(model_file).parent.mkdir(parents=True, exist_ok=True)
        if self.multi_gpu:
            save_model = self.model.module
        else:
            save_model = self.model
        torch.save(save_model.state_dict(), model_file)

    def load(self, model_file: AnyStr, new_classifier: bool = False, add_missing_keys: bool = True):
        """
        Loads the model given by model_file
        :param model_file:
        :return:
        """
        if self.multi_gpu:
            model_dict = self.model.module.state_dict()
            load_model = self.model.module
        else:
            model_dict = self.model.state_dict()
            load_model = self.model
        if new_classifier:
            weights = {k: v for k, v in torch.load(model_file, map_location=lambda storage, loc: storage).items() if "classifier" not in k and "pooler" not in k}
            model_dict.update([(k, weights[k]) for k in weights if k in model_dict or add_missing_keys])
            load_model.load_state_dict(model_dict)
        else:
            weights = torch.load(model_file, map_location=lambda storage, loc: storage)
            model_dict.update([(k, weights[k]) for k in weights if k in model_dict or add_missing_keys])
            load_model.load_state_dict(model_dict)

    def freeze(self, exclude_params: List=[]):
        """
        Freeze the model weights
        :return:
        """
        for n,p in self.model.named_parameters():
            if n not in exclude_params:
                p.requires_grad = False


class TransformerClassificationTrainer(AbstractTransformerTrainer):
    """
    A class to encapsulate all of the training and evaluation of a
    transformer model for classification
    """
    def __init__(
            self,
            transformer_model: Union[AnyStr, torch.nn.Module],
            device: torch.device,
            num_labels: Union[int, List],
            multi_task: bool = False,
            tokenizer=None,
            multi_gpu: bool = False,
            sequence_modeling = False
    ):
        if type(num_labels) != list:
            num_labels = [num_labels]

        if type(transformer_model) == str:
            self.model_name = transformer_model
            # Create the model
            if multi_task:
                if sequence_modeling:
                    self.model = AutoTransformerForSentenceSequenceModelingMultiTask(transformer_model, num_labels).to(device)
                else:
                    self.model = AutoTransformerMultiTask(transformer_model, num_labels).to(device)
            else:
                config = AutoConfig.from_pretrained(transformer_model, num_labels=num_labels[0])
                self.model = AutoModelForSequenceClassification.from_pretrained(transformer_model, config=config).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        else:
            self.model_name = 'custom'
            self.model = transformer_model
            self.tokenizer = tokenizer
            if tokenizer == None:
                print("WARNING: No tokenizer passed to trainer, incorrect padding token may be used.")

        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
        self.device = device
        self.num_labels = num_labels
        self.multi_task = multi_task
        self.multi_gpu = multi_gpu

    def evaluate(
            self,
            validation_dset: Dataset,
            num_labels: int,
            eval_averaging: AnyStr = 'micro',
            return_labels_logits: bool = False,
            task_idx: int = None,
            temperature: int = 1.0
    ):
        """
        Runs a round of evaluation on the given dataset
        :param validation_dset:
        :return:
        """
        if self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 0
        # Create the validation evaluator
        validation_evaluator = ClassificationEvaluator(
            validation_dset,
            self.device,
            num_labels=num_labels,
            averaging=eval_averaging,
            pad_token_id=pad_token_id,
            multi_gpu=self.multi_gpu,
            multi_task=self.multi_task,
            task_idx=task_idx,
            temperature=temperature
        )
        return validation_evaluator.evaluate(self.model, return_labels_logits=return_labels_logits)

    def train(
            self,
            train_dset: Union[List, Dataset],
            validation_dset: Union[List, Dataset] = [],
            logger = None,
            lr: float = 3e-5,
            n_epochs: int = 2,
            batch_size: int = 8,
            weight_decay: float = 0.0,
            warmup_steps: int = 200,
            log_interval: int = 1,
            metric_name: AnyStr = 'accuracy',
            patience: int = 10,
            model_file: AnyStr = "model.pth",
            class_weights=None,
            use_scheduler: bool = True,
            eval_averaging: AnyStr = ['binary'],
            lams: List = None,
            clip_grad: float = None,
            num_dataset_workers: int = 10,
            eval_task: int = 0,
            temperature: int = 1.0,
            sequential_multitask: bool = False,
            task: int = None
    ):
        if type(train_dset) != list:
            train_dset = [train_dset]
        if type(validation_dset) != list:
            validation_dset = [validation_dset]
        if self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 0
        if lams is None:
            lams = [1.0] * len(train_dset)

        collate_fn = partial(collate_batch_transformer, pad_token_id)
        distill_loss = partial(distillation_loss, temperature)

        # Create the loss function
        if class_weights is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        elif (isinstance(class_weights, str) and class_weights == 'sample_based_weight'):
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            collate_fn = partial(collate_batch_transformer_with_weight, pad_token_id)
        elif isinstance(class_weights, str) and class_weights == 'balanced':
            # Calculate the weights
            if self.multi_task:
                raise Exception('Not Implemented!')
            if isinstance(train_dset[0], Subset):
                labels = train_dset[0].dataset.getLabels(train_dset[0].indices).astype(np.int64)
            else:
                labels = train_dset[0].getLabels().astype(np.int64)
            weight = torch.tensor(len(labels) / (self.num_labels[0] * np.bincount(labels)))
            weight = weight.type(torch.FloatTensor).to(self.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        elif isinstance(class_weights, List) and self.multi_task:
            loss_fn = [torch.nn.CrossEntropyLoss(weight=torch.tensor(w).type(torch.FloatTensor).to(self.device)) for w in class_weights]
        else:
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).type(torch.FloatTensor).to(self.device))

        # Create the training dataloader(s)
        train_dls = [DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_dataset_workers
        ) for ds in train_dset]

        # Create the optimizer
        optimizer = self.create_optimizer(lr, weight_decay)

        total = sum(len(dl) for dl in train_dls) if task is None else len(train_dls[task])

        if use_scheduler:
            # Create the scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                warmup_steps,
                n_epochs * len(train_dls[0])#total
            )

        # Set up metric tracking
        best_metric = 0.0 if metric_name != 'loss' else -float('inf')
        patience_counter = 0
        # Save before training
        self.save(model_file)
        # Main training loop
        for ep in range(n_epochs):
            # Training loop
            dl_iters = [iter(dl) for dl in train_dls]
            dl_idx = list(reversed(range(len(dl_iters)))) if task is None else [task]
            finished = [0] * len(dl_iters)
            if task is not None:
                finished = [1] * len(dl_iters)
                finished[task] = 0
            i = 0
            with tqdm(total=total, desc="Training") as pbar:
                while sum(finished) < len(dl_iters):
                    if not sequential_multitask:
                        random.shuffle(dl_idx)
                    for d in dl_idx:
                        run_sequential = True
                        while finished[d] != 1 and run_sequential:
                            run_sequential = sequential_multitask
                            task_dl = dl_iters[d]
                            try:
                                batch = next(task_dl)
                            except StopIteration:
                                finished[d] = 1
                                continue

                            self.model.train()
                            optimizer.zero_grad()

                            batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
                            input_ids = batch[0]
                            masks = batch[1]
                            labels = batch[2]

                            if not self.multi_task:
                                outputs = self.model(input_ids, attention_mask=masks)
                                logits = outputs.logits
                            else:
                                logits_mask = batch[3] if len(batch) > 3 else None
                                outputs = self.model(input_ids, attention_mask=masks, logits_mask=logits_mask, task_num=d)
                                logits = outputs[0]

                            # Calculate what the weight of the loss should be
                            loss_weight = lams[d]
                            labels = labels.view(logits.shape[0], -1)
                            if labels.shape[1] > 1:
                                # Soft target loss
                                loss = loss_weight * distill_loss(logits.view(-1, self.num_labels[d]), labels.view(-1, self.num_labels[d]))
                            elif (isinstance(class_weights, str) and class_weights == 'sample_based_weight'):
                                sample_weight = batch[-1]
                                loss_weight *= sample_weight
                                loss = (loss_weight * loss_fn(logits.view(-1, self.num_labels[d]), labels.view(-1))).mean()
                            elif isinstance(loss_fn, List):
                                loss = loss_weight * loss_fn[d](logits.view(-1, self.num_labels[d]), labels.view(-1))
                            else:
                                loss = loss_weight * loss_fn(logits.view(-1, self.num_labels[d]), labels.view(-1))

                            if self.multi_gpu:
                                loss = loss.mean()

                            if i % log_interval == 0 and logger is not None:
                                logger.log({"Loss": loss.item()})
                            loss.backward()
                            # Clip gradients
                            if clip_grad:
                              torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

                            optimizer.step()
                            i += 1
                            pbar.update(1)

                            if use_scheduler:
                                scheduler.step()

            gc.collect()

            if len(validation_dset) == 0:
                self.save(model_file)
            # Inline evaluation
            for i in range(len(validation_dset)):
                (val_loss, acc, P, R, F1) = self.evaluate(validation_dset[i], self.num_labels[i], eval_averaging[i], task_idx=i, temperature=temperature)
                if metric_name == 'accuracy':
                    metric = acc
                elif metric_name == 'loss':
                    metric = -val_loss # negative since we are always maximizing
                else:
                    metric = F1
                    if eval_averaging is None:
                        # Macro average if averaging is None
                        metric = sum(F1) / len(F1)

                print(f"{metric_name}: {abs(metric)}")

                if logger is not None:
                    # Log
                    logger.log({
                        'Validation accuracy - Task {}'.format(i): acc,
                        'Validation Precision - Task {}'.format(i): P,
                        'Validation Recall - Task {}'.format(i): R,
                        'Validation F1 - Task {}'.format(i): F1,
                        'Validation loss - Task{}'.format(i): val_loss}
                    )
                else:
                    print({
                        'Validation accuracy': acc,
                        'Validation Precision': P,
                        'Validation Recall': R,
                        'Validation F1': F1,
                        'Validation loss': val_loss}
                )

                # Saving the best model and early stopping
                # if val_loss < best_loss:
                if i == eval_task:
                    if metric > best_metric:
                        best_metric = metric
                        if logger != None:
                            logger.log({f"best_{metric_name}": abs(metric)})
                        else:
                            print(f"Best {metric_name}: {abs(metric)}")
                        self.save(model_file)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        # Stop training once we have lost patience
                        if patience_counter == patience:
                            break

            gc.collect()

        self.load(model_file)
