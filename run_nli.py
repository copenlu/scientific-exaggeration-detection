import os
import random
import numpy as np
import torch
import wandb
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from metrics import ClassificationEvaluator
from metrics import acc_f1
import ipdb

from datareader import GoldSuttonDataset
from datareader import ClassificationDataset
from datareader import text_to_batch_transformer
from datareader import NLI_LABELS
from trainer import TransformerClassificationTrainer


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_loc", help="Location of the training data", default=None, type=str)
    parser.add_argument("--pet_data", help="Location of the PET soft labelled data", default=None,
                        type=str)

    parser.add_argument("--eval_metric", help="Metric to use for validation", default='F1',
                        type=str)

    parser.add_argument("--test_data_loc", help="Location of the test data", required=True, type=str)
    parser.add_argument("--test_nli_sentences", action="store_true", default=False, help="Whether to use sentences when testing NLI formulation")

    parser.add_argument("--model_name",
                        help="The name of the model being tested. Can be a directory for a local model",
                        required=True, type=str)
    parser.add_argument("--model_dir", help="Top level directory to save the models", required=True, type=str)

    parser.add_argument("--run_name", help="A name for this run", required=True, type=str)
    parser.add_argument("--tag", help="A tag to give this run (for wandb)", required=True, type=str)

    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--temperature", help="The temperature to use for distillation loss", type=float, default=1.0)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--learning_rate", help="The learning rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", help="Amount of weight decay", type=float, default=0.0)
    parser.add_argument("--dropout_prob", help="The dropout probability", type=float, default=0.1)
    parser.add_argument("--n_epochs", help="The number of epochs to run", type=int, default=2)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--warmup_steps", help="The number of warmup steps", type=int, default=200)
    parser.add_argument("--balance_class_weight", action="store_true", default=False, help="Whether or not to use balanced class weights")

    args = parser.parse_args()

    seed = args.seed
    # Always first
    enforce_reproducibility(seed)

    lr = args.learning_rate
    weight_decay = args.weight_decay
    warmup_steps = args.warmup_steps
    dropout_prob = args.dropout_prob
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    class_weights = 'balanced' if args.balance_class_weight else None
    model_name = args.model_name
    use_scheduler = True
    num_labels = [3]
    # if not multi_task:
    #     num_labels = num_labels[args.eval_task]
    assert batch_size % args.n_gpu == 0, "Batch must be divisible by the number of GPUs used"
    assert (args.train_data_loc != None or args.pet_data != None), "Need to specify some training data"

    config = {
            "epochs": n_epochs,
            "learning_rate": lr,
            "warmup": warmup_steps,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "model_name": model_name,
            "seed": seed,
            "use_scheduler": use_scheduler,
            "balance_class_weight": args.balance_class_weight,
            "temperature": args.temperature,
            "nli-sentences": args.test_nli_sentences,
            "eval-metric": args.eval_metric
        }
    # wandb initialization
    run = wandb.init(
        project="computational-science-journalism",
        name=args.run_name,
        config=config,
        reinit=True,
        tags=[args.tag]
    )
    wandb_path = Path(wandb.run.dir)

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # Train the model
    data_loc = args.train_data_loc
    model = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_fn = text_to_batch_transformer


    if args.pet_data is not None:
        pet_dset = ClassificationDataset(args.pet_data, tokenizer, tokenizer_fn=tokenizer_fn, soft_labels=True, nli=True)
        # train_dset += [pet_dset]
        # num_labels += [4]
        train_dset = [pet_dset]
        num_labels = [3]
        class_weights = None
        valid_dset = [dset] if args.eval_metric != 'none' else []
    else:
        dset = ClassificationDataset(data_loc, tokenizer, tokenizer_fn=tokenizer_fn, nli=True)
        if args.eval_metric == 'none':
            train_dset = [dset]
            valid_dset = []
        else:
            train_size = int(len(dset) * 0.8)
            val_size = len(dset) - train_size
            subsets = random_split(dset, [train_size, val_size])
            train_dset = [subsets[0]]
            valid_dset = [subsets[1]]

    trainer = TransformerClassificationTrainer(
        model,
        device,
        num_labels=num_labels,
        tokenizer=tokenizer,
        multi_gpu=args.n_gpu > 1
    )

    # Create a new directory to save the model
    model_dir = f"{args.model_dir}/{wandb.run.id}"
    # Create save directory for model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Train it
    trainer.train(
        train_dset,
        valid_dset,
        weight_decay=weight_decay,
        model_file=f"{model_dir}/model.pth",
        class_weights=class_weights,
        metric_name=args.eval_metric,
        logger=wandb,
        lr=lr,
        warmup_steps=warmup_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        use_scheduler=use_scheduler,
        eval_averaging=['macro'],
        temperature=args.temperature
    )


    # If we get actual test data
    test_dset = GoldSuttonDataset(args.test_data_loc, tokenizer, None, tokenizer_fn=tokenizer_fn, mode=GoldSuttonDataset.NLI_SENTENCES if args.test_nli_sentences else GoldSuttonDataset.NLI)
    validation_evaluator = ClassificationEvaluator(
        test_dset,
        device,
        num_labels=3,
        averaging='macro',
        pad_token_id=tokenizer.pad_token_id,
        multi_gpu=args.n_gpu > 1,
        batch_size=16
    )
    (loss, acc,P,R,F1) = validation_evaluator.evaluate(trainer.model)


    wandb.run.summary['NLI-acc'] = acc
    wandb.run.summary['NLI-P'] = P
    wandb.run.summary['NLI-R'] = R
    wandb.run.summary['NLI-F1'] = F1