import argparse
import glob
import json
import os
import pytorch_lightning as pl
import random
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "True"

from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from typing import List

from model import T5Finetuner
from data import MyDataset

import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    #parse arguments
    parser = argparse.ArgumentParser(description='Train T5 on arithmetic problems.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save checkpoint and results.')
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument(
        '--min_digits_train', type=int, default=2,
        help='Minimum number of digits sampled for training and validation examples.')
    parser.add_argument(
        '--min_digits_test', type=int, default=2,
        help='Minimum number of digits sampled for test examples.')
    parser.add_argument(
        '--max_digits_train', type=int, required=True,
        help='Maximum number of digits sampled for training and validation examples.')
    parser.add_argument(
        '--max_digits_test', type=int, default=50,
        help='Maximum number of digits sampled for test examples.')
    parser.add_argument("--seed", default=123, type=int, help="Seed.")
    parser.add_argument("--train_size", default=1000, type=int, help="Number of examples for training.")
    parser.add_argument("--val_size", default=1000, type=int, help="Number of examples for training.")
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length (in tokens).')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--scheduler', type=str, default='StepLR',
                        help='learning rate scheduler. Currently, only StepLR is supported.)')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma factor for ExponentialLR or StepLR')
    parser.add_argument('--step_size', type=int, default=2, help='period of learning rate decay (StepLR)')
    parser.add_argument('--t_0', type=int, default=2,
                        help='number of iterations for the first restart (CosineAnnealingWarmRestarts)')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='a factor increases t_i after a restart (CosineAnnealingWarmRestarts)')
    parser.add_argument("--num_workers", default=4, type=int, help="Number of CPU workers for loading data.")

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    pl.seed_everything(args.seed)

    #train dataset
    dataset_train = MyDataset(n_examples=args.train_size,
                              min_digits=args.min_digits_train,
                              max_digits=args.max_digits_train)
          
    train_dataloader = DataLoader(dataset_train, batch_size=args.train_batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    #validation dataset
    dataset_val = MyDataset(n_examples=args.val_size,
                            min_digits=args.min_digits_train,
                            max_digits=args.max_digits_train)

    val_dataloader = DataLoader(dataset_val, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)

    #to save checkpoints
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir, filename='{epoch}-{val_exact_match:.4f}',
                                          save_top_k=1, mode='max', monitor='val_exact_match', every_n_epochs=1)
    
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])

    model = T5Finetuner(hparams=args, train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader)
    
    #fit model
    trainer.fit(model)
