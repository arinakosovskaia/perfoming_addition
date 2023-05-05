import argparse
import glob
import json
import os
import pytorch_lightning as pl
import random
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "True"

from argparse import Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from typing import List

from model import T5Finetuner
from data import MyDataset, make_sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evalute T5 on arithmetic problems.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save checkpoint and results.')
    
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument("--seed", default=123, type=int, help="Seed.")
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length (in tokens).')
    parser.add_argument("--test_size", default=2000, type=int, help="Number of examples for testing.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of CPU workers for loading data.")
    parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--min_digits_test', type=int, required=True,
        help='Minimum number of digits sampled for test examples.')
    parser.add_argument(
        '--max_digits_test', type=int, required=True,
        help='Maximum number of digits sampled for test examples.')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint_name, map_location=lambda storage, loc: storage)
    
    new_args = {**checkpoint["hyper_parameters"], **vars(args)}
    new_args = Namespace(**new_args)
    trainer = pl.Trainer.from_argparse_args(new_args)
    
    dataset_test = MyDataset(n_examples=new_args.test_size,
                             min_digits=new_args.min_digits_test,
                             max_digits=new_args.max_digits_test)
    
    test_dataloader = DataLoader(dataset_test, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers)
                                 
                                                                                 
    model = T5Finetuner.load_from_checkpoint(args.checkpoint_name, hparams=new_args,
                                             test_dataloader=test_dataloader)
                                            
                                            
                                
    results = trainer.test(model)
    print("Test accuracy: ", results[0]['test_exact_match'])
