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

import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Evalute T5 on arithmetic problems.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save checkpoint and results.')
    parser.add_argument('--first_number', type=int, required=True, help='First number')
    parser.add_argument('--second_number', type=int, required=True, help='Second number')
    parser.add_argument("--seed", default=1, type=int, help="Seed.")
    parser.add_argument('--checkpoint_name', default='epoch=4-val_exact_match=0.9960.ckpt', type=str)
    parser.add_argument('--max_seq_length', type=int, default=1024, help='Maximum sequence length (in tokens).')
    parser.add_argument("--num_workers", default=4, type=int, help="Number of CPU workers for loading data.")

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint_name, map_location=lambda storage, loc: storage)
    
    new_args = {**checkpoint["hyper_parameters"], **vars(args)}
    new_args = Namespace(**new_args)
    trainer = pl.Trainer.from_argparse_args(new_args)
    
    dataset_test = MyDataset(n_examples=1, examples=[[args.first_number, args.second_number]])
        
    test_dataloader = DataLoader(dataset_test, batch_size=1,
                                 shuffle=False, num_workers=args.num_workers)
                                 
                                                                                 
    model = T5Finetuner.load_from_checkpoint(args.checkpoint_name, hparams=new_args,
                                             test_dataloader=test_dataloader)
                                            
                                            
                                
    result = trainer.predict(model, test_dataloader)[0]
    result = "Predicted answer" + str(result)
    print(result)
    
    with open(os.path.join(args.output_dir, 'prediction.txt'), 'w') as fout:
        fout.write(result)
