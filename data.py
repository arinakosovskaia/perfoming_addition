import argparse
import glob
import json
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import random
import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import torch
from torch.utils.data import DataLoader, Dataset
from typing import List

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def convert_number(number: int) -> str:
    number = str(number)
    signal = None
        
    if number[0] == '-':
        signal = '-'
        number = number[1:]

    output = []
    for i, digit in enumerate(number[::-1]):
        output.append('10e' + str(i))
        output.append(digit)

    if signal:
        output.append(signal)

    output = output[::-1]

    return ' '.join(output)

def make_sample(first_term, second_term):
    operation_term = 'plus'
    result = first_term + second_term

    first_term = convert_number(first_term)
    second_term = convert_number(second_term)
    answer = convert_number(result)

    return f'What is {first_term} {operation_term} {second_term}?', answer

class MyDataset(Dataset):
    def __init__(self, n_examples: int = 1, min_digits: int = 2, max_digits: int = 60, examples=None):
        self.max_digits = max_digits

        if examples:
            self.examples = examples
        else:
            self.examples = []
            for _ in range(n_examples):
                example = []
                for _ in range(2):
                    max_digits_i = random.randint(min_digits, max_digits)
                    min_number = int((max_digits_i - 1) * '9') + 1
                    max_number = int(max_digits_i * '9')
                    example.append(random.randint(min_number, max_number))
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        first_term, second_term = self.examples[idx]

        operation_term = 'plus'
        result = first_term + second_term

        first_term = convert_number(first_term)
        second_term = convert_number(second_term)
        answer = convert_number(result)

        return f'What is {first_term} {operation_term} {second_term}?', answer
