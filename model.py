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

def convert_string(string : str) -> int:
    regex = r' ?10e\d* ?'
    digits_array = re.split(regex, string)[:-1] #without last space
    answer = ''.join(digits_array)
    return int(answer)

def compute_exact_match(predicted_answer, correct_answer) -> bool:
    predicted_answer = predicted_answer.strip().lower()
    correct_answer = correct_answer.strip().lower()
    return predicted_answer == correct_answer

class T5Finetuner(pl.LightningModule):

    def __init__(self, hparams, train_dataloader=None, val_dataloader=None, test_dataloader=None):
        super(T5Finetuner, self).__init__()

        self.hparams.update(vars(hparams))
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.model_name_or_path)

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def prepare_batch(self, questions: List[str], answers: List[str]) -> List[str]:
        input_dict = self.tokenizer.batch_encode_plus(
            list(questions), padding=True, truncation=False, return_tensors='pt')

        labels = self.tokenizer.batch_encode_plus(
            list(answers), padding=True, truncation=False,
            return_tensors='pt')['input_ids']

        assert input_dict['input_ids'].shape[1] <= self.hparams.max_seq_length
        assert labels.shape[1] <= self.hparams.max_seq_length

        input_ids = input_dict['input_ids'].to(self.model.device)
        attention_mask = input_dict['attention_mask'].to(self.model.device)
        labels = labels.to(self.model.device)

        return input_ids, attention_mask, labels

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_nb):
        questions, correct_answers = batch

        input_ids, attention_mask, labels = self.prepare_batch(
            questions=questions, answers=correct_answers)

        loss = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)[0]

        self.log('train_loss', loss)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def predict_step(self, batch, batch_nb):
        questions, correct_answers = batch

        input_ids, attention_mask, _ = self.prepare_batch(
            questions=questions, answers=correct_answers)

        batch_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=self.hparams.max_seq_length)

        predicted_answers = [
            self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs]

        return convert_string(predicted_answers[0])


    def inference_step(self, batch, batch_nb: int):
        questions, correct_answers = batch

        input_ids, attention_mask, _ = self.prepare_batch(
            questions=questions, answers=correct_answers)

        batch_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=self.hparams.max_seq_length)

        predicted_answers = [
            self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs]

        exact_matches = [
            compute_exact_match(predicted_answer=predicted_answer, correct_answer=correct_answer)
            for predicted_answer, correct_answer in zip(predicted_answers, correct_answers)]

        metrics = {'exact_matches': exact_matches}
        return metrics

    def validation_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

    def test_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

    def validation_epoch_end(self, outputs):
        exact_matches = []
        for x in outputs:
            exact_matches.extend(x['exact_matches'])
        exact_match = sum(exact_matches) / len(exact_matches)

        metrics = {'val_exact_match': exact_match}
        self.log('val_exact_match', exact_match)

        output = metrics.copy()
        output['progress_bar'] = metrics

        return output

    def test_epoch_end(self, outputs):
        exact_matches = []
        for x in outputs:
            exact_matches.extend(x['exact_matches'])
        exact_match = sum(exact_matches) / len(exact_matches)

        metrics = {'test_exact_match': exact_match}
        self.log('test_exact_match', exact_match)

        output = metrics.copy()
        output['progress_bar'] = metrics

        return output

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def get_optimizer(self):
        optimizer_name = self.hparams.optimizer
        scheduler_name = self.hparams.scheduler
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay

        optimizer = getattr(torch.optim, optimizer_name)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = optimizer(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)

        print(f'=> Using {optimizer_name} optimizer')

        if scheduler_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
            print(f'=> Using StepLR (step_size = {self.hparams.step_size}, gamma = {self.hparams.gamma})')
        else:
            raise Exception(f'Scheduler not implemented: {scheduler_name}')

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer
