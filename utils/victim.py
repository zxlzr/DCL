import sys
import os
import re
import json
import logging
import OpenAttack
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from OpenAttack.utils.dataset import Dataset, DataInstance
from tqdm import tqdm
import logging
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
    BertTokenizer,
    BertForMaskedLM,
)
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures




class GlueClassifier(OpenAttack.Classifier):
    def __init__(self, task_name, max_seq_len, model_name_or_path, label2id, id2label, device, cache_dir=None):
        # Some basic settings
        root_path = '../'
        sys.path.append(root_path)
        self.device = device

        # Load original BERT model
        # sentence_encoder = opennre.encoder.BERTEncoder(
        #     max_length=max_seq_len,
        #     pretrain_path=os.path.join(
        #         root_path, 'pretrain/bert-base-uncased'),
        #     mask_entity=False)

        self.task_name = task_name.lower()
        try:
            self.num_labels = glue_tasks_num_labels[self.task_name]
            self.output_mode = glue_output_modes[self.task_name]
        except KeyError:
            raise ValueError("Task not found: %s" % (self.task_name))

        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=task_name,
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=self.config,
            cache_dir = cache_dir
        )
        self.model.load_state_dict(torch.load('./glue_model/'+task_name+'/pytorch_model.bin'))
        self.id2label = id2label
        self.model.to(self.device)
        # self.model.load_state_dict(torch.load('/home/chenxiang/pretrain_model/transformers/examples/text-classification/new_result/bert_direct_finetune/CoLA/lr_5/checkpoint-804/pytorch_model.bin')
        self.max_seq_len = max_seq_len
        self.current_label = -1

    # def tokenize_word_list(self, word_list):
    #     return self.tokenizer.tokenize(' '.join(word_list))

    def infer(self, sample):
        model = self.model
        model.eval()

        # batch_encoding = self.tokenizer(
        #     [(sample[0], sample[1])],
        #     max_length=self.max_seq_len,
        #     padding="max_length",
        #     truncation=True,
        # )
        batch_encoding = self.tokenizer(
            [(sample[0])],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: torch.tensor(batch_encoding[k]) for k in batch_encoding}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        logits = model(**inputs)[0]
        logits = F.softmax(logits,dim=1)
        score, pred = logits.max(-1)
        return self.id2label[pred.item()], score.item()

    def get_prob(self, input_):
        ret = []
        self.model.eval()
        correct_answer = np.zeros(len(self.id2label))
        correct_answer[self.current_label] = 1.0
        for sent in input_:
            # valid = True
            # for special in ['sep']:
            #     if sent.count(special) != 1:
            #         valid = False
            #         break
            # # Ignore sentences whose special tokens are not valid!
            # if not valid:
            #     ret.append(correct_answer)
            #     continue
            # Convert data instance to sample
            # sent_list = sent.split(';')
            # text_a = sent_list[0].strip()
            # text_b = sent_list[1].strip()
            sample = (sent,self.id2label[self.current_label])

            # Predict sample label
            batch_encoding = self.tokenizer(
                [sample[0]],
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
            )
            inputs = {k: torch.tensor(batch_encoding[k]) for k in batch_encoding}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs)[0]
                logits = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            ret.append(logits)

        return np.array(ret)


def sample2data(sample, label2id):
    # Convert a single sample to a DataInstance
    # Process the sentence by adding indicating tokens to head / tail tokens

    # text_a, text_b = sample[0:2]
    # sent=text_a + ' ; ' + text_b
    return DataInstance(x=sample[0], y=label2id[sample[-1]])


def data2sample(data, id2label):
    sent, label = data.x.strip(), id2label[data.y]
    # text_a = sent.split(';')[0].strip()
    # text_b = sent.split(';')[1].strip()

    # Convert into a legal sample
    return (sent, label)


def sample2dataset(sample_list, label2id):
    # Convert list of samples to dataset object
    data_list = []

    for sample in sample_list:
        data = sample2data(sample, label2id)
        data_list.append(data)
    dataset = Dataset(data_list=data_list)

    return dataset


def dataset2sample(dataset, id2label):
    # Convert dataset object to list of samples
    sample_list = []
    for data in dataset:
        sample = data2sample(data, id2label)
        sample_list.append(sample)

    return sample_list


if __name__ == "__main__":
    # Test sample-data convert functions
    fin, fout = sys.argv[1:]
    rel2id = json.load(open('../data/tacred/rel2id.json', 'r'))
    id2rel = {v: k for k, v in rel2id.items()}
    samples = []
    for line in open(fin, 'r').readlines():
        samples.append(json.loads(line.strip()))
    with open(fout, 'w') as f:
        for sample in samples:
            f.write('original:' + json.dumps(sample) + '\n')
            data = sample2data(sample, rel2id)
            sample2 = data2sample(data, id2rel)
            f.write('after  2:' + json.dumps(sample2) + '\n')
            f.write('after  1:' + data.x + '\n')
            try:
                assert sample2['h']['pos'] == sample['h']['pos'] and sample2['t']['pos'] == sample['t']['pos']
            except Exception:
                print(sample)
                print(sample2)
                print()
