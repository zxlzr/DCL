from victim import *
import argparse
import logging
import torch
import json
import os
import csv
from tqdm import tqdm
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures


logging.basicConfig(level=logging.INFO)
# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True,
                    help='Where the input file containing full dataset is')
parser.add_argument('--task_name', '-t', type=str, required=True,
                    help='Which task is')
parser.add_argument('--model_path', '-m', type=str, required=True,
                    help='Full path for loading weights of model to attack')
parser.add_argument('--output_file', '-o', type=str, required=True,
                    help='Place to save samples that model predicts correctly')
parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                    help='Maximum sequence length of bert model')

args = parser.parse_args()

# Initialize model
label_list = ["0", "1"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
device = torch.device('cuda:0')
model = GlueClassifier(args.task_name, args.max_seq_len, args.model_path, label2id, id2label, device)

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        if test_mode:
            lines = lines[1:]
        text_index = 1 if test_mode else 3
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



class SnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def _read_tsv( input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


# Filter out correct samples
correct_samples = []
full_samples = []
lines = _read_tsv(os.path.join(args.input_file, "dev.tsv"))
for (i,line) in enumerate(tqdm(lines, desc='Filtering correct samples')):
    # if i == 0:
    #     continue
    text_a = line[3]
    gold_label = line[1]
    # sample= (text_a,text_b,gold_label)
    sample = (text_a, gold_label)
    full_samples.append(sample)
    pred_label, _ = model.infer(sample)
    if str(pred_label) != str(sample[-1]):
        continue
    correct_samples.append(sample)

# Dump correct samples
with open(os.path.join(args.output_file,'dev_correct.txt'), 'w') as f:
    for sample in tqdm(correct_samples, desc='Dumping correct samples'):
        f.write(json.dumps(sample))
        f.write('\n')
    f.close()
with open(os.path.join(args.output_file, 'dev_all.txt'), 'w') as g:
    for sample in tqdm(full_samples, desc='Dumping full samples'):
        g.write(json.dumps(sample))
        g.write('\n')
    g.close()
logging.info('Dumped {} correct samples to {}'.format(
    len(correct_samples), args.output_file))
logging.info('Dumped {} full samples to {}'.format(
    len(full_samples), args.output_file))
