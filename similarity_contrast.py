
import os
import sys
import torch
import yaml
from torchvision import datasets
from models.mlp_head import MLPHead
from models.bert_base_network import BertNet
from trainer import BertBYOLTrainer
from dataclasses import dataclass, field
from typing import Optional
import logging
import numpy as np
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置当前使用的GPU设备仅为0号设备

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BYOLDataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
    RobertaModel,
    RobertaTokenizer,
    RobertaForMaskedLM,
    BertForMaskedLM,
    BertTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    glue_processors
)
from utils.attack_util import GlueAttackDataset

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def prepare_bert():
    model_class = BertForMaskedLM
    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-uncased'

    # 载入预训练的tokenizer和model
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    # model1 = RobertaModel.from_pretrained(pretrained_weights)
    # id = tokenizer.encode("Here is some text to encode", add_special_tokens=True)
    # input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])
    # with torch.no_grad():
    #     last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
    return tokenizer, model


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    byol_config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # 导入roberta相关模型
    tokenizer, bertformlm = prepare_bert()
    bertformlm.resize_token_embeddings(len(tokenizer))
    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

        # Get datasets

    # online encoder
    online_network = BertNet(bert=bertformlm, norm_type='power', **byol_config['network'])


    # load pre-trained model if defined
    if model_args.model_name_or_path:
        try:

            # load pre-trained parameters
            # online2 = BertNet(bert=robertaformlm,norm_type='batch', **byol_config['network'])
            # load_params = torch.load(os.path.join(os.path.join('wikidata_pretrain_roberta/wikidata160_172', 'pytorch_model.bin')),
            #                          map_location=torch.device(torch.device(training_args.device)))
            #
            # online2.load_state_dict(load_params['online_network_state_dict'])
            # online_network.roberta.load_state_dict(online2.roberta.state_dict())
            # del load_params,online2

            load_params = torch.load(
                os.path.join(os.path.join('wikidata_pretrain_roberta/wikidata160_172', 'pytorch_model.bin')),
                                          map_location=torch.device(torch.device(training_args.device)))
            online_network.load_state_dict(load_params['online_network_state_dict'])
            online_network.to(training_args.device)
            logger.info("Training online_network parameters from %s", model_args.model_name_or_path)
        except FileNotFoundError:
            logger.info("Pre-trained weights not found. Training from scratch.")

    def bit_product_sum(x, y):
        return sum([item[0] * item[1] for item in zip(x, y)])

    def cosine_similarity(x, y, norm=False):
        """ 计算两个向量x和y的余弦相似度 """
        # method 2
        cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))
        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

    features = []
    raw_sample = ''
    pos_sample = ''
    neg_sample = ''

    for sample in [raw_sample, pos_sample, neg_sample]:
        batch_encoding = tokenizer(
            [sample],
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: torch.tensor(batch_encoding[k]) for k in batch_encoding}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(training_args.device)

        with torch.no_grad():
            sequence_output, pooled_output = online_network.roberta(**inputs)
            pooled_output = pooled_output.squeeze(0).cpu().numpy()

        features.append(pooled_output)

    pos_similarity = cosine_similarity(features[0],features[1])
    neg_similarity = cosine_similarity(features[0],features[2])

    print(pos_similarity, neg_similarity)




if __name__ == '__main__':
    main()
