from .base_data_module import BaseDataModule
from .processor import get_dataset, processors
from transformers import AutoTokenizer



class DIALOGUE(BaseDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)

        self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
        self.num_training_steps = len(self.data_train) // self.batch_size // self.args.accumulate_grad_batches * self.args.max_epochs
        self.num_labels = len(self.processor.get_labels())



    def setup(self, stage=None):
        self.data_val = get_dataset("val", self.args, self.tokenizer, self.processor)
        self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)

    def prepare_data(self):
        pass


    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--task_name", type=str, default="normal", help="Number of examples to operate on per forward step.")
        parser.add_argument("--model_name_or_path", type=str, default="/home/xx/bert-base-uncased", help="Number of examples to operate on per forward step.")
        parser.add_argument("--max_seq_length", type=int, default=512, help="Number of examples to operate on per forward step.")
        return parser