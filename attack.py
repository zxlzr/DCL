import OpenAttack
import torch

def prepare_data():
    vocab = {
        "<UNK>": 0,
        "<PAD>": 1
    }
    train, valid, test = OpenAttack.loadDataset("SST")
    tp = OpenAttack.text_processors.DefaultTextProcessor()
    for dataset in [train, valid, test]:
        for inst in dataset:
            inst.tokens = list(map(lambda x:x[0], tp.get_tokens(inst.x)))
            for token in inst.tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
    return train, valid, test, vocab

train, valid, test, vocab = prepare_data()