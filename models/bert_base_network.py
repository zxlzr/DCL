import torchvision.models as models
import torch
from models.mlp_head import MLPHead
from torch import nn

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class BertNet(torch.nn.Module):
    def __init__(self, bert, norm_type, **kwargs):
        super(BertNet, self).__init__()

        # self.bert_encoder = torch.nn.Sequential(*list(bert.children())[:-1])
        # self.mlm_encoder = torch.nn.Sequential(list(bert.children())[-1])
        self.config = bert.config
        self.roberta = bert.bert
        self.lm_head = bert.cls
        self.norm_type = norm_type
        self.projetion = MLPHead(self.norm_type, in_channels=self.roberta.pooler.dense.in_features,  **kwargs['projection_head'],layer=1)

    def forward(self, x):
        sequence_output, pooled_output = self.roberta(x)
        prediction_scores = self.lm_head(sequence_output)
        projection = self.projetion(pooled_output)

        return prediction_scores, projection
