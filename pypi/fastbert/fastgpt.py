# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from .config import *
from .utils import *
from .fastbert import FastBERT, FastBERT_S2


class GptMiniClassifier(nn.Module):

    def __init__(self,
                 args,
                 input_size,
                 labels_num):
        super(GptMiniClassifier, self).__init__()
        self.input_size = input_size
        self.labels_num = labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(input_size, input_size)
        self.output_layer_2 = nn.Linear(input_size, labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,
                hidden,
                mask):
        output = hidden
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        return logits


class FastGPT(FastBERT):

    MiniClassifier = GptMiniClassifier

    def _mask_transfer(self,
                       mask, # batch_size x seq_length
                       emb):
        batch_size, seq_length, emb_size = emb.size()
        mask = torch.ones(seq_length, seq_length, device=emb.device)
        mask = torch.tril(mask)
        mask = (1.0 - mask) * -10000.0
        mask = mask.repeat(batch_size, 1, 1, 1)
        return mask


class FastGPT_S2(FastBERT_S2, FastGPT):

    MiniClassifier = FastGPT.MiniClassifier

    def _mask_transfer(self,
                       mask,
                       emb): 
        return FastGPT._mask_transfer(self, mask, emb)


