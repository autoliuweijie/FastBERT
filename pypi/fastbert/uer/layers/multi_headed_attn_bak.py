# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
import time

class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)
        self.mm = lambda x, y: torch.matmul(x,y)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, hidden_size)

        start = time.time()
        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]
        linear_end = time.time()
        print("linear: ", linear_end-start)

        # scores = torch.matmul(query, key.transpose(-2, -1))
        scores = self.mm(query, key.transpose(-2, -1))
        mult_end = time.time()
        print('mul: {}'.format(mult_end - linear_end))

        scores = scores / math.sqrt(float(per_head_size)) 
        sqrt_end = time.time() 
        print('sqrt: {}'.format(sqrt_end-mult_end))

        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output_end = time.time() 
        print('output: {}'.format(output_end-mult_end))

        output = self.final_linear(output)
        final_end = time.time() 
        print('final: {}'.format(final_end-mult_end))
        
        return output
