# coding:utf-8
import os, sys
from .config import *


sys.path.append(LIB_DIR)
if not os.path.exists(FASTBERT_HOME_DIR):
    os.mkdir(FASTBERT_HOME_DIR)


from .fastbert import FastBERT
from .fastbert import FastBERT_S2
from .fastgpt import FastGPT
from .fastgpt import FastGPT_S2
from .fastgcnn import FastGCNN


SINGLE_SENTENCE_CLS_KERNEL_MAP = {
    'google_bert_base_en': FastBERT,
    'google_bert_base_zh': FastBERT,
    'uer_bert_large_zh': FastBERT,
    'uer_bert_small_zh': FastBERT,
    'uer_bert_tiny_zh': FastBERT,
    'uer_gpt_zh': FastGPT,
    'uer_gpt_en': FastGPT,
}
def single_sentence_cls(kernel_name,
                        **kwargs):
    model = SINGLE_SENTENCE_CLS_KERNEL_MAP[kernel_name](kernel_name, **kwargs)
    return model


DOUBLE_SENTENCE_CLS_KERNEL_MAP = {
    'google_bert_base_en': FastBERT_S2,
    'google_bert_base_zh': FastBERT_S2,
    'uer_bert_large_zh': FastBERT_S2,
    'uer_bert_small_zh': FastBERT_S2,
    'uer_bert_tiny_zh': FastBERT_S2,
    'uer_gpt_zh': FastGPT_S2,
    'uer_gpt_en': FastGPT_S2,
}
def double_sentence_cls(kernel_name,
                        **kwargs):
    model = DOUBLE_SENTENCE_CLS_KERNEL_MAP[kernel_name](kernel_name, **kwargs)
    return model

