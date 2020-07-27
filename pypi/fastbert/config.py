# coding: utf-8
import os


__version__ = "0.1.1"


LIB_DIR = os.path.dirname(os.path.abspath(__file__))
USER_HOME_DIR = os.path.expanduser('~')
FILES_DIR = os.path.join(LIB_DIR, 'files/')
FASTBERT_HOME_DIR = os.path.join(USER_HOME_DIR, '.fastbert/')
TMP_DIR = '/tmp/'


MODEL_CONFIG_FILE = {
    'google_bert_base_en': os.path.join(FILES_DIR, 'google_bert_base_en.json'),
    'google_bert_base_zh': os.path.join(FILES_DIR, 'google_bert_base_zh.json'),
    'facebook_roberta_base_zh': os.path.join(FILES_DIR, 'facebook_roberta_base_zh.json'),
    'facebook_roberta_base_en': os.path.join(FILES_DIR, 'facebook_roberta_base_en.json'),
    'uer_bert_large_zh': os.path.join(FILES_DIR, 'uer_bert_large_zh.json'),
    'uer_bert_small_zh': os.path.join(FILES_DIR, 'uer_bert_small_zh.json'),
    'uer_bert_tiny_zh': os.path.join(FILES_DIR, 'uer_bert_tiny_zh.json'),
    'uer_gpt_zh': os.path.join(FILES_DIR, 'uer_gpt_zh.json'),
    'uer_gpt_en': os.path.join(FILES_DIR, 'uer_gpt_en.json'),
    'uer_gcnn_13_zh': os.path.join(FILES_DIR, 'uer_gcnn_13_zh.json'),
    'uer_gcnn_9_zh': os.path.join(FILES_DIR, 'uer_gcnn_9_zh.json'),
}


DEFAULT_SEQ_LENGTH = 128  # Default sentence length.
DEFAULT_DEVICE = 'cpu'  # Default device.

