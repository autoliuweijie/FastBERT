# coding:utf-8
import os, sys


from .config import LIB_DIR, FASTBERT_HOME_DIR
sys.path.append(LIB_DIR)
if not os.path.exists(FASTBERT_HOME_DIR):
    os.mkdir(FASTBERT_HOME_DIR)


from .fastbert import FastBERT
from .fastbert import FastBERT_S2

