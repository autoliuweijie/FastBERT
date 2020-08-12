# coding:utf-8
import os, sys
from .config import *


sys.path.append(LIB_DIR)
if not os.path.exists(FASTBERT_HOME_DIR):
    os.mkdir(FASTBERT_HOME_DIR)


from .fastbert import FastBERT
from .fastbert import FastBERT_S2

