# coding: utf-8
"""
Some utils for fastbert.

@author: Weijie Liu
"""
import os
import json
import torch
import random
import numpy as np
from argparse import Namespace
import urllib
import hashlib
from functools import partial
import ssl
from .config import FASTBERT_HOME_DIR
ssl._create_default_https_context = ssl._create_unverified_context


def md5sum(filename):
    with open(filename, 'rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()


def hello():
    print("Hello FastBERT!")


def load_hyperparam(config_path,
                    file_dir=None,
                    args=None):
    with open(config_path, "r", encoding="utf-8") as f:
        param = json.load(f)
        for key, value in param.items():
            if isinstance(key, str) and key.endswith('_path'):
                if isinstance(value, str) and value.endswith('.bin'):
                    param[key] = os.path.join(FASTBERT_HOME_DIR, value)
                else:
                    param[key] = os.path.join(file_dir, value)

    if args is None:
        args_dict = {}
    else:
        args_dict = vars(args)
    args_dict.update(param)

    args = Namespace(**args_dict)
    return args


def cbk_for_urlretrieve(a, b, c):
    '''
    Callback function for showing process
    '''
    per = 100.0 * a * b / c
    if per > 100:
        per = 100
    print('\r%.1f%% of %.2fM' % (per,c/(1024*1024)), end='')


def check_or_download(file_path,
                      file_url,
                      file_md5,
                      file_name='',
                      file_url_bak=None):
    is_exist = False
    if os.path.exists(file_path):
        this_file_md5 = md5sum(file_path)
        if this_file_md5 == file_md5:
            is_exist = True
        else:
            os.remove(file_path)

    if not is_exist:
        print("{} are not exist or md5 is wrong.".format(file_path))
        print("Download {} file from {}".format(file_name, file_url))
        try:
            urllib.request.urlretrieve(file_url, file_path, cbk_for_urlretrieve)
            this_file_md5 = md5sum(file_path)
            if this_file_md5 == file_md5:
                print("\nDownload {} file successfully.".format(file_name))
            else:
                raise Exception("Md5 wrong.")
        except Exception as error:
            infos = "\n[Error]: Download {} file failed!".format(file_name)
            options = \
                "[Option]: You can download the file from [URL_A] or [URL_B], " + \
                "and save it as [PATH] by yourself. \n" + \
                "URL_A: {}\nURL_B:{}\nPATH: {} ". \
                format(file_url, file_url_bak, file_path)
            raise Exception(infos + '\n' + options)


def calc_uncertainty(p,
                     labels_num):
    entropy = torch.distributions.Categorical(probs=p).entropy()
    normal = -np.log(1.0/labels_num)
    return entropy / normal


def shuffle_pairs(list_a,
                  list_b):
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(list_a)
    random.seed(randnum)
    random.shuffle(list_b)
    return list_a, list_b
