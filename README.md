# FastBERT
![](https://img.shields.io/badge/license-MIT-000000.svg)

Source code for ["FastBERT: a Self-distilling BERT with Adaptive Inference Time"](https://www.aclweb.org/anthology/2020.acl-main.537/).


## Good News

**2020/07/05 - Update**: Pypi version of FastBERT has been launched. Please see [fastbert-pypi](https://pypi.org/project/fastbert/).

Install ``fastbert`` with ``pip``
```sh
$ pip install fastbert
```


## Requirements

``python >= 3.4.0``, Install all the requirements with ``pip``.
``` 
$ pip install -r requirements.txt
```


## Quick start on the Chinese Book review dataset

Download the pre-trained Chinese BERT parameters from [here](https://share.weiyun.com/gHkb3N6L), and save it to the ``models`` directory with the name of "Chinese_base_model.bin". 

Run the following command to validate our FastBERT with ``Speed=0.5`` on the Book review datasets.
```sh
$ CUDA_VISIBLE_DEVICES="0" python3 -u run_fastbert.py \
        --pretrained_model_path ./models/Chinese_base_model.bin \
        --vocab_path ./models/google_zh_vocab.txt \
        --train_path ./datasets/douban_book_review/train.tsv \
        --dev_path ./datasets/douban_book_review/dev.tsv \
        --test_path ./datasets/douban_book_review/test.tsv \
        --epochs_num 3 --batch_size 32 --distill_epochs_num 5 \
        --encoder bert --fast_mode --speed 0.5 \
        --output_model_path  ./models/douban_fastbert.bin
```

Meaning of each option.
```
usage: --pretrained_model_path Path to initialize model parameters.
       --vocab_path Path to the vocabulary.
       --train_path Path to the training dataset.
       --dev_path Path to the validating dataset.
       --test_path Path to the testing dataset.
       --epochs_num The epoch numbers of fine-tuning.
       --batch_size Batch size.
       --distill_epochs_num The epoch numbers of the self-distillation.
       --encoder The type of encoder.
       --fast_mode Whether to enable the fast mode of FastBERT.
       --speed The Speed value in the paper.
       --output_model_path Path to the output model parameters.
```

Test results on the Book review dataset.
```
Test results at fine-tuning epoch 3 (Baseline): Acc.=0.8688;  FLOPs=21785247744;
Test results at self-distillation epoch 1     : Acc.=0.8698;  FLOPs=6300902177;
Test results at self-distillation epoch 2     : Acc.=0.8691;  FLOPs=5844839008;
Test results at self-distillation epoch 3     : Acc.=0.8664;  FLOPs=5170940850;
Test results at self-distillation epoch 4     : Acc.=0.8664;  FLOPs=5170940327;
Test results at self-distillation epoch 5     : Acc.=0.8664;  FLOPs=5170940327;
```


## Quick start on the English Ag.news dataset

Download the pre-trained English BERT parameters from [here](https://share.weiyun.com/gHkb3N6L), and save it to the ``models`` directory with the name of "English_uncased_base_model.bin". 

Download the ``ag_news.zip`` from [here](https://share.weiyun.com/ZctQJP8h), and then unzip it to the ``datasets`` directory. 

Run the following command to validate our FastBERT with ``Speed=0.5`` on the Ag.news datasets.
```sh
$ CUDA_VISIBLE_DEVICES="0" python3 -u run_fastbert.py \
        --pretrained_model_path ./models/English_uncased_base_model.bin \
        --vocab_path ./models/google_uncased_en_vocab.txt \
        --train_path ./datasets/ag_news/train.tsv \
        --dev_path ./datasets/ag_news/test.tsv \
        --test_path ./datasets/ag_news/test.tsv \
        --epochs_num 3 --batch_size 32 --distill_epochs_num 5 \
        --encoder bert --fast_mode --speed 0.5 \
        --output_model_path  ./models/ag_news_fastbert.bin
```

Test results on the Ag.news dataset.
```
Test results at fine-tuning epoch 3 (Baseline): Acc.=0.9447;  FLOPs=21785247744;
Test results at self-distillation epoch 1     : Acc.=0.9308;  FLOPs=2172009009;
Test results at self-distillation epoch 2     : Acc.=0.9311;  FLOPs=2163471246;
Test results at self-distillation epoch 3     : Acc.=0.9314;  FLOPs=2108341649;
Test results at self-distillation epoch 4     : Acc.=0.9314;  FLOPs=2108341649;
Test results at self-distillation epoch 5     : Acc.=0.9314;  FLOPs=2108341649;
```


## Datasets

More datasets can be downloaded from [here](https://share.weiyun.com/ZctQJP8h).


## Other implementations

There are some other excellent implementations of FastBERT.

* BitVoyage/FastBERT (Pytorch): https://github.com/BitVoyage/FastBERT


## Acknowledgement

This work is funded by 2019 Tencent Rhino-Bird Elite Training Program. Work done while this author was an intern at Tencent.

If you use this code, please cite this paper:
```
@inproceedings{weijie2020fastbert,
  title={{FastBERT}: a Self-distilling BERT with Adaptive Inference Time},
  author={Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Haotang Deng, Qi Ju},
  booktitle={Proceedings of ACL 2020},
  year={2020}
}
```


