# coding: utf-8
"""
An example of training two sentences classification model with
QNLI dataset.

@author: Weijie Liu
"""
import os
import torch
import sys
fastbert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(fastbert_dir)
from fastbert import FastBERT_S2


train_dataset_path = "../../datasets/lcqmc/train.tsv"
dev_dataset_path = "../../datasets/lcqmc/dev.tsv"
model_saving_path = "/tmp/fastbert_lcqmc.bin"


def loading_dataset(dataset_path):
    sents_a, sents_b, labels = [], [], []
    with open(dataset_path, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i == 0:
                continue
            line = line.strip().split('\t')
            sents_a.append(line[0])
            sents_b.append(line[1])
            labels.append(line[2])
    return sents_a, sents_b, labels


def main():

    sents_a_train, sents_b_train, labels_train = loading_dataset(train_dataset_path)
    sents_a_dev, sents_b_dev, labels_dev = loading_dataset(dev_dataset_path)
    labels = list(set(labels_train))
    print("Labels: ", labels)  # [0, 1]

    # FastBERT
    model = FastBERT_S2(
        kernel_name="uer_bert_tiny_zh",
        labels=labels,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # # FastGPT
    # model = FastGPT_S2(
    #     kernel_name="uer_gpt_zh",
    #     labels=labels,
    #     device="cuda:0" if torch.cuda.is_available() else "cpu"
    # )

    model.fit(
        sents_a_train=sents_a_train,
        sents_b_train=sents_b_train,
        labels_train=labels_train,
        sents_a_dev=sents_a_dev,
        sents_b_dev=sents_b_dev,
        labels_dev=labels_dev,
        batch_size=32,
        finetuning_epochs_num=5,
        distilling_epochs_num=10,
        learning_rate=2e-5,
        report_steps=500,
        model_saving_path=model_saving_path,
        verbose=True,
    )


if __name__ == "__main__":
    main()
