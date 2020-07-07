# coding: utf-8
import sys
sys.path.append("../")
from fastbert import FastBERT_S2


def load_dataset(path):

    sents, labels = [], []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split('\t')
            sents.append(line[1])
            labels.append(line[0])

    return sents, labels,



def main():

    train_dataset_path = '../datasets/douban_book_review/train.tsv'
    dev_dataset_path = '../datasets/douban_book_review/dev.tsv'
    test_dataset_path = '../datasets/douban_book_review/test.tsv'
    labels = ['0', '1']
    sents_train, labels_train = load_dataset(train_dataset_path)
    sents_dev, labels_dev = load_dataset(dev_dataset_path)
    sents_test, labels_test = load_dataset(test_dataset_path)

    model = FastBERT_S2("google_bert_base_zh", labels=labels, device='cuda:0')
    model.show()

    model.fit(
        sents_train,
        sents_train,
        labels_train,
        sentences_dev=sents_dev,
        labels_dev=labels_dev,
        finetuning_epochs_num=1,
        distilling_epochs_num=2
    )


if __name__ == "__main__":
    main()

