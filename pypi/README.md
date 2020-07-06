# FastBERT-pypi

The pypi version of [FastBERT](https://github.com/autoliuweijie/FastBERT).


## Install

Install ``fastbert`` with ``pip``.
```sh
$ pip install fastbert
```

## Single sentence classification

An example of single sentence classification are shown in [single_sentence_classification](examples/single_sentence_classification/).

```python
from fastbert import FastBERT

# Loading your dataset
labels = ['T', 'F']
sents_train = [
    'Do you like FastBERT?',
    'Yes, it runs faster than BERT!',
    ...
]
labels_train = [
    'T',
    'F',
    ...
]

# Creating and training model
model = FastBERT(
    kernel_name="google_bert_base_en",  # "google_bert_base_zh" for Chinese
    labels=labels,
    device='cuda:0'
)

model.fit(
    sents_train,
    labels_train,
    model_saving_path='./fastbert.bin',
)

# Loading model and making inference
model.load_model('./fastbert.bin')
label, exec_layers = model('I like FastBERT', speed=0.7)
```


## Two sentence flassification

```python

wait...
```


