# FastBERT_pypi

The pypi version of [FastBERT](https://github.com/autoliuweijie/FastBERT).


## Install

Install ``fastbert`` with ``pip``.
```sh
$ pip install fastbert
```

## Usage

Currently ``fastbert`` only supports single sentence classification. More function will be added later.

### Chinese single sentence classification

```python
from fastbert import FastBERT

# Loading your dataset
labels = ['T', 'F']
sents_train = [
    '你吃北京烤鸭吗?',
    '我吃宫爆鸡丁!',
    ...
]
labels_train = [
    'T',
    'F',
    ...
]

# Create and training model
model = FastBERT("google_bert_base_zh", labels=labels, device='cuda:0')
model.fit(
    sents_train,
    labels_train,
    model_saving_path='./fastbert.bin',
)

# Loading model and make inference
model.load_model('./fastbert.bin')
label, exec_layers = model('还是吃老干妈吧', speed=0.7)
```

### English single sentence classification

```python

...

# Create and training model
model = FastBERT("google_bert_base_en", labels=labels, device='cuda:0')
model.fit(
    sents_train,
    labels_train,
    model_saving_path='./fastbert.bin',
)

...

```
