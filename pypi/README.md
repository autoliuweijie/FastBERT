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


## Two sentences classification

```python
from fastbert import FastBERT_S2

# Loading your dataset
labels = ['T', 'F']
questions_train = [
    'FastBERT快吗?',
    '你在业务里使用FastBERT了吗?',
    ...
]
answers_train = [
    '快！而且速度还可调.',
    '用了啊，帮我省了好几百台机器.',
    ...
]
labels_train = [
    'T',
    'T',
    ...
]

# Creating and training model
model = FastBERT_S2(
    kernel_name="google_bert_base_zh",  # "google_bert_base_en" for English
    labels=labels,
    device='cuda:0'
)

model.fit(
    sents_a_train=questions_train,
    sents_b_train=answers_train,
    labels_train=labels_train,
    model_saving_path='./fastbert.bin',
)

# Loading model and making inference
model.load_model('./fastbert.bin')
label, exec_layers = model(
    sent_a='我也要用FastBERT!',
    sent_b='来，吃老干妈!',
    speed=0.7)
```


