# FastBERT-pypi

The pypi version of [FastBERT](https://github.com/autoliuweijie/FastBERT).


## Install

Install ``fastbert`` with ``pip``.
```sh
$ pip install fastbert
```


## Supported Models

FastBERT-pypi is supported by the [UER](https://github.com/dbiir/UER-py) project, and all of UER high-quality models can be accelerated in the FastBERT way.

``FastBERT`` object supports the following models:

|Models (kernel_name)  |URL                               |Description                                               |
|----------------------|----------------------------------|----------------------------------------------------------|
|google_bert_base_en   |https://share.weiyun.com/fpdOtcmz | Google pretrained English BERT-base model on Wiki corpus.|
|google_bert_base_zh   |https://share.weiyun.com/AykBph9V | Google pretrained Chinese BERT-base model on Wiki corpus.|
|uer_bert_large_zh     |https://share.weiyun.com/chx2VhGk | UER pretrained Chinese BERT-large model on mixed corpus. |
|uer_bert_small_zh     |https://share.weiyun.com/wZuVBM5g | UER pretrained Chinese BERT-small model on mixed corpus. |
|uer_bert_tiny_zh      |https://share.weiyun.com/VJ3JEN9Z | UER pretrained Chinese BERT-tiny model on mixed corpus.  |


In fact, you don't have to download the model yourself. FastBERT will download the corresponding model file automatically at the first time you use it. If the automatically downloading failed, you can download these model files from the above URLs, and saving them to the directory of "~/.fastbert/".


## Quick Start

### Single sentence classification

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

# Creating a model
model = FastBERT(
    kernel_name="google_bert_base_en",  # "google_bert_base_zh" for Chinese
    labels=labels,
    device='cuda:0'
)

# Training the model
model.fit(
    sents_train,
    labels_train,
    model_saving_path='./fastbert.bin',
)

# Loading the model and making inference
model.load_model('./fastbert.bin')
label, exec_layers = model('I like FastBERT', speed=0.7)
```


### Two sentences classification

An example of two sentences classification are presented in [two_sentences_classification](examples/two_sentences_classification/).

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

# Creating a model
model = FastBERT_S2(
    kernel_name="google_bert_base_zh",  # "google_bert_base_en" for English
    labels=labels,
    device='cuda:0'
)

# Training the model
model.fit(
    sents_a_train=questions_train,
    sents_b_train=answers_train,
    labels_train=labels_train,
    model_saving_path='./fastbert.bin',
)

# Loading the model and making inference
model.load_model('./fastbert.bin')
label, exec_layers = model(
    sent_a='我也要用FastBERT!',
    sent_b='来，吃老干妈!',
    speed=0.7)
```


## Usage

Args of ``FastBERT``/``FastBERT_S2``:

|Args|Type|Examples|Explanation|
|----|----|--------|-----------|
|kernel_name|str|'google_bert_base_en'|The name of the kernel model, including 'google_bert_base_en', 'google_bert_base_zh'.|
|labels|list|['T', 'F']| A list of all labels.|
|seq_length (optional)|int|256| The sentence length for FastBERT. Default 128|
|device (optional)|str|'cuda:0'| The device for runing FastBERT, default 'cpu'|

Args of ``FastBERT.fit()``:

|Args|Type|Examples|Explanation|
|----|----|--------|-----------|
|sentences_train |list|['sent 1', 'sent 2',...] | A list of training sentences.|
|labels_train | list |['T', 'F', ...] | A list of training labels.|
|batch_size (optional)| int | 32| batch_size for training. Default 16|
|sentences_dev (optional)| list | [] | A list of validation sentences.|
|labels_dev (optional) | list | [] | A list of validation labels.|
|learning_rate (optional) | float | 2e-5 | learning rate.|
|finetuning_epochs_num (optional) | int | 5 | The epoch number of finetuning.|
|distilling_epochs_num (optional) | int | 10| The epoch number of distilling.|
|report_steps (optional) | int |100 | Report the training process every [report_steps] steps.|
|warmup (optional) | float | 0.1 |The warmup rate for training.|
|dev_speed (optional) | float | 0.5 | The speed for evaluating in the self-distilling process.|
|model_saving_path (optional) | str | './model.bin' | The path to saving model. |

Args of ``FastBERT.forward()``:

|Args|Type|Examples|Explanation|
|----|----|--------|-----------|
|sentence | str | 'How are you' | The input sentence.|
|speed (optional) | float | 0.5 | The speed value for inference. Default 0.0.|

Args of ``FastBERT_S2.fit()``:

|Args|Type|Examples|Explanation|
|----|----|--------|-----------|
|sents_a_train |list|['sent a 1', 'sent a 2',...] | A list of training A-sentences.|
|sents_b_train |list|['sent b 1', 'sent b 2',...] | A list of training B-sentences.|
|labels_train | list |['T', 'F', ...] | A list of training labels.|
|batch_size (optional)| int | 32| batch_size for training. Default 16|
|sents_a_dev (optional)| list | [] | A list of validation A-sentences.|
|sents_b_dev (optional)| list | [] | A list of validation B-sentences.|
|labels_dev (optional) | list | [] | A list of validation labels.|
|learning_rate (optional) | float | 2e-5 | learning rate.|
|finetuning_epochs_num (optional) | int | 5 | The epoch number of finetuning.|
|distilling_epochs_num (optional) | int | 10| The epoch number of distilling.|
|report_steps (optional) | int |100 | Report the training process every [report_steps] steps.|
|warmup (optional) | float | 0.1 |The warmup rate for training.|
|dev_speed (optional) | float | 0.5 | The speed for evaluating in the self-distilling process.|
|model_saving_path (optional) | str | './model.bin' | The path to saving model. |

Args of ``FastBERT_S2.forward()``:

|Args|Type|Examples|Explanation|
|----|----|--------|-----------|
|sents_a | str | 'How are you' | The input A-sentence.|
|sents_b | str | 'How are you' | The input B-sentence.|
|speed (optional) | float | 0.5 | The speed value for inference. Default 0.0.|


