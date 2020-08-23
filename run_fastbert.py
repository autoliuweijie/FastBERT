# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to the fine-tuning and self-distillation 
  peocess of the FastBERT.
"""
import os, sys
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.model_loader import load_model
from uer.layers.multi_headed_attn import MultiHeadedAttention
import numpy as np
import time
from thop import profile


torch.set_num_threads(1)
#use the KLDivLoss
#class LabelSmoothingLoss(nn.Module):
#    def __init__(self, classes, smoothing=0.0, dim=-1):
#        super(LabelSmoothing, self).__init__()
#        self.criterion = nn.KLDivLoss(reduction='batchmean')
#        self.confidence = 1.0 - smoothing
#        self.smoothing = smoothing
#        self.classes = classes
#    def forward(self, pred, target):
#        pred = nn.LogSoftmax(pred)
#        with torch.no_grad():
#            true_dist = torch.zeros_like(pred)
#            true_dist.fill_(self.smoothing / (self.classes - 1))
#            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#        return self.criterion(pred, true_dist)

#use the CrossEntropyLoss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def normal_shannon_entropy(p, labels_num):
    entropy = torch.distributions.Categorical(probs=p).entropy()
    normal = -np.log(1.0/labels_num)
    return entropy / normal


class Classifier(nn.Module):

    def __init__(self, args, input_size, labels_num):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.cla_hidden_size = 128
        self.cla_heads_num = 2
        self.labels_num = labels_num
        self.pooling = args.pooling
        self.output_layer_0 = nn.Linear(input_size, self.cla_hidden_size)
        self.self_atten = MultiHeadedAttention(self.cla_hidden_size, self.cla_heads_num, args.dropout)
        self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.cla_hidden_size)
        self.output_layer_2 = nn.Linear(self.cla_hidden_size, labels_num)
    
    def forward(self, hidden, mask):

        hidden = torch.tanh(self.output_layer_0(hidden))
        hidden = self.self_atten(hidden, hidden, hidden, mask)
        
        if self.pooling == "mean":
            hidden = torch.mean(hidden, dim=-1)
        elif self.pooling == "max":
            hidden = torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            hidden = hidden[:, -1, :]
        else:
            hidden = hidden[:, 0, :]

        output_1 = torch.tanh(self.output_layer_1(hidden))
        logits = self.output_layer_2(output_1)
        return logits


class FastBertClassifier(nn.Module):
    def __init__(self, args, model):
        super(FastBertClassifier, self).__init__()

        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.classifiers = nn.ModuleList([
                Classifier(args, args.hidden_size, self.labels_num) \
                for i in range(self.encoder.layers_num)
             ])
        self.softmax = nn.LogSoftmax(dim=-1)
        #self.criterion = nn.NLLLoss()
        self.criterion = LabelSmoothingLoss(self.labels_num, args.labelsmoothing, -1)
        self.soft_criterion = nn.KLDivLoss(reduction='batchmean')
        self.threshold = args.speed

    def forward(self, src, label, mask, fast=True):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask)

        # Encoder.
        seq_length = emb.size(1)
        mask = (mask > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
         
        if self.training:

            if label is not None:

                # training main part of the model
                hidden = emb
                for i in range(self.encoder.layers_num):
                    hidden = self.encoder.transformer[i](hidden, mask)
                logits = self.classifiers[-1](hidden, mask)
                loss = self.criterion(logits.view(-1, self.labels_num), label.view(-1))
                return loss, logits
            else:
                # distillate the subclassifiers
                loss, hidden, hidden_list = 0, emb, []
                with torch.no_grad():
                    for i in range(self.encoder.layers_num):
                        hidden = self.encoder.transformer[i](hidden, mask)
                        hidden_list.append(hidden)
                    teacher_logits = self.classifiers[-1](hidden_list[-1], mask).view(-1, self.labels_num) 
                teacher_probs = nn.functional.softmax(teacher_logits, dim=1)
                loss = 0
                for i in range(self.encoder.layers_num - 1):
                    student_logits = self.classifiers[i](hidden_list[i], mask).view(-1, self.labels_num)
                    loss += self.soft_criterion(self.softmax(student_logits), teacher_probs) 
                return loss, teacher_logits

        else:
            # inference 
            if fast:
                # fast mode 
                hidden = emb  # (batch_size, seq_len, emb_size)
                batch_size = hidden.size(0)
                logits = torch.zeros(batch_size, self.labels_num, dtype=hidden.dtype, device=hidden.device)
                abs_diff_idxs = torch.arange(0, batch_size, dtype=torch.long, device=hidden.device)
                for i in range(self.encoder.layers_num):
                    
                    hidden = self.encoder.transformer[i](hidden, mask)

                    logits_this_layer = self.classifiers[i](hidden, mask)  # (batch_size, labels_num)
                    logits[abs_diff_idxs] = logits_this_layer

                    # filter easy sample
                    abs_diff_idxs, rel_diff_idxs = self._difficult_samples_idxs(abs_diff_idxs, logits_this_layer) 
                    hidden = hidden[rel_diff_idxs, :, :]
                    mask = mask[rel_diff_idxs, :, :]
                    
                    if len(abs_diff_idxs) == 0:
                        break

                return None, logits
            else:
                # normal mode
                hidden = emb
                for i in range(self.encoder.layers_num):
                    hidden = self.encoder.transformer[i](hidden, mask)
                logits = self.classifiers[-1](hidden, mask)
                return None, logits
                    
    def _difficult_samples_idxs(self, idxs, logits):
        # logits: (batch_size, labels_num)
        probs = nn.Softmax(dim=1)(logits)
        entropys = normal_shannon_entropy(probs, self.labels_num)
        # torch.nonzero() is very time-consuming on GPU 
        # Please see https://github.com/pytorch/pytorch/issues/14848
        # If anyone can optimize this operation, please contact me, thank you!
        rel_diff_idxs = (entropys > self.threshold).nonzero().view(-1)
        abs_diff_idxs = torch.tensor([idxs[i] for i in rel_diff_idxs], device=logits.device)
        return abs_diff_idxs, rel_diff_idxs
        
        
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/fastbert.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    # labelsmoothing
    parser.add_argument("--labelsmoothing", type=float, default=0.0, help="the value of label smoothing")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--distill_epochs_num", type=int, default=10,
                        help="Number of distillation epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")
    parser.add_argument("--fast_mode", dest='fast_mode', action='store_true', help="Whether turn on fast mode")
    parser.add_argument("--speed", type=float, default=0.5, help="Threshold of Uncertainty, i.e., the Speed in paper.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set) 

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build classification model.
    model = FastBertClassifier(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Read dataset.
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                try:
                    line = line.strip().split('\t')
                    if len(line) == 2:
                        label = int(line[columns["label"]])
                        text = line[columns["text_a"]]
                        tokens = [vocab.get(t) for t in tokenizer.tokenize(text)]
                        tokens = [CLS_ID] + tokens
                        mask = [1] * len(tokens)
                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask))
                    elif len(line) == 3: # For sentence pair input.
                        label = int(line[columns["label"]])
                        text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]

                        tokens_a = [vocab.get(t) for t in tokenizer.tokenize(text_a)]
                        tokens_a = [CLS_ID] + tokens_a + [SEP_ID]
                        tokens_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                        tokens_b = tokens_b + [SEP_ID]

                        tokens = tokens_a + tokens_b
                        mask = [1] * len(tokens_a) + [2] * len(tokens_b)
                        
                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask))
                    elif len(line) == 4: # For dbqa input.
                        qid=int(line[columns["qid"]])
                        label = int(line[columns["label"]])
                        text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]

                        tokens_a = [vocab.get(t) for t in tokenizer.tokenize(text_a)]
                        tokens_a = [CLS_ID] + tokens_a + [SEP_ID]
                        tokens_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                        tokens_b = tokens_b + [SEP_ID]

                        tokens = tokens_a + tokens_b
                        mask = [1] * len(tokens_a) + [2] * len(tokens_b)

                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask, qid))
                    else:
                        pass
                        
                except:
                    pass
        return dataset

    # Evaluation function.
    def evaluate(args, is_test, fast_mode=False):
        if is_test:
            dataset = read_dataset(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])

        batch_size = 1
        instances_num = input_ids.size()[0]

        print("The number of evaluation instances: ", instances_num)
        print("Fast mode: ", fast_mode)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()
        
        if not args.mean_reciprocal_rank:
            total_flops, model_params_num = 0, 0
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                with torch.no_grad():

                    # Get FLOPs at this batch
                    inputs = (input_ids_batch, label_ids_batch, mask_ids_batch, fast_mode)
                    flops, params = profile(model, inputs, verbose=False)
                    total_flops += flops
                    model_params_num = params
                    
                    # inference
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, fast=fast_mode)

                logits = nn.Softmax(dim=1)(logits)
                pred = torch.argmax(logits, dim=1)
                gold = label_ids_batch
                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                correct += torch.sum(pred == gold).item()

            print("Number of model parameters: {}".format(model_params_num))
            print("FLOPs per sample in average: {}".format(total_flops / float(instances_num)))
        
            if is_test:
                print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")
            for i in range(confusion.size()[0]):
                p = confusion[i,i].item()/confusion[i,:].sum().item()
                r = confusion[i,i].item()/confusion[:,i].sum().item()
                f1 = 2*p*r / (p+r)
                if is_test:
                    print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
            return correct/len(dataset)
        else:
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch)
                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all=logits
                if i >= 1:
                    logits_all=torch.cat((logits_all,logits),0)
        
            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][3]
                label = dataset[i][1]
                if qid == order:
                    j += 1
                    if label == 1:
                        gold.append((qid,j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid,j))


            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order=gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid=int(dataset[i][3])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            for i in range(len(score_list)):
                if len(label_order[i])==1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i],reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            print("Mean Reciprocal Rank: {:.4f}".format(MRR))
            return MRR

    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    input_ids = torch.LongTensor([example[0] for example in trainset])
    label_ids = torch.LongTensor([example[1] for example in trainset])
    mask_ids = torch.LongTensor([example[2] for example in trainset])

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps*args.warmup, t_total=train_steps)
   
    # traning main part of model
    print("Start fine-tuning the backbone of the model.")
    total_loss = 0.
    result = 0.0
    best_result = 0.0 
    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)

            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch)  # training
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, backbone fine-tuning steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.
            loss.backward()
            optimizer.step()
            scheduler.step()
        result = evaluate(args, False, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
        else:
            continue

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation after bakbone fine-tuning.")
        model = load_model(model, args.output_model_path)
        print("Test on normal model")
        evaluate(args, True, False)
        if args.fast_mode:
            print("Test on Fast mode")
            evaluate(args, True, args.fast_mode)

    # Distillate subclassifiers
    print("Start self-distillation for student-classifiers.")
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate*10, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps*args.warmup, t_total=train_steps)

    model = load_model(model, args.output_model_path)
    total_loss = 0.
    result = 0.0
    best_result = 0.0 
    for epoch in range(1, args.distill_epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)

            loss, _ = model(input_ids_batch, None, mask_ids_batch)  # distillation
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, self-distillation steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.
            loss.backward()
            optimizer.step()
            scheduler.step()
        result = evaluate(args, False, args.fast_mode)
        save_model(model, args.output_model_path) 

        # Evaluation phase.
        if args.test_path is not None:
            print("Test set evaluation after self-distillation.")
            model = load_model(model, args.output_model_path)
            evaluate(args, True, args.fast_mode)


if __name__ == "__main__":
    main()
