import sys
import re
import torch
import numpy as np
import pickle
from collections.abc import Iterable
from collections import deque
from config import *

CUDA = torch.cuda.is_available() and CUDA_ENABLE
id2relation = [
    "Other",
    "Cause-Effect",
    "Instrument-Agency",
    "Product-Producer",
    "Content-Container",
    "Entity-Origin",
    "Entity-Destination",
    "Component-Whole",
    "Member-Collection",
    "Message-Topic"
]
with open(TRAIN_B, 'rb') as inp:
    word2id = pickle.load(inp)
    unkown_id = pickle.load(inp) - 1
max_len = MAX_LEN
model = torch.load(MODEL_B)
model.eval()

def flatten(x):
    result = []
    for el in x:
        if isinstance(x, Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def fe1e2(x):     # 找到e1e2
    e1 = 0
    e2 = 0
    words = []
    i = 0
    for el in x:
        if len(el) > 3:
            en = el[0:3]
            if en == "E11":
                e1 = i
                words.append(el[3:])
                i += 1
                continue
            if en == "E21":
                e2 = i
                words.append(el[3:])
                i += 1
                continue
        i += 1
        words.append(el)
    if i > max_len:
        i = max_len
    return e1, e2, words, i

def X_padding(words):
    ids = []
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(unkown_id)

    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids

def pos_padding(index, l):
    ids=[]
    if index < max_len:
        for i in range(index):
            ids.append(index - i)
        for i in range(l - index):
            ids.append(i + max_len)
        if l < max_len:
            ids.extend([0] * (max_len -l))
    else:
        for i in range(max_len):
            if index - i < max_len:
                ids.append(index - i)
            else:
                ids.append(max_len - 1)
    return ids

def load_line(lines):

    for line in lines:
        if len(line) == 0:
            continue
        text = line.lower().replace("<e1>","E11").replace("</e1>","").replace("<e2>","E21").replace("</e2>","")
        text = re.sub(r"[^A-Za-z0-9'-]", " ", text)
        text = re.sub(r"what's ", "what is ", text)
        text = re.sub(r"that's ", "that is ", text)
        text = re.sub(r"there's ", "there is ", text)
        text = re.sub(r"it's ", "it is ", text)
        text = re.sub(r"'s ", " ", text)
        text = re.sub(r"'ve ", " have ", text)
        text = re.sub(r"can't ", "can not ", text)
        text = re.sub(r"n't ", " not ", text)
        text = re.sub(r"i'm ", "i am ", text)
        text = re.sub(r"'re ", " are ", text)
        text = re.sub(r"'d ", " would ", text)
        text = re.sub(r"'ll ", " will ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r" \d[\w-]* ", " __NUM__ ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"e-mail", "email", text)
        text = re.sub(r"-", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        e1, e2, words, i = fe1e2(text.split())
        
        sentence = torch.LongTensor([X_padding(words)])
        pos1 = torch.LongTensor([pos_padding(e1, i)])
        pos2 = torch.LongTensor([pos_padding(e2, i)])
        if CUDA:
            sentence = sentence.cuda()
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()
        y = model.pred(sentence, pos1, pos2)
        print(id2relation[y])

lines = []
line = sys.stdin.readline()
while line:
    lines.append(line.strip())
    line = sys.stdin.readline()

load_line(lines)