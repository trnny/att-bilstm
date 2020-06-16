import codecs
import sys
import re
import pandas as pd
import numpy as np
from collections.abc import Iterable
from collections import deque
from config import *

def flatten(x):
    result = []
    for el in x:
        if isinstance(x, Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

max_len = MAX_LEN

def fe1e2(x):     # 找到e1e2
    e1 = 0
    e2 = 0
    result = []
    i = 0
    for el in x:
        if len(el) > 3:
            en = el[0:3]
            if en == "E11":
                e1 = i
                result.append(el[3:])
                i += 1
                continue
            if en == "E21":
                e2 = i
                result.append(el[3:])
                i += 1
                continue
        i += 1
        result.append(el)
    if i > max_len:
        i = max_len
    return e1, e2, result, i    # i 是长度

relation2id = {
    "Other": 0,
    "Cause-Effect": 1,
    "Instrument-Agency":2,
    "Product-Producer":3,
    "Content-Container":4,
    "Entity-Origin":5,
    "Entity-Destination":6,
    "Component-Whole":7,
    "Member-Collection":8,
    "Message-Topic":9
}

def load_data_and_labels(path):
    datas = deque()     # 每行为单词的数组
    labels = deque()    # 标签的数组(数字)
    entity1 = deque()   # 实体1的索引
    entity2 = deque()   # 实体2的索引
    lens = deque()      # 每行单词个数

    lines = [line.strip() for line in open(path)]
    for idx in range(0, len(lines), 4):
        text = lines[idx].split('\t')[1][1:-1].lower().replace("<e1>","E11").replace("</e1>","").replace("<e2>","E21").replace("</e2>","")
        text = re.sub(r"[^A-Za-z0-9'-]", " ", text)
        text = re.sub(r"what's ", "what is ", text)
        text = re.sub(r"that's ", "that is ", text)
        text = re.sub(r"there's ", "there is ", text)
        text = re.sub(r"it's ", "it is ", text)
        text = re.sub(r"'s ", " ", text)
        text = re.sub(r"'ve ", " have ", text)
        text = re.sub(r"can't ", "can not ", text)
        text = re.sub(r"n't ", " not ", text)   # does, did...
        text = re.sub(r"i'm ", "i am ", text)
        text = re.sub(r"'re ", " are ", text)
        text = re.sub(r"'d ", " would ", text)
        text = re.sub(r"'ll ", " will ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r" \d[\w-]* ", " __NUM__ ", text)
        text = re.sub(r" e g ", " eg ", text)   # .已经删掉了
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"e-mail", "email", text)
        text = re.sub(r"-", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        e1, e2, text, i = fe1e2(text.split())
        entity1.append(e1)
        entity2.append(e2)
        datas.append(text)
        lens.append(i)

        if lines[idx + 1] == "Other":
            labels.append(0)
        else:
            labels.append(relation2id[lines[idx + 1].split('(')[0]])
    return datas, labels, entity1, entity2, lens

datas, labels, entity1, entity2, lens = load_data_and_labels('TRAIN_FILE.TXT')

set_words = pd.Series(flatten(datas)).value_counts().index   # 编号
word2id = pd.Series(range(1, len(set_words) + 1), index = set_words)
unkown_id = len(word2id) + 1
word2id['__UNKOWN__'] = unkown_id

def X_padding(words):
    ids = []
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(unkown_id)

    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len-len(ids)))
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


x = deque()
pos_e1 = deque()
pos_e2 = deque()
for index in range(len(datas)):     # 8000
    x.append(X_padding(datas[index]))
    pos_e1.append(pos_padding(entity1[index], lens[index]))
    pos_e2.append(pos_padding(entity2[index], lens[index]))

x = np.asarray(x)               #<int,int>
y = np.asarray(labels)          #<int>
pos_e1 = np.asarray(pos_e1)     #<int,int>
pos_e2 = np.asarray(pos_e2)     #<int,int>

import pickle
with open(TRAIN_B, 'wb') as outp:
    pickle.dump(word2id, outp)
    pickle.dump(unkown_id + 1, outp)
    pickle.dump(len(relation2id), outp)
    pickle.dump(x, outp)
    pickle.dump(y, outp)
    pickle.dump(pos_e1, outp)
    pickle.dump(pos_e2, outp)
print('** Finished saving train data.')

datas, labels, entity1, entity2, lens = load_data_and_labels('TEST_FILE_FULL.TXT')
x = deque()
pos_e1 = deque()
pos_e2 = deque()
for index in range(len(datas)):
    x.append(X_padding(datas[index]))
    pos_e1.append(pos_padding(entity1[index], lens[index]))
    pos_e2.append(pos_padding(entity2[index], lens[index]))
    
x = np.asarray(x)
y = np.asarray(labels)
pos_e1 = np.asarray(pos_e1)
pos_e2 = np.asarray(pos_e2)

import pickle
with open(TEST_B, 'wb') as outp:
    pickle.dump(x, outp)
    pickle.dump(y, outp)
    pickle.dump(pos_e1, outp)
    pickle.dump(pos_e2, outp)
print('** Finished saving test data.')
