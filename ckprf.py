import numpy as np
import pickle
import sys


if len(sys.argv) == 1:
    print('请输入prf文件名')
    exit()

with open(sys.argv[1], 'rb') as inp:
    prt = pickle.load(inp)

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

RELCNT = 10

def get_f1_micro(item):
    count_predict = item[0]  # 猜的
    count_right = item[1]    # 猜对的
    count_total = item[2]    # 实际的
    P = 0
    R = 0
    F = 0
    sum_r = sum(count_right)
    sum_p = sum(count_predict)
    sum_t = sum(count_total)
    if sum_p:
        P = sum(count_right) / sum(count_predict) * 100
    if sum_t:
        R = sum(count_right) / sum(count_total) * 100
    if P + R:
        F = (2*P*R)/(P+R)
    return P, R, F

def get_f1(item):
    count_predict = item[0]  # 猜的
    count_right = item[1]    # 猜对的
    count_total = item[2]    # 实际的
    precision = [0]*RELCNT      # 猜的里面有多少对的
    recall = [0]*RELCNT         # 有多少的被猜出来
    f1 = [0]*RELCNT
    for i in range(RELCNT):
        if count_predict[i] != 0:
            precision[i] = float(count_right[i]) / count_predict[i] * 100
        if count_total[i] != 0:
            recall[i] = float(count_right[i]) / count_total[i]  * 100
        if precision[i] + recall[i] != 0:
            f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
    return precision, recall, f1

w = len(prt) - 1
if len(sys.argv) == 3:
    v = sys.argv[2]
    if 97 <= ord(v[0]) <= 122:
            v = chr(ord(v[0]) - 32)
    if v == 'C':
        config = prt[0]
        print("BEGIN_TIMES = {}; MAX_LEN = {}; BATCH_SIZE = {}; HIDDEN_DIM = {}; EMBED_DIM = {}; LR = {}; WEIGHT_DECAY = {}; NUM_LAYERS = {}; DROPOUT = {}".format(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8]))
        exit()
    else:
        w = int(v)


def display(i):
    if i>= len(prt):
        print("没有该条记录")
        exit()
    else:
        item = prt[i]
        precision, recall, f1 = get_f1(item)
        print ("{:20s} | {:5s} | {:5s} | {:5s}".format("RELATION", "P(%)", "R(%)", "F1(%)"))
        for i in range(RELCNT):
            print("{:20s} | {:>5.2f} | {:>5.2f} | {:>5.2f}".format(id2relation[i], precision[i], recall[i], f1[i]))
        print ("{:20s} | {:>5.2f} | {:>5.2f} | {:>5.2f}".format("AVERAGE - macro", sum(precision) / RELCNT, sum(recall) / RELCNT, sum(f1) / RELCNT))
        P, R, F = get_f1_micro(item)
        print ("{:20s} | {:>5.2f} | {:>5.2f} | {:>5.2f}".format("AVERAGE - micro", P, R, F))

display(w)