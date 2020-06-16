import numpy as np
import pickle
import sys


if len(sys.argv) == 1:
    print('Usage: \n    logfile [options]\nOptions: \n    [-w int] : 单条记录打印\n    [-b int] : 第几条记录开始打印\n    [-e int] : 到第几条记录\n    [-s int] : 每次间隔条数')
    exit()

with open(sys.argv[1], 'rb') as inp:
    log = pickle.load(inp)

config = log[0]
bt = config[0]
print("BEGIN_TIMES = {}; MAX_LEN = {}; BATCH_SIZE = {}; HIDDEN_DIM = {}; EMBED_DIM = {}; LR = {}; WEIGHT_DECAY = {}; NUM_LAYERS = {}; DROPOUT = {}".format(bt, config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8]))

w = 0
b = 1
e = len(log)
s = 1
if len(sys.argv) == 3:
    w = int(sys.argv[2])
else:
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '-b':
            b = int(sys.argv[i+1])
        elif sys.argv[i] == '-e':
            e = int(sys.argv[i+1])
        elif sys.argv[i] == '-s':
            s = int(sys.argv[i+1])
        elif sys.argv[i] == '-w':
            w = int(sys.argv[i+1])
            break

max_f1 = [0,0]

def display(i):
    if i>= len(log):
        print("没有该条记录", i)
        exit()
    else:
        logitem = log[i]
        if logitem[2] > max_f1[0]:
            max_f1[0] = logitem[2]
            max_f1[1] = i
        print("t = {}; acc = {:.4f}; loss = {:.4f}; f1 = {:.4f};".format(bt + i, logitem[0], logitem[1], logitem[2]))

if w:
    display(w)
else:
    for i in range(b, e, s):
        display(i)

# ... 
if max_f1:
    print("第{}次f1最大，f1={}".format(bt + max_f1[1], max_f1[0]))