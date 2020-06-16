import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from torch.autograd import Variable
from collections import deque
from model import bilstm_att
from config import *

torch.manual_seed(1)
CUDA = torch.cuda.is_available() and CUDA_ENABLE

print("config has been loaded...\nMAX_LEN = {}; BATCH_SIZE = {}; EPOCH_SAVE = {}; HIDDEN_DIM = {}; EMBED_DIM = {}; LR = {}; WEIGHT_DECAY = {}; NUM_LAYERS = {}; DROPOUT = {}; CUDA = {}".format(MAX_LEN, BATCH_SIZE, EPOCH_SAVE, HIDDEN_DIM, EMBED_DIM, LR, WEIGHT_DECAY, NUM_LAYERS, DROPOUT, CUDA))

print("train datas are loading...")
with open(TRAIN_B, 'rb') as inp:
    word2id = pickle.load(inp)      # <str,int>
    EMBED_SIZE = pickle.load(inp)   # int
    RELCNT = pickle.load(inp)       # int
    train = pickle.load(inp)        # <int,int>
    labels = pickle.load(inp)       # <int>
    position1 = pickle.load(inp)    # <int,int>
    position2 = pickle.load(inp)    # <int,int>
print("test datas are loading...")
with open(TEST_B, 'rb') as inp:
    test = pickle.load(inp)
    labels_t = pickle.load(inp)
    position1_t = pickle.load(inp)
    position2_t = pickle.load(inp)

if len(sys.argv) >= 2:
    model = torch.load(sys.argv[1])
    if len(sys.argv) >= 3:
        EPOCHS = int(sys.argv[2])
else:
    initW = torch.zeros(EMBED_SIZE, EMBED_DIM)
    # 读预训练
    if PRE_TRAIN != None:
        print("Load glove file", PRE_TRAIN)
        f = open(PRE_TRAIN, 'r', encoding = 'utf8')
        for line in f:
            splitLine = line.split(' ')
            word = splitLine[0]
            embedding = torch.from_numpy(np.asarray(splitLine[1:], dtype='float32'))
            if word in word2id:
                idx = word2id[word]
                initW[idx] = embedding
    model = bilstm_att(EMBED_SIZE, RELCNT, initW)
    if CUDA:
        model = model.cuda()
if OPTI == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY, amsgrad=True)
else:
    optimizer = optim.Adadelta(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(reduction='sum')    # 交叉熵

train = torch.LongTensor(train)
position1 = torch.LongTensor(position1)
position2 = torch.LongTensor(position2)
labels = torch.LongTensor(labels)
if CUDA:
    train = train.cuda()
    position1 = position1.cuda()
    position2 = position2.cuda()
    labels = labels.cuda()
train_datasets = D.TensorDataset(train, position1, position2, labels)
train_dataloader = D.DataLoader(train_datasets, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0, drop_last=True)

test = torch.LongTensor(test)
position1_t = torch.LongTensor(position1_t)
position2_t = torch.LongTensor(position2_t)
labels_t = torch.LongTensor(labels_t)
if CUDA:
    test = test.cuda()
    position1_t = position1_t.cuda()
    position2_t = position2_t.cuda()
    labels_t = labels_t.cuda()
test_datasets = D.TensorDataset(test, position1_t, position2_t, labels_t)
test_dataloader = D.DataLoader(test_datasets, batch_size = BATCH_SIZE, num_workers=0, drop_last=True)

tlog = [model.train_times,MAX_LEN,BATCH_SIZE,HIDDEN_DIM,EMBED_DIM,LR,WEIGHT_DECAY,NUM_LAYERS,DROPOUT]
log = [tlog]        ## 每次训练的记录
count_prt = [tlog]    ## 记录每次训练的prf

def train_once():
    model.train()
    acc = 0
    total = 0
    global tlog
    tlog = [0,0,0]  # train loss f1
    for sentence, pos1, pos2, tag in train_dataloader: # # <batch, max_len> 3个 <batch> 1个
        optimizer.zero_grad()
        y = model(sentence, pos1, pos2) # <batch, RELCNT>
        loss = criterion(y, tag)
        loss.backward()
        optimizer.step()
        tlog[1] += loss.item()
        if CUDA:
            y = np.argmax(y.data.cpu().numpy(), axis = 1)
        else:
            y = np.argmax(y.data.numpy(), axis = 1)
        for y1, y2 in zip(y, tag):
            if y1 == y2:
                acc += 1
            total += 1
    model.train_times += 1
    tlog[0] = 100 * float(acc) / total
    print("train: {:.4f}%, loss: {:.4f}".format(tlog[0], tlog[1]))

def test_once():
    model.eval()
    count_predict = [0]*RELCNT
    count_right = [0]*RELCNT
    count_total = [0]*RELCNT
    for sentence, pos1, pos2, tag in test_dataloader:
        y = model(sentence, pos1, pos2)
        if CUDA:
            y = np.argmax(y.data.cpu().numpy(), axis=1)
        else:
            y = np.argmax(y.data.numpy(), axis = 1)
        for y1, y2 in zip(y, tag):
            count_predict[y1] += 1
            count_total[y2] += 1
            if y1 == y2:
                count_right[y1] += 1
    count_prt.append([count_predict, count_right, count_total])     # 预测数、正确数、总数
    P = 0
    R = 0
    F = 0
    if PRF == 'micro':  # PRF micro
        P = sum(count_right) / sum(count_predict)
        R = sum(count_right) / sum(count_total)
        if P + R != 0:
            F = (2 * P * R) / ( P + R)
    else:   # PRF macro
        precision = [0]*RELCNT
        recall = [0]*RELCNT
        f1 = [0]*RELCNT
        for i in range(RELCNT):
            if count_predict[i] != 0:
                precision[i] = float(count_right[i]) / count_predict[i]
            if count_total[i] != 0:
                recall[i] = float(count_right[i]) / count_total[i]
            if precision[i] + recall[i] != 0:
                f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
        P = sum(precision) / RELCNT
        R = sum(recall) / RELCNT
        if PRF == 'macroA':
            if P + R != 0:
                F = (2 * P * R) / (P + R)
        else:
            F = sum(f1) / RELCNT
    tlog[2] = F
    log.append(tlog)
    print("P: {:.4f}, R: {:.4f}, F1: {:.4f}\n".format(P, R, F))

try:
    for epoch in range(1, EPOCHS+1):
        print("EPOCH: [{}/{}], TOTAL: {}".format(epoch, EPOCHS, model.train_times + 1))
        train_once()
        test_once()
        if epoch % EPOCH_SAVE == 0:
            torch.save(model, "./temp/model_" + str(model.train_times))
except KeyboardInterrupt:
    print("canceled, trained", model.train_times, "times.")

torch.save(model, "model")
print ("model has been saved")
with open('log/log.'+str(model.train_times), 'wb') as outp:
    pickle.dump(log, outp)
with open('log/prt2prf.'+str(model.train_times), 'wb') as outp:
    pickle.dump(count_prt, outp)
print('** Finished saving log.')
