## 行数据处理参数
MAX_LEN = 70
## 训练参数
BATCH_SIZE = 20
EPOCHS = 400
EPOCH_SAVE = 100
PRF = 'macroB'
## 维度信息 本版本没有POS_DIM
HIDDEN_DIM = 50
EMBED_DIM = 100
## 优化参数
OPTI = 'delta'
LR = 1.0
WEIGHT_DECAY = 1e-05
## 网络参数
NUM_LAYERS = 2
DROPOUT = 0.2
EMBED_DROPOUT = 0.2
LSTM_OUT_DROPOUT = 0.2
ATT_OUT_DROPOUT = 0.2
## CUDA
CUDA_ENABLE = True
## 预训练
PRE_TRAIN = './glove.6B.100d.txt'
## 文本处理
TRAIN_B = './train.b'
TEST_B = './test.b'
## 预测
MODEL_B = './model'
