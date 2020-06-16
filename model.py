import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from config import *

class bilstm_att(torch.nn.Module):
    def __init__(self, EMBED_SIZE, RELCNT, initW):
        super(bilstm_att, self).__init__()

        self.train_times = 0
        self.use_cuda = torch.cuda.is_available() and CUDA_ENABLE
        self.batch = BATCH_SIZE
        self.tag_size = RELCNT
        self.embed_size = EMBED_SIZE
        self.hidden_dim = HIDDEN_DIM
        self.pos_size = MAX_LEN * 2
        self.embed_dim = EMBED_DIM
        self.dropout = DROPOUT
        self.layer_size = NUM_LAYERS

        self.embed_dropout = nn.Dropout(p=EMBED_DROPOUT)
        self.lstm_out_dropout = nn.Dropout(p=LSTM_OUT_DROPOUT)
        self.att_out_dropout = nn.Dropout(p=ATT_OUT_DROPOUT)
        self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(initW), freeze = False)
        self.pos1_embeds = nn.Embedding(self.pos_size, self.embed_dim)
        self.pos1_embeds.weight.data.uniform_(-0.25, 0.25)
        self.pos1_embeds.weight.requires_grad = True
        self.pos2_embeds = nn.Embedding(self.pos_size, self.embed_dim)
        self.pos2_embeds.weight.data.uniform_(-0.25, 0.25)
        self.pos2_embeds.weight.requires_grad = True
        self.lstm = nn.LSTM(input_size = self.embed_dim, hidden_size = self.hidden_dim, num_layers = self.layer_size, dropout = self.dropout, bidirectional = True)

        if self.use_cuda:
            self.lstm = self.lstm.cuda()

            self.word_embeds = self.word_embeds.cuda()
            self.pos1_embeds = self.pos1_embeds.cuda()
            self.pos2_embeds = self.pos2_embeds.cuda()

            self.att_weight = nn.Parameter(torch.randn(1, self.hidden_dim * 2).cuda())
            self.relation_bias = nn.Parameter(torch.zeros(self.tag_size, 1).cuda())
            self.relation = nn.Parameter(torch.randn(self.tag_size, self.hidden_dim * 2).cuda())
        else:
            self.att_weight = nn.Parameter(torch.randn(1, self.hidden_dim * 2))
            self.relation_bias = nn.Parameter(torch.zeros(self.tag_size, 1))
            self.relation = nn.Parameter(torch.randn(self.tag_size, self.hidden_dim * 2))

    def attention(self, H):     # lstm_out  <batch, hid_dim*2, max_len>
        M = torch.tanh(H)       # <batch, hid_dim*2, max_len>
        a = torch.softmax(torch.bmm(self.att_weight.repeat(self.batch, 1, 1), M), 2) # <batch, 1, max_len>
        a = torch.transpose(a, 1, 2)    # <batch, max_len, 1>
        return torch.bmm(H, a)  # <batch, hid_dim*2, 1>

    def forward(self, sentence, pos1, pos2):    # <batch, max_len>
        embeds = self.word_embeds(sentence) + self.pos1_embeds(pos1) + self.pos2_embeds(pos2) # <batch, max_len, embed_dim>
        embeds = self.embed_dropout(embeds)
        embeds = torch.transpose(embeds, 0, 1)  # <max_len, batch, embed_dim>

        lstm_out, _ = self.lstm(embeds)    # <max_len, batch, hid_dim*2>, _ : (h_t, c_t)
        lstm_out = self.lstm_out_dropout(lstm_out)
        lstm_out = lstm_out.permute(1, 2, 0)    # <batch, hid_dim*2, max_len>
        att_out = self.attention(lstm_out)                      # <batch, hid_dim*2, 1>
        att_out = self.att_out_dropout(att_out)

        res = torch.add(torch.bmm(self.relation.repeat(self.batch, 1, 1), att_out), self.relation_bias.repeat(self.batch, 1, 1))   # <batch, tag_size, 1>
        res = torch.softmax(res, 1) # <batch, tag_size, 1>
        return res.view(self.batch, -1) # <batch, tag_size>

    def pred(self, sentence, pos1, pos2):
        embeds = self.word_embeds(sentence) + self.pos1_embeds(pos1) + self.pos2_embeds(pos2)
        embeds = torch.transpose(embeds, 0, 1)

        H, _ = self.lstm(embeds)
        H = H.permute(1, 2, 0)
        M = torch.tanh(H)
        a = torch.softmax(torch.bmm(self.att_weight.repeat(1, 1, 1), M), 2) # <1, max_len>
        a = a.permute(0, 2, 1)     # <max_len, 1>
        att_out = torch.bmm(H, a)   # <hid_dim*2, 1>

        res = torch.add(torch.bmm(self.relation.repeat(1, 1, 1), att_out), self.relation_bias.repeat(1, 1, 1))   # <tag_size, 1>
        res = torch.softmax(res, 1) # <tag_size, 1>
        res = res.view(self.tag_size)
        
        y = torch.argmax(res)
        return y.item()