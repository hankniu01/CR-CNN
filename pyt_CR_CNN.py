import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def one_hot(indices, depth, on_value=1, off_value=0):
    np_ids = np.array(indices.cpu().data.numpy()).astype(int)
    if len(np_ids.shape) == 2:
        encoding = np.zeros([np_ids.shape[0], np_ids.shape[1], depth], dtype=int)
        added = encoding + off_value
        for i in range(np_ids.shape[0]):
            for j in range(np_ids.shape[1]):
                added[i, j, np_ids[i, j]] = on_value
        return Variable(torch.FloatTensor(added.astype(float))).cuda()
    if len(np_ids.shape) == 1:
        encoding = np.zeros([np_ids.shape[0], depth], dtype=int)
        added = encoding + off_value
        for i in range(np_ids.shape[0]):
            added[i, np_ids[i]] = on_value
        return Variable(torch.FloatTensor(added.astype(float))).cuda()


class CR_CNN(nn.Module):
    # def __init__(self, max_len, vocab_size, embedding_size, pos_embed_size,
    #              pos_embed_num, slide_window, class_num,
    #              num_filters, keep_prob):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(CR_CNN, self).__init__()
        # self.dw = embedding_size
        # self.vac_len = vocab_size
        self.dw = embedding.shape[1]
        self.vac_len = embedding.shape[0]
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.p = (self.k - 1) // 2
        self.n = max_len
        self.x_embedding = nn.Embedding(self.vac_len, self.dw)
        self.x_embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.d1_embedding = nn.Embedding(self.np, self.dp)
        self.d2_embedding = nn.Embedding(self.np, self.dp)
        self.init_r = np.sqrt(6 / (self.nr + self.dc))
        self.rel_weight = nn.Parameter(self.init_r * (torch.rand(self.dc, self.nr) - 0.5))
        self.dropout = nn.Dropout(self.keep_prob)
        self.conv = nn.Conv2d(1, self.dc, (self.k, self.d), (1, self.d), (self.p, 0), bias=True)  # renewed
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d((1, self.n), (1, self.dc))

    def concat_input(self, x, dist1, dist2, is_training=True):
        x_embed = self.x_embedding(x)  # (bz, n, dw)
        d1_embed = self.d1_embedding(dist1)
        d2_embed = self.d2_embedding(dist2)
        x_concat = torch.cat([x_embed, d1_embed, d2_embed], 2)
        if is_training:
            x_concat = self.dropout(x_concat)
        return x_concat

    def convolution(self, R):
        s = R.data.size()  # bz, n, d
        R = self.conv(R.view(s[0], 1, s[1], s[2]))  # bz, dc, n, 1
        rx = R.view(s[0], self.dc, s[1])
        rx = self.tanh(rx)  # added
        return rx  # bz, dc, n

    def max_pooling(self, rx, rel_weight):
        bz = rx.data.size()[0]
        max_rx = self.max_pool(rx.view(bz, 1, self.dc, self.n))  # (bz, dc)
        sc = torch.mm(max_rx.view(bz, self.dc), rel_weight)  # (bz, nr)
        return sc

    def forward(self, x, dist1, dist2, is_training=True):
        R = self.concat_input(x, dist1, dist2, is_training)
        R_star = self.convolution(R)
        sc = self.max_pooling(R_star, self.rel_weight)
        return sc


class PairwiseRankingLoss(nn.Module):
    def __init__(self, nr, pos_margin=2.5, neg_margin=0.5, gamma=2):
        super(PairwiseRankingLoss, self).__init__()
        self.nr = nr
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.gamma = gamma

    def forward(self, sc, in_y):
        pos_mask = one_hot(in_y, self.nr, 1000, 0)  # (bz, nr)
        neg_mask = one_hot(in_y, self.nr, 0, 1000)
        sc_neg = torch.max(sc - pos_mask, 1)[0]
        sc_pos = torch.max(sc - neg_mask, 1)[0]
        pos_ele = torch.mul((self.pos_margin - sc_pos), self.gamma)
        neg_ele = torch.mul((self.neg_margin + sc_neg), self.gamma)
        loss = torch.mean(torch.log1p(torch.exp(pos_ele)) + torch.log1p(torch.exp(neg_ele)))
        return loss
