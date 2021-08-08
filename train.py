import numpy as np
import torch
import torch.utils.data as D
from torch.autograd import Variable
from sklearn.metrics import f1_score
import data_pro as pro
import pyt_CR_CNN as pa
import json
DW = 50
NDW = 100
NN = 123
NDP = 70
NNP = 123
NNR = 19
NDC = 1000
NKP = 0.5
NK = 3
NLR = 0.025
N_BATCH_SIZE = 80
epochs = 100
target_dict = json.load(open('./target_dict.txt', 'r', encoding='utf-8'))
c_target_dict = {value: key for key, value in target_dict.items()}
tr_target_dict = json.load(open('./tr_target_dict.txt', 'r', encoding='utf-8'))

data = pro.load_data('./nine_train.txt')
t_data = pro.load_data('./nine_test.txt')
word_dict = pro.build_dict(data[0])
x, y, e1, e2, dist1, dist2 = pro.vectorize(data, word_dict, NN)

y = np.array(y).astype(np.int64)
np_cat = np.concatenate((x, np.array(dist1), np.array(dist2)), 1)
e_x, e_y, e_e1, e_e2, e_dist1, e_dist2 = pro.vectorize(t_data, word_dict, NN)
y = np.array(y).astype(np.int64)
eval_cat = np.concatenate((e_x, np.array(e_dist1), np.array(e_dist2)), 1)

glove = '/home/niuhao/project/v2_ABSA_baseline/InterGCN-ABSA/glove.42B.300d.txt'
embedding = pro.load_embedding(glove, word_dict)

model = pa.CR_CNN(NN, embedding, NDP, NNP, NK, NNR, NDC, NKP).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=NLR, weight_decay=0.001)  # optimize all rnn parameters
loss_func = pa.PairwiseRankingLoss(NNR)


def data_unpack(cat_data, target):
    list_x = np.split(cat_data.numpy(), [NN, NN + NNP], 1)
    bx = Variable(torch.from_numpy(list_x[0])).cuda()
    bd1 = Variable(torch.from_numpy(list_x[1])).cuda()
    bd2 = Variable(torch.from_numpy(list_x[2])).cuda()
    target = Variable(target).cuda()
    return bx, bd1, bd2, target


def prediction(sc, y):
    ay = list(y.cpu().data.numpy())
    c_y = [c_target_dict[item] for item in ay]
    new_y = np.array([tr_target_dict[item] for item in c_y])
    predict = torch.max(sc, 1)[1].long()
    ap = list(predict.cpu().data.numpy())
    c_p = [c_target_dict[item] for item in ap]
    new_p = np.array([tr_target_dict[item] for item in c_p])
    f1 = f1_score(new_y, new_p, average='micro')
    return f1 * 100


for i in range(epochs):
    acc = 0
    loss = 0
    train = torch.from_numpy(np_cat.astype(np.int64))
    y_tensor = torch.LongTensor(y)
    train_datasets = D.TensorDataset(train, y_tensor)
    train_dataloader = D.DataLoader(train_datasets, N_BATCH_SIZE, True, num_workers=2)
    j = 0
    for (b_x_cat, b_y) in train_dataloader:
        bx, bd1, bd2, by = data_unpack(b_x_cat, b_y)
        wo = model(bx, bd1, bd2)
        acc += prediction(wo, by)
        l = loss_func(wo, by)
        j += 1
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss += l
    eval = torch.from_numpy(eval_cat.astype(np.int64))
    eval_acc = 0
    ti = 0
    y_tensor = torch.LongTensor(e_y)
    eval_datasets = D.TensorDataset(eval, y_tensor)
    eval_dataloader = D.DataLoader(eval_datasets, N_BATCH_SIZE, True, num_workers=2)
    for (b_x_cat, b_y) in eval_dataloader:
        bx, bd1, bd2, by = data_unpack(b_x_cat, b_y)
        wo = model(bx, bd1, bd2, False)
        eval_acc += prediction(wo, by)
        ti += 1
    print('epoch:', i, 'f1:', acc / j, '%   loss:', loss.cpu().data.numpy() / j, 'test_f1:', eval_acc / ti, '%')

torch.save(model.state_dict(), 'acnn_params.pkl')
# eval = torch.from_numpy(np_cat.astype(np.int64))
# model.load_state_dict(torch.load('acnn_params.pkl'))
