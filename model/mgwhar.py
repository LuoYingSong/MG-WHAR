import argparse

import torch
from torch import nn as nn
from torch.nn import init, functional as F

from util.util import makedir, device
from model.base import BaseModel

begin_idx = 12
para_dict = {
    "word": "55",
    "topic": "55",
    "node_num": 15,
    "seed": 2022,
    "length": 1,
    "sample_size": 24,
    "dropout": 0.0,
    "lstm_size": 64,
    "lstm_layer_num": 2,
    "gconv1_dim": 64,
    "class_num": 17,
    "seq_size": 24,
    "chanels": 96,
    "batch_size": 128,
    "lr": 0.001,
    "lr_decay": 0.07588660882395623,
    "lr_step": 20,
    "init_weights": "orthogonal",
    "optimizer": "Adam",
    "epochs": 30,
    "print_freq": 100,
    "K": 2,
    "dataset": "opportunity"
}
print(para_dict)
cfg = argparse.Namespace(**para_dict)


class GraphConv(nn.Module):

    def __init__(self, in_dim, out_dim, graph) -> None:
        super(GraphConv, self).__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.graph = self.normalize_adj(torch.from_numpy(graph).float().to(device)).detach()
        self._in_dim = in_dim
        self._out_dim = out_dim

    @staticmethod
    def normalize_adj(adj):
        adj = adj + torch.eye(adj.size(0)).to(device)
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(adj).mm(D)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)  # (128,24,35,output_dim)
        batch, length, sensors, dim = x.shape
        x = x.reshape(batch * length, sensors, dim)
        x = torch.matmul(self.graph, x)
        x = self.dense(x)
        x = F.relu(x)
        x = x.reshape(batch, length, sensors, self._out_dim)
        x = x.permute(0, 2, 1, 3)
        return x


class SensorConv(nn.Module):

    def __init__(self):
        super(SensorConv, self).__init__()
        self.conv1 = nn.Conv2d(1, cfg.chanels, (5, 3), padding=(2, 0))  # 设置为 （5，3）
        self.bn1 = nn.BatchNorm2d(cfg.chanels)
        self.conv2 = nn.Conv2d(cfg.chanels, cfg.chanels, (5, 1), padding=(2, 0))
        self.bn2 = nn.BatchNorm2d(cfg.chanels)

    def forward(self, x):
        x = x.unsqueeze(-1)  # -1, length, 3, 1
        x = x.permute(0, 3, 1, 2)  # -1, 1, length, 1
        x = F.relu(self.bn1(self.conv1(x)))  # (128, 64, length, 45)
        x = F.relu(self.bn2(self.conv2(x)))  # (128, 64, length, 45)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], cfg.sample_size // 1, -1)

        return x


class LSTM(nn.Module):

    def __init__(self, in_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_size=cfg.lstm_size, num_layers=cfg.lstm_layer_num, batch_first=True)
        self.dense = nn.Linear(cfg.lstm_size, cfg.class_num)

    def forward(self, inputs):
        # inputs shape [size, cfg.sample_size, cfg.fc]
        x = inputs  # cfg.batch_size * cfg.sample_size, cfg.fc
        out, _ = self.lstm(x)
        output = out[:, -1, :]
        d = self.dense(output)
        res = F.relu(d)
        return res


class GCN(nn.Module):
    def __init__(self, graph):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(cfg.chanels, cfg.gconv1_dim, graph)
        self.gcn_list = nn.ModuleList([GraphConv(cfg.gconv1_dim, cfg.gconv1_dim, graph) for i in range(5)])
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        x = inputs  # (128,35 , 24,output_dim)
        g1 = F.relu(self.gcn1(x))  # (128,35 , 24, 64)
        pre = g1
        for i in range(cfg.K - 1):
            out = F.relu(self.gcn_list[i](pre))
            pre = out
        g2 = pre
        g2 = g2.permute(0, 2, 1, 3)  # (128  24, 35, 64)
        return g2


class ANGELO(BaseModel):
    def __init__(self, g1, g2, g3, dataset='opportunity'):
        super(ANGELO, self).__init__(dataset)
        self.conv = nn.ModuleList([SensorConv() for _ in range(cfg.node_num)])
        self.gcn1 = GCN(g1)
        self.gcn2 = GCN(g2)
        self.gcn3 = GCN(g3)
        self.lstm = LSTM(cfg.gconv1_dim * cfg.node_num)
        makedir(self.path_checkpoints)
        makedir(self.path_logs)
        makedir(self.path_visuals)

    def forward(self, x):  # (128, sensors, length * sample_size, 3)
        batch, sensors_num, length, dim = x.shape
        sensors_data = []
        for i in range(x.shape[1]):
            sensor_data = x[:, i, ...]  # (128,length * sample_size , 3)
            sensor_data = sensor_data.reshape(-1, cfg.length * cfg.sample_size, 3)  # (128 , length * sample_size, 3)
            after_conv = self.conv[i](sensor_data)  # (128,sample_size , output_dim)!!!
            sensors_data.append(after_conv)
        sensors_data = torch.stack(sensors_data, dim=1)  # (128 , 35,sample_size ,output_dim)
        g1 = self.gcn1(sensors_data)
        g2 = self.gcn2(sensors_data)
        g3 = self.gcn3(sensors_data)
        out = (g1 + g2 + g3) / 3
        gcn_mean = out.reshape(batch, length, -1)
        output = self.lstm(gcn_mean)
        return output
