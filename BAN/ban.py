import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k  #3
        self.v_dim = v_dim  #128 视觉特征向量的维度
        self.q_dim = q_dim  #128 问题特征向量的维度
        self.h_dim = h_dim  #256 中间隐藏特征的维度
        self.h_out = h_out  #2 输出的注意力头数
        # K:注意力机制的次数，用于多次交互
        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)  #1D 平均池化，用于双线性池化后特征的降维

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())  #头的数量较少,用参数化的矩阵实现双线性交互
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)  #头的数量较多,使用线性变换实现注意力权重计算

        self.bn = nn.BatchNorm1d(h_dim)  #批归一化

    def attention_pooling(self, v, q, att_map):
        #根据计算得到的注意力映射 att_map，对视觉特征 v 和问题特征 q 进行加权融合
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))  #使用 torch.einsum 计算三者的张量乘积（双线性交互）
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits  #融合后的特征，维度为 (batch_size, h_dim)

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)  #非线性映射
            q_ = self.q_net(q)  #非线性映射
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias  #使用 h_mat 和 h_bias 计算注意力映射
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    """
    动态构建一个多层全连接网络
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)