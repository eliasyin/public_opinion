import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):
    def __init__(self, config, num_feature, hidden_size, num_heads):
        super(GraphAttention, self).__init__()

        self.dropout = config.dropout
        self.gat_layers = nn.ModuleList([
            BasicGraphAttentionLayer(config, num_feature, hidden_size) for _ in range(num_heads)
        ])
        self.elu = nn.ELU()
        self.final_att_layer = BasicGraphAttentionLayer(
            config, num_heads * hidden_size, hidden_size)

    def forward(self, x, adj):
        x = F.dropout(x.float(), self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.gat_layers], dim=1)
        x = F.dropout(self.elu(x), self.dropout, training=self.training)
        x = F.elu(self.final_att_layer(x, adj))
        # x = F.log_softmax(x, dim=1)
        return x

class BasicGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    简单来说，就是先计算出一个权重矩阵 Wh.shape: (N, out_features)
    然后根据这个权重矩阵构造出attention_weight，这个构造的过程就是不断改变Wh的形状的过程，
    将改造后得到的attention_weight归一化就是attention weight
    """

    def __init__(self, config,  in_features, out_features, concat=True):
        super(BasicGraphAttentionLayer, self).__init__()
        self.dropout = config.dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化参数，这里选择均匀分布
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.adj_w1 = nn.Parameter(torch.empty(size=(config.embedding_dim, 1)))
        nn.init.xavier_uniform_(self.adj_w1.data, gain=1.414)
        # self.adj_w2 = nn.Parameter(torch.empty(size=(config.embedding_dim, config.batch_size)))
        # nn.init.xavier_uniform_(self.adj_w2.data, gain=1.414)


        self.leakyrelu = nn.LeakyReLU(0.5)

    def forward(self, h, adj):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        # torch.matmul(a_input, self.a) : (N, N, 1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # a_input的第三个维度就是计算后的hidden_output, 上边这一步将hidden_output,压缩为1个值
        zero_vec = -9e15*torch.ones_like(e)  # (N, N)
        adj = torch.matmul(adj, self.adj_w1).squeeze()
        adj = torch.matmul(adj, adj.T)
        # 合并两个tensor adj>0 保存e，否则保存zero_vec  attention : (N, N)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # 将attention的权重随机置零。 一种增强鲁棒性的方法
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(
            N, dim=0)  # (N * N, out_features)
        Wh_repeated_alternating = Wh.repeat(N, 1)  # (N * N, out_features)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionWithMask(nn.Module):
    def __init__(self, config, num_feature, hidden_size, num_heads):
        super(GraphAttentionWithMask, self).__init__()

        self.dropout = config.dropout
        self.gat_layers = nn.ModuleList([
            BasicGraphAttentionLayerWithMask(config, num_feature, hidden_size) for _ in range(num_heads)
        ])
        self.act = config.activation
        # self.final_att_layer = BasicGraphAttentionLayerWithMask(
        #     config, num_heads * hidden_size, hidden_size)
        self.dense = nn.Linear(num_heads * hidden_size, hidden_size)

    def forward(self, x, mask, adj):
        # x = F.dropout(x.float(), self.dropout, training=self.training)
        x = torch.cat([att(x, mask, adj) for att in self.gat_layers], dim=1)
        # x = F.dropout(self.elu(x), self.dropout, training=self.training)
        x = self.act(self.dense(x))
        # x = F.log_softmax(x, dim=1)
        return x

class BasicGraphAttentionLayerWithMask(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    简单来说，就是先计算出一个权重矩阵 Wh.shape: (N, out_features)
    然后根据这个权重矩阵构造出attention_weight，这个构造的过程就是不断改变Wh的形状的过程，
    将改造后得到的attention_weight归一化就是attention weight
    """

    def __init__(self, config,  in_features, out_features, concat=True):
        super(BasicGraphAttentionLayerWithMask, self).__init__()
        self.dropout = config.dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.config = config

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化参数，这里选择均匀分布
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.adj_w1 = nn.Parameter(torch.empty(size=(config.embedding_dim, 1)))
        nn.init.xavier_uniform_(self.adj_w1.data, gain=1.414)
        # self.adj_w2 = nn.Parameter(torch.empty(size=(config.embedding_dim, config.batch_size)))
        # nn.init.xavier_uniform_(self.adj_w2.data, gain=1.414)

        self.act = config.activation

    def forward(self, h, mask, adj):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        
        # h = self._feature_mask(h, mask) * h
        h = mask * h
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        # torch.matmul(a_input, self.a) : (N, N, 1)
        e = self.act(torch.matmul(a_input, self.a).squeeze(2))
        # a_input的第三个维度就是计算后的hidden_output, 上边这一步将hidden_output,压缩为1个值
        zero_vec = -9e15*torch.ones_like(e)  # (N, N)
        adj = torch.matmul(adj, self.adj_w1).squeeze()
        adj = torch.matmul(adj, adj.T)
        # 合并两个tensor adj>0 保存e，否则保存zero_vec  attention : (N, N)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # 将attention的权重随机置零。 一种增强鲁棒性的方法
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _feature_mask(self, feature, mask):
        dim = feature.shape[2]
        mask = mask.unsqueeze(0).repeat(dim, 1, 1).permute(1, 2, 0)
        return mask

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(
            N, dim=0)  # (N * N, out_features)
        Wh_repeated_alternating = Wh.repeat(N, 1)  # (N * N, out_features)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

