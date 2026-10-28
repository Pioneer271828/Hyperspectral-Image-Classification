import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_matrix):
        adj_matrix = adj_matrix.to(torch.float32)
        x = torch.matmul(adj_matrix, x)  # Graph convolution
        x = self.linear(x)
        return x

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,Q,A):
        super(GCN, self).__init__()
        self.gc1 = GraphConvLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, 64)
        self.Q=Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        self.A=A
        self.bn1=nn.BatchNorm1d(input_dim)
        self.bn2=nn.BatchNorm1d(hidden_dim)
        self.bn3=nn.BatchNorm1d(64)
        self.lin=nn.Linear(64,num_classes)
    def forward(self, x):
        (h, w, c) = x.shape
        x_flaten=x.reshape([h * w, -1])
        x_flaten=self.bn1(x_flaten)
        supX = torch.mm(self.norm_col_Q.t(), x_flaten)
        supX = F.relu(self.gc1(supX, self.A))
        supX=self.bn2(supX)
        supX = self.gc2(supX, self.A)
        supX=self.bn3(supX)
        supX=self.lin(supX)
        Y=torch.matmul(self.Q, supX)

        return F.softmax(Y, -1)