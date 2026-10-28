import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.sparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))    #初始化为可训练的参数
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        G = G.to(torch.float32)
        if self.bias is not None:
            x = x + self.bias

        x = torch.sparse.mm(G.to_sparse(), x)
        return x
class HGNN_weight(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,Q,A, dropout=0.5, momentum=0.1):
        super(HGNN_weight, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(input_dim, hidden_dim)
        self.hgc2 = HGNN_conv(hidden_dim, 64)
        # self.computeG = compute_G(W)
        self.batch_normalzation1 = nn.BatchNorm1d(input_dim)
        self.batch_normalzation2 = nn.BatchNorm1d(hidden_dim)
        self.batch_normalzation3 = nn.BatchNorm1d(64)
        self.A = A

        self.lin = nn.Linear(64, num_classes)
        self.Q=Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
    def forward(self, x):
        (h, w, c) = x.shape
        x_flaten = x.reshape([h * w, -1])
        x_flaten = self.batch_normalzation1(x_flaten)
        supX = torch.mm(self.norm_col_Q.t(), x_flaten)
        supX = self.hgc1(supX, self.A)
        supX = F.relu(supX)
        supX = self.batch_normalzation2(supX)
        supX = F.dropout(supX, self.dropout)
        supX = self.hgc2(supX, self.A)
        supX = F.relu(supX)
        supX = self.batch_normalzation3(supX)
        supX=self.lin(supX)
        Y = torch.matmul(self.Q, supX)
        return F.softmax(Y, -1)