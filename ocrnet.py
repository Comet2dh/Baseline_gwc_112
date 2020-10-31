import torch
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *

class ObjectAttention(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels):
        super(ObjectAttention,self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.out_channels = out_channels
        self.phi = nn.Sequential(convbn(self.in_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU(),
                                 convbn(self.in_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU())
        self.psi = nn.Sequential(convbn(self.in_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU(),
                                 convbn(self.in_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU())
        self.f_down = nn.Sequential(convbn(self.in_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU())
        self.f_up = nn.Sequential(convbn(self.key_channels, self.in_channels, 1, 1, 0, 1),
                                 nn.ReLU())

    def forward(self, x, f_k):
        batch_size, _, h, w = x.shape
        # x:(N,C,H,W), f_k:(N,C,K,1)
        query = self.psi(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1) # N * hw * c
        key = self.phi(f_k).view(batch_size, self.key_channels, -1)
        value = self.f_down(f_k).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1) # N * k * c

        # 3. 计算权重w
        sim_map = torch.matmul(query, key) # N * hw * k
        sim_map = F.softmax(sim_map, dim=-1) # 在k维上做的相似性度量
        # 4. Object contextual representations
        context = torch.matmul(sim_map, value) # N * hw * c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context) # N * c * h * w

        return context

class OCR(nn.Module):
    def __init__(self, d_num, c_num, dropout=0.05):
        super(OCR, self).__init__()
        self.d_num = d_num
        self.c_num = c_num
        self.dropout = dropout
        self.objAtten = ObjectAttention(in_channels=self.c_num, key_channels=self.c_num, out_channels=self.c_num)
        self.disp_conv = nn.Sequential(convbn_3d(self.d_num, self.d_num, 1, 1, 0),
                      nn.ReLU(),
                      convbn_3d(self.d_num, 1, 1, 1, 0),
                      nn.ReLU())
        self.final_conv = nn.Sequential(convbn(self.c_num *2, 128, 1, 1, 0, 1),
                               nn.ReLU(),
                               nn.Dropout2d(self.dropout))
        self.classifier = nn.Conv2d(128, self.d_num, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, cost):
        cost = torch.squeeze(cost, 1) #[1, 48, 64, 128]
        # 1. d张概率图（这里d=D/4=48）
        batch, n_class, height, width = cost.shape
        d_flat = cost.view(batch, n_class, -1)
        # M:(N,K,L)
        M = F.softmax(d_flat, dim=2)

        # 2. 计算每个d类特征
        feats = self.disp_conv(x.permute(0, 2, 1, 3, 4))
        feats = torch.squeeze(feats, 1) # (N,C,H,W)
        channel = feats.shape[1]
        x_flat = feats.view(batch, channel, -1)  # x:(N,C,L)
        x_flat = x_flat.permute(0, 2, 1)  # x:(N,L,C)
        d_feas = torch.matmul(M, x_flat).permute(0, 2, 1).unsqueeze(3)  # (N,C,K,1)

        y = self.objAtten(feats, d_feas) # y:(N,C,H,W)
        output = self.final_conv(torch.cat([y, feats], 1))
        output = self.classifier(output)
        return output

