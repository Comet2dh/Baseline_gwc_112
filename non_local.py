import torch
from torch import nn
from torch.nn import functional as F

class PAM(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super(PAM, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        # bn = nn.BatchNorm2d
        #
        # self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                  kernel_size=1, stride=1, padding=0)
        # self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                      kernel_size=1, stride=1, padding=0)
        # self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)
        #
        # if sub_sample:
        #     self.g = nn.Sequential(self.g, max_pool_layer)
        #     self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x_l, x_r, num_groups):
        '''
        :param x: (b, c, h, w)
        :return:
        '''

        B, C, H, W = x_l.shape
        assert C % num_groups == 0
        N_c = C // num_groups

        # g_x = self.g(x_r).view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)

        theta_x = x_r.permute(0, 2, 3, 1)
        phi_x = x_l.permute(0, 2, 1, 3)
        f = x_l.new_zeros([B, num_groups, H, W, W])
        for i in range(num_groups):
            f[:, i, :, :, :] = torch.matmul(theta_x[:, :, :, N_c*i:N_c*(i+1)], phi_x[:, :, N_c*i:N_c*(i+1), :])//N_c
        volume = refimg_fea.new_zeros([B, num_groups, H, W, W])
        for j in range(W):
            if i > 0:
                volume[:, :, :, j, :W-j] = f[:, :, :, j, :-W+j]
            else:
                volume = f
        volume = volume.permute(0, 1, 3, 2, 4)
        volume = volume.contiguous()
        M = F.softmax(f, dim=-1)

        V = M.sum(4)>1

        # y = torch.matmul(f_div_C, g_x)
        # y = y.permute(0, 2, 1).contiguous()
        # y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        # W_y = self.W(y)
        # z = W_y + x

        return volume, V
