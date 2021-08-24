import torch
import torch.nn as nn
import torch.nn.functional as F
from Upsample import Upsample
from correlation_package.correlation import Correlation

import numpy as np

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

def predict_mask(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class ABMRNet(nn.Module):
    """
    Asymmetric Bilateral Motion Refinement netwrok
    """

    def __init__(self, md=2):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(ABMRNet, self).__init__()
        
        self.conv1a  = conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b  = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a  = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b  = conv(32, 32, kernel_size=3, stride=1)

        self.conv1_ASFM = conv(16,16, kernel_size=3, stride=1)
        self.conv2_ASFM = conv(32,32, kernel_size=3, stride=1)

        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([64, 64, 48, 32, 16])

        od = nd + 32 + 2
        self.conv2_0 = conv(od, 64, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 64, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 48, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 32, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 16, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.predict_mask2 = predict_mask(od + dd[4])
        self.upfeat2 = deconv(od + dd[4], 16, kernel_size=4, stride=2, padding=1) # Updates at ASFM_Occ.py 

        od = nd + 16 + 18
        self.conv1_0 = conv(od, 64, kernel_size=3, stride=1)
        self.conv1_1 = conv(od + dd[0], 64, kernel_size=3, stride=1)
        self.conv1_2 = conv(od + dd[1], 48, kernel_size=3, stride=1)
        self.conv1_3 = conv(od + dd[2], 32, kernel_size=3, stride=1)
        self.conv1_4 = conv(od + dd[3], 16, kernel_size=3, stride=1)
        self.predict_flow1 = predict_flow(od + dd[4])
        #self.deconv1 = deconv(2, 2, kernel_size=4, stride=2, padding=1) # Updates at ASFM_Occ.py 

        self.dc_conv1 = conv(od + dd[4], 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(64, 64, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(64, 48, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(48, 32, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(32, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(16)

        self.conv1_Res = conv(16, 16, kernel_size = 3, stride=1, padding=1 ,activation=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.to(x.device)

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask
    
    def forward(self, x, V_2, Z_2):
        '''
        :param x: two input frames
        :param V: symmetric bilateral motion vector field
        :param Z: initial reliability map (1 denotes Non-occ & 0 denotes Occ)
        '''

        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]


        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))

        warp2 = self.warp(self.conv2_ASFM(c22), V_2 * 5.0)
        corr2 = self.corr(c12 * Z_2, warp2)
        corr2 = self.leakyRELU(corr2)

        x = torch.cat((corr2, c12, V_2), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        V_2 = V_2 + self.predict_flow2(x)
        Z_2 = self.predict_mask2(x)
        
        
        feat1 = self.leakyRELU(self.upfeat2(x))
        V_1 = Upsample(V_2, 2)
        Z_1 = Upsample(Z_2, 2)
        warp1 = self.warp(self.conv1_ASFM(c21), V_1 * 10.0)
        
        c11_Res = feat1
        c11 = (c11 * F.sigmoid(Z_1)) + self.conv1_Res(c11_Res)
        c11 = self.leakyRELU(c11)
        corr1 = self.corr(c11, warp1)
        corr1 = self.leakyRELU(corr1)
        x = torch.cat((corr1, c11, V_1, feat1), 1)
        x = torch.cat((self.conv1_0(x), x), 1)
        x = torch.cat((self.conv1_1(x), x), 1)
        x = torch.cat((self.conv1_2(x), x), 1)
        x = torch.cat((self.conv1_3(x), x), 1)
        x = torch.cat((self.conv1_4(x), x), 1)
        V_1 = V_1 + self.predict_flow1(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow = V_1 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x))) # Normalized flow

        return F.interpolate(flow, scale_factor=2, mode='bilinear'), F.sigmoid(Z_1)