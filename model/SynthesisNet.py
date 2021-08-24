import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynFilter(nn.Module):
    def __init__(self, kernel_size=(3,3), padding=1, DDP=False):
        super(DynFilter, self).__init__()

        self.padding = padding
        
        filter_localexpand_np = np.reshape(np.eye(np.prod(kernel_size), np.prod(kernel_size)), (np.prod(kernel_size), 1, kernel_size[0], kernel_size[1]))
        if DDP:
            self.register_buffer('filter_localexpand', torch.FloatTensor(filter_localexpand_np)) # for DDP model
        else:
            self.filter_localexpand = torch.FloatTensor(filter_localexpand_np).cuda() # for single model

    def forward(self, x, filter):
        x_localexpand = []

        for c in range(x.size(1)):
            x_localexpand.append(F.conv2d(x[:, c:c + 1, :, :], self.filter_localexpand, padding=self.padding))

        x_localexpand = torch.cat(x_localexpand, dim=1)
        x = torch.sum(torch.mul(x_localexpand, filter), dim=1).unsqueeze(1)

        return x


class Feature_Pyramid(nn.Module):
    def __init__(self):
        super(Feature_Pyramid, self).__init__()

        self.Feature_First = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU())

        self.Feature_Second = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU())

        self.Feature_Third = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU())

    def forward(self, Input):
        Feature_1 = self.Feature_First(Input)
        Feature_2 = self.Feature_Second(Feature_1)
        Feature_3 = self.Feature_Third(Feature_2)

        return Feature_1, Feature_2, Feature_3


class GridNet_Filter(nn.Module):
    def __init__(self, output_channel):
        super(GridNet_Filter, self).__init__()

        def First(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )

        def lateral(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )

        def downsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(2, 2),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
            )

        def upsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
            )

        def Last(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )

        self.First_Block = First(4 * (3+32), 32)  # 4*RGB(3) + 4* 1st features(32)

        self.Row1_1 = lateral(32, 32)
        self.Row1_2 = lateral(32, 32)
        self.Row1_3 = lateral(32, 32)
        self.Row1_4 = lateral(32, 32)
        self.Row1_5 = lateral(32, 32)
        self.Last_Block = Last(32, output_channel) 

        self.Row2_0 = First(4 * 64, 64)

        self.Row2_1 = lateral(64, 64)  
        self.Row2_2 = lateral(64, 64)
        self.Row2_3 = lateral(64, 64)
        self.Row2_4 = lateral(64, 64)
        self.Row2_5 = lateral(64, 64)

        self.Row3_0 = First(4 * 96, 96)

        self.Row3_1 = lateral(96, 96)  
        self.Row3_2 = lateral(96, 96)
        self.Row3_3 = lateral(96, 96)
        self.Row3_4 = lateral(96, 96)
        self.Row3_5 = lateral(96, 96)

        self.Col1_1 = downsampling(32, 64)
        self.Col2_1 = downsampling(64, 96)
        self.Col1_2 = downsampling(32, 64)
        self.Col2_2 = downsampling(64, 96)
        self.Col1_3 = downsampling(32, 64)
        self.Col2_3 = downsampling(64, 96)

        self.Col1_4 = upsampling(64, 32)
        self.Col2_4 = upsampling(96, 64)
        self.Col1_5 = upsampling(64, 32)
        self.Col2_5 = upsampling(96, 64)
        self.Col1_6 = upsampling(64, 32)
        self.Col2_6 = upsampling(96, 64)

    def forward(self, V_0_t_SBM, V_0_t_ABM, V_1_t_SBM, V_1_t_ABM):
        Variable1_1 = self.First_Block(torch.cat((V_0_t_SBM[0], V_0_t_ABM[0], V_1_t_SBM[0], V_1_t_ABM[0]), dim=1))  # 1
        Variable1_2 = self.Row1_1(Variable1_1) + Variable1_1  # 2
        Variable1_3 = self.Row1_2(Variable1_2) + Variable1_2  # 3

        Variable2_0 = self.Row2_0(torch.cat((V_0_t_SBM[1][:, 3:, :, :], V_0_t_ABM[1][:, 3:, :, :], V_1_t_SBM[1][:, 3:, :, :], V_1_t_ABM[1][:, 3:, :, :]), dim=1))  # 4
        Variable2_1 = self.Col1_1(Variable1_1) + Variable2_0  # 5
        Variable2_2 = self.Col1_2(Variable1_2) + self.Row2_1(Variable2_1) + Variable2_1  # 6
        Variable2_3 = self.Col1_3(Variable1_3) + self.Row2_2(Variable2_2) + Variable2_2  # 7

        Variable3_0 = self.Row3_0(torch.cat((V_0_t_SBM[2][:, 3:, :, :], V_0_t_ABM[2][:, 3:, :, :], V_1_t_SBM[2][:, 3:, :, :], V_1_t_ABM[2][:, 3:, :, :]), dim=1))  # 8
        Variable3_1 = self.Col2_1(Variable2_1) + Variable3_0  # 9
        Variable3_2 = self.Col2_2(Variable2_2) + self.Row3_1(Variable3_1) + Variable3_1  # 10
        Variable3_3 = self.Col2_3(Variable2_3) + self.Row3_2(Variable3_2) + Variable3_2  # 11

        Variable3_4 = self.Row3_3(Variable3_3) + Variable3_3  # 10
        Variable3_5 = self.Row3_4(Variable3_4) + Variable3_4  # 11
        Variable3_6 = self.Row3_5(Variable3_5) + Variable3_5  # 12

        Variable2_4 = self.Col2_4(Variable3_4) + self.Row2_3(Variable2_3) + Variable2_3  # 13
        Variable2_5 = self.Col2_5(Variable3_5) + self.Row2_4(Variable2_4) + Variable2_4  # 14
        Variable2_6 = self.Col2_6(Variable3_6) + self.Row2_5(Variable2_5) + Variable2_5  # 15

        Variable1_4 = self.Col1_4(Variable2_4) + self.Row1_3(Variable1_3) + Variable1_3  # 16
        Variable1_5 = self.Col1_5(Variable2_5) + self.Row1_4(Variable1_4) + Variable1_4  # 17
        Variable1_6 = self.Col1_6(Variable2_6) + self.Row1_5(Variable1_5) + Variable1_5  # 18

        return self.Last_Block(Variable1_6)  # 19


class GridNet_Refine(nn.Module):
    def __init__(self):
        super(GridNet_Refine, self).__init__()

        def First(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )

        def lateral(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )

        def downsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(2, 2),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
            )

        def upsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
            )

        def Last(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )

        self.First_Block = First(3 + 32 + 4 * 32, 32)

        self.Row1_1 = lateral(32, 32)
        self.Row1_2 = lateral(32, 32)
        self.Row1_3 = lateral(32, 32)
        self.Row1_4 = lateral(32, 32)
        self.Row1_5 = lateral(32, 32)
        self.Last_Block = Last(32, 3)

        self.Row2_0 = First(4 * 64, 64)

        self.Row2_1 = lateral(64, 64)
        self.Row2_2 = lateral(64, 64)
        self.Row2_3 = lateral(64, 64)
        self.Row2_4 = lateral(64, 64)
        self.Row2_5 = lateral(64, 64)

        self.Row3_0 = First(4 * 96, 96)

        self.Row3_1 = lateral(96, 96)
        self.Row3_2 = lateral(96, 96)
        self.Row3_3 = lateral(96, 96)
        self.Row3_4 = lateral(96, 96)
        self.Row3_5 = lateral(96, 96)

        self.Col1_1 = downsampling(32, 64)
        self.Col2_1 = downsampling(64, 96)
        self.Col1_2 = downsampling(32, 64)
        self.Col2_2 = downsampling(64, 96)
        self.Col1_3 = downsampling(32, 64)
        self.Col2_3 = downsampling(64, 96)

        self.Col1_4 = upsampling(64, 32)
        self.Col2_4 = upsampling(96, 64)
        self.Col1_5 = upsampling(64, 32)
        self.Col2_5 = upsampling(96, 64)
        self.Col1_6 = upsampling(64, 32)
        self.Col2_6 = upsampling(96, 64)

    def forward(self, V_t, V_SBM_bw, V_ABM_bw, V_SBM_fw, V_ABM_fw):
        Variable1_1 = self.First_Block(torch.cat((V_t, V_SBM_bw[0][:, 3:, :, :], V_ABM_bw[0][:, 3:, :, :], V_SBM_fw[0][:, 3:, :, :], V_ABM_fw[0][:, 3:, :, :]), dim=1))  # 1
        Variable1_2 = self.Row1_1(Variable1_1) + Variable1_1  # 2
        Variable1_3 = self.Row1_2(Variable1_2) + Variable1_2  # 3

        Variable2_0 = self.Row2_0(torch.cat((V_SBM_bw[1][:, 3:, :, :], V_ABM_bw[1][:, 3:, :, :], V_SBM_fw[1][:, 3:, :, :], V_ABM_fw[1][:, 3:, :, :]), dim=1))  # 4
        Variable2_1 = self.Col1_1(Variable1_1) + Variable2_0  # 5
        Variable2_2 = self.Col1_2(Variable1_2) + self.Row2_1(Variable2_1) + Variable2_1  # 6
        Variable2_3 = self.Col1_3(Variable1_3) + self.Row2_2(Variable2_2) + Variable2_2  # 7

        Variable3_0 = self.Row3_0(torch.cat((V_SBM_bw[2][:, 3:, :, :], V_ABM_bw[2][:, 3:, :, :], V_SBM_fw[2][:, 3:, :, :], V_ABM_fw[2][:, 3:, :, :]), dim=1))  # 8
        Variable3_1 = self.Col2_1(Variable2_1) + Variable3_0  # 9
        Variable3_2 = self.Col2_2(Variable2_2) + self.Row3_1(Variable3_1) + Variable3_1  # 10
        Variable3_3 = self.Col2_3(Variable2_3) + self.Row3_2(Variable3_2) + Variable3_2  # 11

        Variable3_4 = self.Row3_3(Variable3_3) + Variable3_3  # 12
        Variable3_5 = self.Row3_4(Variable3_4) + Variable3_4  # 13
        Variable3_6 = self.Row3_5(Variable3_5) + Variable3_5  # 14

        Variable2_4 = self.Col2_4(Variable3_4) + self.Row2_3(Variable2_3) + Variable2_3  # 15
        Variable2_5 = self.Col2_5(Variable3_5) + self.Row2_4(Variable2_4) + Variable2_4  # 16
        Variable2_6 = self.Col2_6(Variable3_6) + self.Row2_5(Variable2_5) + Variable2_5  # 17

        Variable1_4 = self.Col1_4(Variable2_4) + self.Row1_3(Variable1_3) + Variable1_3  # 18
        Variable1_5 = self.Col1_5(Variable2_5) + self.Row1_4(Variable1_4) + Variable1_4  # 19
        Variable1_6 = self.Col1_6(Variable2_6) + self.Row1_5(Variable1_5) + Variable1_5  # 20

        return self.Last_Block(Variable1_6)  # 21


class SynthesisNet(nn.Module):
    def __init__(self, args):
        super(SynthesisNet, self).__init__()
        
        self.ctxNet = Feature_Pyramid()

        self.FilterNet = GridNet_Filter(3 * 3 * 4)

        self.RefineNet = GridNet_Refine()
        
        self.Filtering = DynFilter(kernel_size=(3,3), padding=1, DDP=args.DDP)

    def warp(self, x, flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float().to(x.device)

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

    def Flow_pyramid(self, flow):
        flow_pyr = []
        flow_pyr.append(flow)
        for i in range(1, 3):
            flow_pyr.append(F.interpolate(flow, scale_factor=0.5 ** i, mode='bilinear') * (0.5 ** i))
        return flow_pyr

    def Img_pyramid(self, Img):
        img_pyr = []
        img_pyr.append(Img)
        for i in range(1, 3):
            img_pyr.append(F.interpolate(Img, scale_factor=0.5 ** i, mode='bilinear'))
        return img_pyr

    def forward(self, input, time_step=0.5):
        I0 = input[:, :3, :, :]  # First frame
        I1 = input[:, 3:6, :, :]  # Second frame
        SBM_t_1 = input[:, 6:8, :, :]  
        SBM_Pyr_t_1 = self.Flow_pyramid(SBM_t_1)
        ABM_t_1 = input[:, 8:10, :, :] 
        ABM_t_0 = input[:, 10:12, :, :]
        
        ABM_Pyr_t_0 = self.Flow_pyramid(ABM_t_0)
        ABM_Pyr_t_1 = self.Flow_pyramid(ABM_t_1)

        V_Pyr_0 = self.ctxNet(I0)  # Feature pyramid of first frame
        V_Pyr_1 = self.ctxNet(I1)  # Feature pyramid of second frame

        I_Pyr_0 = self.Img_pyramid(I0)
        I_Pyr_1 = self.Img_pyramid(I1)

        V_Pyr_0_t_SBM = []
        V_Pyr_1_t_SBM = []
        V_Pyr_0_t_ABM = []
        V_Pyr_1_t_ABM = []

        for i in range(3):
            V_0_t_SBM = self.warp(torch.cat((I_Pyr_0[i], V_Pyr_0[i]), dim=1), SBM_Pyr_t_1[i] * (-1))
            V_0_t_ABM = self.warp(torch.cat((I_Pyr_0[i], V_Pyr_0[i]), dim=1), ABM_Pyr_t_0[i])

            V_1_t_SBM = self.warp(torch.cat((I_Pyr_1[i], V_Pyr_1[i]), dim=1), SBM_Pyr_t_1[i])
            V_1_t_ABM = self.warp(torch.cat((I_Pyr_1[i], V_Pyr_1[i]), dim=1), ABM_Pyr_t_1[i])

            V_Pyr_0_t_SBM.append(V_0_t_SBM)
            V_Pyr_0_t_ABM.append(V_0_t_ABM)

            V_Pyr_1_t_SBM.append(V_1_t_SBM)
            V_Pyr_1_t_ABM.append(V_1_t_ABM)

        DF = F.softmax(self.FilterNet(V_Pyr_0_t_SBM, V_Pyr_0_t_ABM, V_Pyr_1_t_SBM, V_Pyr_1_t_ABM), dim=1)
        
        Filtered_input = []
        for i in range(V_Pyr_0_t_SBM[0].size(1)):
            Filtered_input.append(self.Filtering(torch.cat((V_Pyr_0_t_SBM[0][:, i:i + 1, :, :], V_Pyr_0_t_ABM[0][:, i:i + 1, :, :],
                                                            V_Pyr_1_t_SBM[0][:, i:i + 1, :, :], V_Pyr_1_t_ABM[0][:, i:i + 1, :, :]), dim=1), DF))

        Filtered_t = torch.cat(Filtered_input, dim=1)

        R_t = self.RefineNet(Filtered_t, V_Pyr_0_t_SBM, V_Pyr_0_t_ABM, V_Pyr_1_t_SBM, V_Pyr_1_t_ABM)

        output = Filtered_t[:, :3, :, :] + R_t

        return output