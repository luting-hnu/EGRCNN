"# -- coding: UTF-8 --"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from model import mlstm


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class BN_Re(nn.Module):
    def __init__(self, ch_out):
        super(BN_Re, self).__init__()
        self.BN_Re = nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.BN_Re(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=2, patch_size=256):
        super(U_Net, self).__init__()

        self.patch_size = patch_size
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16)
        self.lstm1 = mlstm.ConvLSTM(input_channels=16, hidden_channels=[16], kernel_size=3, step=2,
                                  effective_step=[1], height=self.patch_size, width=self.patch_size)

        self.Conv2 = conv_block(ch_in=16, ch_out=32)
        self.lstm2 = mlstm.ConvLSTM(input_channels=32, hidden_channels=[32], kernel_size=3, step=2,
                                    effective_step=[1], height=self.patch_size/2, width=self.patch_size/2)

        self.Conv3 = conv_block(ch_in=32, ch_out=64)
        self.lstm3 = mlstm.ConvLSTM(input_channels=64, hidden_channels=[64], kernel_size=3, step=2,
                                    effective_step=[1], height=self.patch_size/4, width=self.patch_size/4)

        self.Conv4 = conv_block(ch_in=64, ch_out=128)
        self.lstm4 = mlstm.ConvLSTM(input_channels=128, hidden_channels=[128], kernel_size=3, step=2,
                                    effective_step=[1], height=self.patch_size/8, width=self.patch_size/8)
        self.Conv5 = conv_block(ch_in=128, ch_out=256)
        self.lstm5 = mlstm.ConvLSTM(input_channels=256, hidden_channels=[256], kernel_size=3, step=2,
                                    effective_step=[1], height=self.patch_size/16, width=self.patch_size/16)  #aspp不缩小图像大小

        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv6 = conv_block(ch_in=256, ch_out=256)
        self.Conv_1x1_6 = nn.Conv2d(256, output_ch, kernel_size=1, stride=1, padding=0)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)
        self.Conv_1x1_5 = nn.Conv2d(128, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1_4 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_3_edge = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_3_edge_ = BN_Re(ch_out=output_ch)
        self.Conv_1x1_3 = nn.Conv2d(34, output_ch, kernel_size=1, stride=1, padding=0)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)
        self.Conv_1x1_2_edge = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(18, output_ch, kernel_size=1, stride=1, padding=0)
    def encoder(self, x):
        x1, xout = self.lstm1(self.Conv1, x)
        dif_x1 = torch.abs(xout[0]-xout[1])

        x2, xout = self.lstm2(nn.Sequential(self.Maxpool, self.Conv2), xout)
        dif_x2 = torch.abs(xout[0]-xout[1])

        x3, xout = self.lstm3(nn.Sequential(self.Maxpool, self.Conv3), xout)
        dif_x3 = torch.abs(xout[0]-xout[1])

        x4, xout = self.lstm4(nn.Sequential(self.Maxpool, self.Conv4), xout)
        dif_x4 = torch.abs(xout[0]-xout[1])

        x5, xout = self.lstm5(nn.Sequential(self.Maxpool, self.Conv5), xout)
        dif_x5 = torch.abs(xout[0]-xout[1])

        return x1[0]*dif_x1, x2[0]*dif_x2, x3[0]*dif_x3, x4[0]*dif_x4, x5[0]*dif_x5

    def forward(self, input):
        # encoding path

        x1, x2, x3, x4, x5 = self.encoder(input)

        # decoding + concat path

        x5 = self.Up_conv6(x5)
        d6_out = self.Conv_1x1_6(x5)
        d6_out = F.interpolate(d6_out, scale_factor=16, mode='bilinear')

        x5 = self.Up5(x5)
        d5 = torch.cat((x5, x4), dim=1)
        d5 = self.Up_conv5(d5)
        d5_out = self.Conv_1x1_5(d5)
        d5_out = F.interpolate(d5_out, scale_factor=8, mode='bilinear')   #上采样为原图大小



        d4 = self.Up4(d5)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.Up_conv4(d4)
        d4_out = self.Conv_1x1_4(d4)
        d4_out = F.interpolate(d4_out, scale_factor=4, mode='bilinear')

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv3(d3)
        d3_edge1 = self.Conv_1x1_3_edge(d3)     #输出边界预测
        d3_edge = F.interpolate(d3_edge1, scale_factor=2, mode='bilinear')
        d3_edge_ = self.Conv_1x1_3_edge_(d3_edge1)  #通过BN和RELU后，特征融合，预测分割结果
        d3_ = torch.cat((d3_edge_, d3), dim=1)
        d3_out = self.Conv_1x1_3(d3_)
        d3_out = F.interpolate(d3_out, scale_factor=2, mode='bilinear')



        d2 = self.Up2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.Up_conv2(d2)
        d2_edge = self.Conv_1x1_2_edge(d2)
        d2_edge_ = self.Conv_1x1_3_edge_(d2_edge)
        d2_ = torch.cat((d2_edge_, d2), dim=1)
        d2_out = self.Conv_1x1_2(d2_)
        return d6_out, d5_out, d4_out, d3_out, d2_out, d3_edge, d2_edge