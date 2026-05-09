from __future__ import print_function, division

import einops
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from thop import profile
from torch import einsum

def conv_relu_bn(in_channel, out_channel, dirate=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=dirate,
            dilation=dirate,
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv1(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):

        return self.act(self.conv(x))

class PConv(nn.Module):

    def __init__(self, c1, c2, k, s):
        super().__init__()

        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv1(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv1(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv1(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * res

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x_source

class SAFM(nn.Module):
    def __init__(self, in_ch, out_ch):    #64/128
        super(SAFM, self).__init__()
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, en, de):
        x = torch.cat([en, de], dim=1)
        out_x = torch.cat([en, de], dim=1)
        attn_map = torch.cat([en, de], dim=1)
        avg_map = torch.mean(attn_map, dim=1, keepdim=True)
        max_map, _ = torch.max(attn_map, dim=1, keepdim=True)
        agg = torch.concat([avg_map, max_map], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = en * sig[:, 0, :, :].unsqueeze(1) + de * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        out = out_x * attn
        return self.gamma * out + x



class DSM(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DSM, self).__init__()
        self.conv_layer = nn.Sequential(
            conv_relu_bn(in_ch, in_ch, 1),
            conv_relu_bn(in_ch, out_ch, 1),
            conv_relu_bn(out_ch, out_ch, 1),
        )

        self.dconv_layer = nn.Sequential(
            PConv(in_ch, out_ch, k=3, s=1),
            PConv(out_ch, out_ch, k=5, s=1),
            PConv(out_ch, out_ch, k=3, s=1),
        )
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()
        self.final_layer = conv_relu_bn(out_ch * 2, out_ch, 1)

    def forward(self, x):
        conv_out = self.conv_layer(x)
        dconv_out = self.dconv_layer(x)
        out = torch.concat([conv_out, dconv_out], dim=1)
        out = self.final_layer(out)
        out = self.ca(out)
        out = self.sa(out)
        return out

class up_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class SANet(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, deep_supervision=True, **kwargs):
        super(SANet, self).__init__()
        self.deep_supervision = deep_supervision
        n1 = 24
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_conv = nn.Conv2d(in_ch, filters[0],1, 1)
        self.Conv1 = DSM(filters[0], filters[0])
        self.Conv2 = DSM(filters[0], filters[1])
        self.Conv3 = DSM(filters[1], filters[2])
        self.Conv4 = DSM(filters[2], filters[3])
        self.Conv5 = DSM(filters[3], filters[4])
        self.neck5 = SAFM(filters[3], filters[4])
        self.neck4 = SAFM(filters[2], filters[3])
        self.neck3 = SAFM(filters[1], filters[2])
        self.neck2 = SAFM(filters[0], filters[1])
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(filters[4], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(filters[3], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters[2], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        e1 = self.Conv1(self.in_conv(x))
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.neck5(e4, d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.neck4(e3, d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.neck3(e2, d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.neck2(e1, d2)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        s1 = self.conv1(d2)
        s2 = self.conv2(d3)
        s3 = self.conv3(d4)
        s4 = self.conv4(d5)
        s5 = self.conv5(e5)

        s2 = F.interpolate(s2, scale_factor=2, mode='bilinear', align_corners=True)
        s3 = F.interpolate(s3, scale_factor=4, mode='bilinear', align_corners=True)
        s4 = F.interpolate(s4, scale_factor=8, mode='bilinear', align_corners=True)
        s5 = F.interpolate(s5, scale_factor=16, mode='bilinear', align_corners=True)

        if self.deep_supervision:
            outs = [torch.sigmoid(s1), torch.sigmoid(s2), torch.sigmoid(s3), torch.sigmoid(s4),
                    torch.sigmoid(s5), torch.sigmoid(out)]
        else:
            outs = torch.sigmoid(s1)

        return outs


if __name__ == "__main__":
    x = torch.rand(1, 1, 256, 256)
    model = SANet()
    outs = model(x)
    flops, params = profile(model, (x,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
