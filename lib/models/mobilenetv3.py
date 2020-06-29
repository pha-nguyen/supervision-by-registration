import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
import numpy as np


__all__ = ['MobileNetV3', 'mobilenetv3']


def find_tensor_peak_batch(heatmap, radius, downsample, threshold=0.000001):
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    assert radius > 0 and isinstance(
        radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()

    def normalize(x, L):
        return -1. + 2. * x.data / (L-1)
    boxes = [index_w - radius, index_h - radius,
             index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    # affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    #theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:, 0, 0] = (boxes[2]-boxes[0])/2
    affine_parameter[:, 0, 2] = (boxes[2]+boxes[0])/2
    affine_parameter[:, 1, 1] = (boxes[3]-boxes[1])/2
    affine_parameter[:, 1, 2] = (boxes[3]+boxes[1])/2
    # extract the sub-region heatmap
    theta = affine_parameter.to(heatmap.device)
    grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
    grid = F.affine_grid(theta, grid_size)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

    X = torch.arange(-radius, radius+1).to(heatmap).view(1, 1, radius*2+1)
    Y = torch.arange(-radius, radius+1).to(heatmap).view(1, radius*2+1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts, -1), 1)
    x = torch.sum((sub_feature*X).view(num_pts, -1), 1) / sum_region + index_w
    y = torch.sum((sub_feature*Y).view(num_pts, -1), 1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y], 1), score


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride,
                       padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, pts_num=1000, input_size=112, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        self.pts_num = pts_num
        last_channel = 512
        self.downsample = 7
        self.num_stgs = 3
        if mode == 'small':
            # refer to Table 2 in paper
            cfg_shared = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                # [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                # [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        last_channel = make_divisible(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in cfg_shared:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(
                input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        last_conv = make_divisible(576 * width_mult)
        self.features.append(conv_1x1_bn(
            input_channel, last_conv, nlin_layer=Hswish))
        # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        # self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 1))
        self.features.append(Hswish(inplace=True))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout),    # refer to paper section 6
        #     nn.Linear(last_channel, n_class),
        # )

        self.CPM_feature = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(
                inplace=True),  # CPM_1
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))  # CPM_2

        assert self.num_stgs >= 1, 'stages of cpm must >= 1 not : {:}'.format(
            self.num_stgs)
        stage1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1,
                      padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        stages = [stage1]
        for i in range(1, self.num_stgs):
            stagex = nn.Sequential(
                nn.Conv2d(128+pts_num, 128, kernel_size=7,
                          dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128,         128, kernel_size=7,
                          dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128,         128, kernel_size=7,
                          dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128,         128, kernel_size=3,
                          dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128,         128, kernel_size=3,
                          dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128,         128, kernel_size=3,
                          dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128,         128, kernel_size=3,
                          dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128,         128, kernel_size=1,
                          padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(128,     pts_num, kernel_size=1, padding=0))
            stages.append(stagex)
        self.stages = nn.ModuleList(stages)

        self._initialize_weights()

    def forward(self, x):
        assert x.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(x.size())
        batch_size, feature_dim, feature_w = x.size(0), x.size(1), x.size(2)
        batch_cpms, batch_locs, batch_scos = [], [], []

        feature = self.features(x)
        # print(feature.shape)
        xfeature = self.CPM_feature(feature)
        for i in range(self.num_stgs):
            if i == 0:
                cpm = self.stages[i](xfeature)
            else:
                cpm = self.stages[i](torch.cat([xfeature, batch_cpms[i-1]], 1))
            batch_cpms.append(cpm)

        # The location of the current batch
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(
                batch_cpms[-1][ibatch], 4, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(
            batch_locs), torch.stack(batch_scos)

        return batch_cpms, batch_locs, batch_scos

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=True)
        # raise NotImplementedError
    return model


if __name__ == '__main__':
    net = mobilenetv3(pts_num=68, input_size=112, width_mult=0.25)
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel()
                                       for p in net.parameters())/1000000.0))
    input_size = (8, 3, 112, 112)

    x = torch.randn(input_size)
    batch_cpms, batch_locs, batch_scos = net(x)

    print("batch_cpms.shape: {}, batch_locs.shape: {}, batch_scos.shape: {}".format(
        batch_cpms[0].shape, batch_locs.shape, batch_scos.shape))
