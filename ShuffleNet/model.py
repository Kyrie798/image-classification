import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    batchs_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchs_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchs_size, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, input_c, out_c, stride):
        super().__init__()
        self.stride = stride
        assert out_c % 2 == 0
        branch_features = out_c // 2
        
        if self.stride == 2:
            self.branch1 = nn.Sequential(nn.Conv2d(input_c, out_c, kernel_size=3, stride=self.stride, padding=1, bias=False),
                                         nn.BatchNorm2d(out_c),
                                         nn.Conv2d(out_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                                         nn.ReLU(True))
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                                               stride=1, padding=0, bias=False),
                                               nn.BatchNorm2d(branch_features),
                                               nn.ReLU(True),
                                               nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, bias=False),
                                               nn.BatchNorm2d(branch_features),
                                               nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                                               nn.BatchNorm2d(branch_features),
                                               nn.ReLU(True))
        
    def forward(self, x):
        if self.stride == 1:
            x1 ,x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out
    
class ShuffeNetv2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes):
        super().__init__()
        self.stages_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self.stages_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(True))
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self.stages_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self.stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True))
        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x


def shufflenet_v2_x0_5(num_classes=1000):
    model = ShuffeNetv2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_0(num_classes=1000):
    model = ShuffeNetv2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_5(num_classes=1000):
    model = ShuffeNetv2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x2_0(num_classes=1000):
    model = ShuffeNetv2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 244, 488, 976, 2048],
                         num_classes=num_classes)

    return model