import torch.nn as nn
import torch.functional as F
import torch

class encoder_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(34, 34)
        self.fc2 = nn.Linear(34, 34)
        self.fc3 = nn.Linear(34, 16)
    
    def forward(self, x):
        # 34 -> 16 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class pooling_shrink_net(nn.Module):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number):
        super(pooling_shrink_net, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        self.expand_conv = nn.Conv1d(in_features, channel, kernel_size=3, stride=2, bias=True)
        self.expand_bn = nn.BatchNorm1d(channel, momentum=0.1)
        self.shrink = nn.Conv1d(channel, out_features, 1)
        self.stage_number = stage_number
        layers = []

        for stage_index in range(0, stage_number):
            for conv_index in range(len(kernel_size_set)):
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(channel, channel, kernel_size_set[conv_index], stride_set[conv_index], dilation=1, bias=True),
                        nn.BatchNorm1d(channel, momentum=0.1)
                    )
                )

        self.stage_layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for layer in self.stage_layers:
            x = self.drop(self.relu(layer(x)))
        x = F.adaptive_max_pool1d(x, 1)
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)


class pooling_net(nn.Module):
    def __init__(self, in_features, out_features, kernel_size_set, stride_set, dilation_set, channel, stage_number):
        super(pooling_net, self).__init__()
        self.drop = nn.Dropout(p=0.25)
        self.relu = nn.LeakyReLU(inplace=True)
        self.expand_conv = nn.Conv1d(in_features, channel, kernel_size=1, stride=1, bias=True)
        self.expand_bn = nn.BatchNorm1d(channel, momentum=0.1)
        self.stage_number = stage_number
        self.conv_depth = len(kernel_size_set)
        layers = []

        for stage_index in range(0, stage_number):
            for conv_index in range(len(kernel_size_set)):
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(channel, channel, kernel_size_set[conv_index], stride_set[conv_index], dilation=1, bias=True),
                        nn.BatchNorm1d(channel, momentum=0.1)
                    )
                )

        self.shrink = nn.Conv1d(channel, out_features, kernel_size=1, stride=1, bias=True)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for stage_index in range(0, self.stage_number):
            output = 0
            for conv_index in range(self.conv_depth):
                output += F.adaptive_avg_pool1d(self.drop(self.relu(self.layers[stage_index*self.conv_depth + conv_index](x))), x.shape[-1]) 
            x = output
        x = self.shrink(x)
        return torch.transpose(x, 1, 2)