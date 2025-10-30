import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

# Define the model

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class CrossScaleBottleneck(nn.Module):
    expansion = 4  # Output channels = input channels * expansion

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=True, norm_layer=True):
        # `scales`: number of feature groups used in the residual block
        # `groups`: number of 3x3 convolutional layers
        # `se`: whether to use SE module
        # `norm_layer`: whether to use BatchNorm
        super(CrossScaleBottleneck, self).__init__()

        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer:
            norm_layer = nn.BatchNorm2d

        bottleneck_planes = groups * planes
        self.scales = scales
        self.stride = stride
        self.downsample = downsample
        # First 1x1 conv layer (may reduce spatial size in second layer)
        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, stride=stride)
        self.bn1 = norm_layer(bottleneck_planes)
        # 3x3 conv layers: total of (scales - 1) layers
        self.conv2 = nn.ModuleList([
            nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales,
                      kernel_size=3, stride=1, padding=1, groups=groups)
            for _ in range(scales - 1)
        ])
        self.bn2 = nn.ModuleList([
            norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)
        ])
        # Final 1x1 conv layer to restore channel dimension
        self.conv3 = nn.Conv2d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # SE module
        self.se = SEModule(planes * self.expansion) if se else None

    def forward(self, x):
        identity = x

        # First 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Cross-scale 3x3 conv structure
        xs = torch.chunk(out, self.scales, dim=1)  # Split into `scales` chunks
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, dim=1)

        # Final 1x1 conv
        out = self.conv3(out)
        out = self.bn3(out)

        # Apply SE module if enabled
        if self.se is not None:
            out = self.se(out)

        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class CrossScaleBlock(nn.Module):
    def __init__(self, layers, num_classes, width=16, scales=4, groups=1,
                 zero_init_residual=True, se=True, norm_layer=True):
        super(CrossScaleBlock, self).__init__()
        if norm_layer:
            norm_layer = nn.BatchNorm2d
        # Channel dimensions: [64, 128, 256, 512]
        planes = [int(width * scales * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        # Adjust input conv layer for input shape [1, 1400] (treated as 2D)
        self.conv1 = nn.Conv2d(1, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Four residual stages
        self.layer1 = self._make_layer(
            CrossScaleBottleneck, planes[0], layers[0], stride=1, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer2 = self._make_layer(
            CrossScaleBottleneck, planes[1], layers[1], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(
            CrossScaleBottleneck, planes[2], layers[2], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer4 = self._make_layer(
            CrossScaleBottleneck, planes[3], layers[3], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3] * CrossScaleBottleneck.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, CrossScaleBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, scales=4, groups=1, se=True, norm_layer=True):
        if norm_layer:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample, stride=stride,
                            scales=scales, groups=groups, se=se, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scales=scales,
                                groups=groups, se=se, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = nn.functional.softmax(logits, dim=1)

        return probas


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1,
                               kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extract spatial info by channel-wise max and avg pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Concatenate and generate spatial attention map
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        out = self.sigmoid(out)
        return out


# Trim excess padding in temporal convolution
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs must have shape (N, C_in, L_in)"""
        inputs = inputs.squeeze(-1)
        y1 = self.tcn(inputs)  # Input shape: (N, C, L)
        o = self.linear(y1[:, :, -1])
        o = F.log_softmax(o, dim=1)
        return o


class Classifier(nn.Module):
    def __init__(self, input_size, output_size=8):  # 8-class classification
        super(Classifier, self).__init__()

        # Adjust conv layer for input shape [1, 1400]
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=64,
                               kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.spatial = SpatialAttention()
        # Replace with CrossScaleBlock for 8-class output
        self.cross_scale_1 = CrossScaleBlock([2, 2, 2, 2], num_classes=8, width=16,
                                             scales=4, groups=1, zero_init_residual=True, se=True, norm_layer=True)
        self.tcn_1 = TCN(input_size=1, output_size=8, num_channels=[1, 2, 4, 8], kernel_size=5, dropout=0.5)
        self.cross_scale_2 = CrossScaleBlock([2, 2, 2, 2], num_classes=8, width=16,
                                             scales=4, groups=1, zero_init_residual=True, se=True, norm_layer=True)
        self.tcn_2 = TCN(input_size=1, output_size=8, num_channels=[1, 2, 4, 8], kernel_size=3, dropout=0.5)

        self.multihead_crossatttion = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)

        self.fc = nn.Linear(in_features=32, out_features=8)  # Final 8-class output

    def forward(self, x):
        # Reshape input [1, 1400] to (N, 1, 1, 1400) for 2D conv
        x = x.view(x.size(0), 1, 1, -1)
        out = self.conv1(x)
        out = self.spatial(out) * out  # Apply spatial attention

        # Prepare input for TCN and CrossScaleBlock
        tcn_in = out.squeeze(1)  # TCN expects (N, C, L)
        tcn_out_1 = self.tcn_1(tcn_in)
        cross_scale_out_1 = self.cross_scale_1(out)

        tcn_out_2 = self.tcn_2(tcn_in)
        cross_scale_out_2 = self.cross_scale_2(out)

        # Feature concatenation
        out_1 = torch.cat([cross_scale_out_1, tcn_out_1], dim=1)
        out_2 = torch.cat([cross_scale_out_2, tcn_out_2], dim=1)

        # Multi-head cross-attention
        out_layer_1, _ = self.multihead_crossatttion(torch.unsqueeze(out_2, dim=1),
                                                     torch.unsqueeze(out_2, dim=1),
                                                     torch.unsqueeze(out_1, dim=1))
        out_layer_2, _ = self.multihead_crossatttion(torch.unsqueeze(out_1, dim=1),
                                                     torch.unsqueeze(out_1, dim=1),
                                                     torch.unsqueeze(out_2, dim=1))

        # Feature fusion and final classification
        out = torch.cat([out_layer_1, out_layer_2], dim=2).squeeze(1)
        out = self.fc(out)

        return out