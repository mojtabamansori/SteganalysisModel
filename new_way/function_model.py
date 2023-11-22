import torch.nn as nn
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiScaleAvgPool2d(nn.Module):
    def __init__(self, kernel_sizes, strides):
        super(MultiScaleAvgPool2d, self).__init__()
        self.pool_layers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
            for kernel_size, stride in zip(kernel_sizes, strides)
        ])

    def forward(self, x):
        pooled_outputs = [pool_layer(x) for pool_layer in self.pool_layers]
        max_height = max(output.size(2) for output in pooled_outputs)
        max_width = max(output.size(3) for output in pooled_outputs)
        padded_outputs = [
            F.pad(output, (0, max_width - output.size(3), 0, max_height - output.size(2)))
            for output in pooled_outputs
        ]

        min_channels = min(output.size(1) for output in padded_outputs)
        padded_outputs = [output[:, :min_channels, :, :] for output in padded_outputs]

        return torch.cat(padded_outputs, dim=1)



class SteganalysisModel(nn.Module):
    def __init__(self, num_classes=6):
        super(SteganalysisModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride= 1, padding=2)
        self.activation1 = nn.Tanh()
        self.batch_norm1 = nn.BatchNorm2d(30, momentum=0.2, eps=0.001)

        self.depthwise_conv = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1, groups=30, bias=False)
        self.activation2 = nn.LeakyReLU(negative_slope= -0.1)
        self.batch_norm2 = nn.BatchNorm2d(30, momentum=0.2, eps=0.001)

        self.depthwise_conv2 = nn.Conv2d(30, 30, kernel_size=5, stride=1, padding=2, groups=30, bias=False)
        self.activation3 = nn.LeakyReLU(negative_slope= -0.1)
        self.batch_norm3 = nn.BatchNorm2d(60, momentum=0.2, eps=0.001)

        self.conv2 = nn.Conv2d(60, 60, kernel_size=5, stride=1, padding=2)
        self.activation4 = nn.LeakyReLU(negative_slope= -0.1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(60, 60, kernel_size=5, stride=1, padding=2)
        self.activation41 = nn.LeakyReLU(negative_slope= -0.1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.depthwise_conv3 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1, groups=30, bias=False)
        self.activation6 = nn.LeakyReLU(negative_slope= -0.1)
        self.batch_norm4 = nn.BatchNorm2d(60, momentum=0.2, eps=0.001)

        self.depthwise_conv4 = nn.Conv2d(60, 30, kernel_size=3, stride=1, padding=1, groups=30, bias=False)
        self.activation7 = nn.LeakyReLU(negative_slope= -0.1)
        self.batch_norm5 = nn.BatchNorm2d(30, momentum=0.2, eps=0.001)

        self.conv4 = nn.Conv2d(30, 30, kernel_size=5, stride=1, padding=2)
        self.activation8 = nn.LeakyReLU(negative_slope= -0.1)

        self.conv5 = nn.Conv2d(30, 30, kernel_size=5, stride=1, padding=2)
        self.activation9 = nn.LeakyReLU(negative_slope= -0.1)

        self.batch_norm7 = nn.BatchNorm2d(60, momentum=0.2, eps=0.001)

        # Replace nn.AvgPool2d with MultiScaleAvgPool2d
        self.avgpool10 = MultiScaleAvgPool2d(kernel_sizes=[2, 3, 4], strides=[2, 2, 2])

        self.conv6 = nn.Conv2d(60, 30, kernel_size=5, stride=1, padding=2)
        self.activation10 = nn.LeakyReLU(negative_slope= -0.1)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(30, 30, kernel_size=5, stride=1, padding=2)
        self.activation11 = nn.LeakyReLU(negative_slope= -0.1)
        self.conv8 = nn.Conv2d(30, 30, kernel_size=5, stride=1, padding=2)
        self.activation12 = nn.LeakyReLU(negative_slope= -0.1)

        out_channels_fc1 = 90 * 17 * 17

        # Add fully connected layers
        self.fc1 = nn.Linear(out_channels_fc1, 512)
        self.activation_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.activation_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        out_out_1 = self.activation1(x)
        out_out_2 = self.batch_norm1(out_out_1)

        x1 = self.depthwise_conv(out_out_2)
        out_out_3 = self.activation2(x1)
        out_out_4 = self.batch_norm2(out_out_3)

        out_out_4 = self.depthwise_conv2(out_out_4)
        out_out_5 = self.activation3(out_out_4)
        out_out_6 = torch.cat((out_out_5, out_out_2), dim=1)
        out_out_6 = self.batch_norm3(out_out_6)

        out_out_7 = self.conv2(out_out_6)
        out_out_7 = self.activation4(out_out_7)

        out_out_8 = self.avgpool(out_out_7)

        out_out_9 = self.conv3(out_out_8)
        out_out_9 = self.activation41(out_out_9)

        out_out_10 = self.avgpool2(out_out_9)

        out_out_11 = self.depthwise_conv3(out_out_10)
        out_out_11 = self.activation6(out_out_11)
        out_out_12 = self.batch_norm4(out_out_11)

        out_out_13 = self.depthwise_conv4(out_out_12)
        out_out_13 = self.activation7(out_out_13)

        out_out_14 = self.batch_norm5(out_out_13)

        out_out_15 = self.conv4(out_out_14)
        out_out_15 = self.activation8(out_out_15)

        out_out_16 = self.conv5(out_out_15)
        out_out_16 = self.activation9(out_out_16)

        out_out_17 = torch.cat((out_out_16, out_out_14), dim=1)
        out_out_17 = self.batch_norm7(out_out_17)

        out_out_18 = self.conv6(out_out_17)
        out_out_18 = self.activation10(out_out_18)

        out_out_19 = self.avgpool3(out_out_18)

        out_out_20 = self.conv7(out_out_19)
        out_out_20 = self.activation11(out_out_20)

        out_out_21 = self.conv8(out_out_20)
        out_out_21 = self.activation11(out_out_21)

        # Change the following line to use multi-scale average pooling
        out_out_22 = self.avgpool10(out_out_21)

        x_flat = out_out_22.view(out_out_22.size(0), -1)

        out_out_23 = self.fc1(x_flat)
        out_out_23 = self.activation_fc1(out_out_23)
        out_out_23 = self.fc2(out_out_23)
        out_out_23 = self.activation_fc2(out_out_23)
        out_out_23 = self.fc3(out_out_23)
        out_out_24 = F.softmax(out_out_23, dim=1)

        return out_out_24
