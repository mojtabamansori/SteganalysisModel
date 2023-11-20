import torch.nn as nn
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SteganalysisModel(nn.Module):
    def __init__(self, num_classes=6):
        super(SteganalysisModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=1, padding=2)
        self.activation1 = nn.Tanh()
        self.batch_norm1 = nn.BatchNorm2d(30)

        self.depthwise_conv = nn.Conv2d(30, 30, kernel_size=1, stride=1, padding=1, groups=30, bias=False)
        self.activation2 = nn.LeakyReLU(negative_slope=0.01)
        self.batch_norm2 = nn.BatchNorm2d(30)

        self.depthwise_conv2 = nn.Conv2d(30, 30, kernel_size=5, stride=1, padding=1, groups=30, bias=False)
        self.activation3 = nn.LeakyReLU(negative_slope=0.01)
        self.batch_norm3 = nn.BatchNorm2d(60)

        self.conv3 = nn.Conv2d(60, 60, kernel_size=5, stride=1, padding=2)
        self.activation4 = nn.LeakyReLU(negative_slope=0.01)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=2)
        self.activation5 = nn.LeakyReLU(negative_slope=0.01)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.depthwise_conv3 = nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1, groups=30, bias=False)
        self.activation6 = nn.LeakyReLU(negative_slope=0.01)
        self.batch_norm4 = nn.BatchNorm2d(60)

        self.depthwise_conv4 = nn.Conv2d(60, 30, kernel_size=3, stride=1, padding=1, groups=30, bias=False)
        self.activation7 = nn.LeakyReLU(negative_slope=0.01)
        self.batch_norm5 = nn.BatchNorm2d(30)

        self.conv5 = nn.Conv2d(30, 30, kernel_size=5, stride=1, padding=2)
        self.activation8 = nn.LeakyReLU(negative_slope=0.01)
        self.batch_norm6 = nn.BatchNorm2d(30)

        self.conv6 = nn.Conv2d(30, 30, kernel_size=5, stride=1, padding=2)
        self.activation9 = nn.LeakyReLU(negative_slope=0.01)
        self.batch_norm7 = nn.BatchNorm2d(30)

        self.avgpool10 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Add fully connected layers
        self.fc1 = nn.Linear(30 * 32 * 32, 512)  # Adjust input size accordingly
        self.activation_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.activation_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        out_conv_1 = self.activation1(x)
        out_bn_1 = self.batch_norm1(out_conv_1)

        x1 = self.depthwise_conv(out_bn_1)
        out_dwconv_1 = self.activation2(x1)
        out_bn_2 = self.batch_norm2(out_dwconv_1)

        x1 = self.depthwise_conv2(out_bn_2)
        out_dwconv_2 = self.activation3(x1)

        input_bn_2 = torch.cat((out_dwconv_2, out_bn_1), dim=1)
        output_bn_2 = self.batch_norm3(input_bn_2)

        x3 = self.conv3(output_bn_2)
        out_conv_2 = self.activation4(x3)
        out_avgpool_1 = self.avgpool(out_conv_2)

        out_avgpool_1 = self.conv3(out_avgpool_1)
        out_conv_3 = self.activation5(out_avgpool_1)
        out_avgpool_2 = self.avgpool2(out_conv_3)

        out_avgpool_2 = self.depthwise_conv3(out_avgpool_2)
        out_dwconv_3 = self.activation6(out_avgpool_2)
        out_bn_4 = self.batch_norm3(out_dwconv_3)

        out_avgpool_2 = self.depthwise_conv4(out_bn_4)
        out_dwconv_4 = self.activation7(out_avgpool_2)
        out_bn_5 = self.batch_norm5(out_dwconv_4)

        out_conv_5 = self.conv5(out_bn_5)
        out_conv_5 = self.activation8(out_conv_5)
        out_conv_5 = self.batch_norm6(out_conv_5)

        out_conv_6 = self.conv6(out_conv_5)
        out_conv_6 = self.activation9(out_conv_6)
        out_conv_6 = self.batch_norm7(out_conv_6)

        out_avgpool_10 = self.avgpool10(out_conv_6)

        # Flatten the output before fully connected layers
        x_flat = out_avgpool_10.view(out_avgpool_10.size(0), -1)

        # Fully connected layers
        x_fc1 = self.fc1(x_flat)
        x_fc1 = self.activation_fc1(x_fc1)
        x_fc2 = self.fc2(x_fc1)
        x_fc2 = self.activation_fc2(x_fc2)
        x_fc3 = self.fc3(x_fc2)
        x_softmax = F.softmax(x_fc3, dim=1)

        return x_softmax
