import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channel, out_channel, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Inception_block(nn.Module):
    def __init__(self, in_channel, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channel, out_1x1, kernel_size = 1)
        self.branch2 = nn.Sequential(
            conv_block(in_channel, red_3x3, kernel_size = 1),
            conv_block(red_3x3, out_3x3, kernel_size = 3, padding = 1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channel, red_5x5, kernel_size = 1),
            conv_block(red_5x5, out_5x5, kernel_size = 5, padding = 2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),
            conv_block(in_channel, out_1x1pool, kernel_size = 1)
        )

    def forward(self, x):
        # N*filter*28*28
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class GoogleNet(nn.Module):
    def __init__(self, in_channel = 3, num_classes = 1000):
        super(GoogleNet, self).__init__()
        self.conv1 = conv_block(in_channel, out_channel=64, kernel_size = 7, stride = 2, padding = 3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size = 3, stride = 1, padding = 1)
        self.max_pool2 =  nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.inception1 = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.inception3 = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4 = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception5 = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception6 = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception7 = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride = 2, padding=1)
        self.inception8 = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception9 = Inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avg_pool1 = nn.AvgPool2d(kernel_size = 7, stride=1)
        self.dropout1 = nn.Dropout2d(0.4)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.max_pool3(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.max_pool4(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.avg_pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(x)
        x = self.linear(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    model = GoogleNet()
    print(model(x).shape)

    

