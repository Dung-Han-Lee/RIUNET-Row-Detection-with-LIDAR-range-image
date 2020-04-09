import torch.nn as nn
import torch
import torch.nn.functional as F

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, kernel, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel, stride=kernel)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]

        #print("input1 shape", inputs1.shape)
        outputs1 = F.pad(inputs1, padding)
  
        #print(outputs1.shape, outputs2.shape)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=2, is_deconv=True, in_channels=2, is_batchnorm=True
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,4))

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,4))

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2,4))

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], (2,4), self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], (2,4), self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], (2,4), self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], (2,2), self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        #print("maxpool1", maxpool1.shape)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        #print("maxpool2", maxpool2.shape)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        #print("maxpool3", maxpool3.shape)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        #print("maxpool4", maxpool4.shape)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        #print("up4", up4.shape)

        up3 = self.up_concat3(conv3, up4)
        #print("up3", up3.shape)
        up2 = self.up_concat2(conv2, up3)
        #print("up2", up2.shape)
        up1 = self.up_concat1(conv1, up2)
        #print("up1", up1.shape)

        final = self.final(up1)

        return final

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)