import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet_50(nn.Module):
    def __init__(self, output_layer=None):
        super(ResNet_50, self).__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.output_layer = output_layer

        self.layers = []
        for name, module in self.pretrained.named_children():
            self.layers.append((name, module))
            if name == self.output_layer:
                break

        self.net = nn.Sequential(*[module for _, module in self.layers])

    def forward(self, x):
        return self.net(x)


class Atrous_Convolution(nn.Module):
    def __init__(
        self, input_channels, kernel_size, pad, dilation_rate, output_channels=256
    ):
        super(Atrous_Convolution, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation_rate,
            bias=False,
        )

        self.batchnorm = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class Atrous_Spatial_Pyramid_Pooling(nn.Module):
    def __init__(self, in_channles, out_channles):
        super(Atrous_Spatial_Pyramid_Pooling, self).__init__()
        self.conv_1x1 = Atrous_Convolution(
            input_channels=in_channles,
            output_channels=out_channles,
            kernel_size=1,
            pad=0,
            dilation_rate=1,
        )

        self.conv_3x3_r6 = Atrous_Convolution(
            input_channels=in_channles,
            output_channels=out_channles,
            kernel_size=3,
            pad=6,
            dilation_rate=6,
        )

        self.conv_3x3_r12 = Atrous_Convolution(
            input_channels=in_channles,
            output_channels=out_channles,
            kernel_size=3,
            pad=12,
            dilation_rate=12,
        )

        self.conv_3x3_r18 = Atrous_Convolution(
            input_channels=in_channles,
            output_channels=out_channles,
            kernel_size=3,
            pad=18,
            dilation_rate=18,
        )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=in_channles,
                out_channels=out_channles,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.LayerNorm([out_channles, 1, 1]),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.final_conv = Atrous_Convolution(
            input_channels=out_channles * 5,
            output_channels=out_channles,
            kernel_size=1,
            pad=0,
            dilation_rate=1,
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3_r6 = self.conv_3x3_r6(x)
        x_3x3_r12 = self.conv_3x3_r12(x)
        x_3x3_r18 = self.conv_3x3_r18(x)
        img_pool_opt = self.image_pool(x)
        img_pool_opt = F.interpolate(
            img_pool_opt, size=x_3x3_r18.size()[2:], mode="bilinear", align_corners=True
        )
        concat = torch.cat((x_1x1, x_3x3_r6, x_3x3_r12, x_3x3_r18, img_pool_opt), dim=1)
        x_final_conv = self.final_conv(concat)
        return x_final_conv


class Deeplabv3Plus(nn.Module):
    def __init__(self, num_classes):
        super(Deeplabv3Plus, self).__init__()
        self.backbone = ResNet_50(output_layer="layer3")
        self.low_level_features = ResNet_50(output_layer="layer1")

        self.aspp = Atrous_Spatial_Pyramid_Pooling(in_channles=1024, out_channles=256)

        self.conv1x1 = Atrous_Convolution(
            input_channels=256,
            output_channels=48,
            kernel_size=1,
            dilation_rate=1,
            pad=0,
        )

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifer = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x_backbone = self.backbone(x)
        x_low_level = self.low_level_features(x)
        x_aspp = self.aspp(x_backbone)
        x_aspp_upsampled = F.interpolate(
            x_aspp, scale_factor=(4, 4), mode="bilinear", align_corners=True
        )
        x_conv1x1 = self.conv1x1(x_low_level)
        x_cat = torch.cat([x_conv1x1, x_aspp_upsampled], dim=1)
        x_3x3 = self.conv_3x3(x_cat)
        x_3x3_upscaled = F.interpolate(
            x_3x3, scale_factor=(4, 4), mode="bilinear", align_corners=True
        )
        x_out = self.classifer(x_3x3_upscaled)
        return x_out
