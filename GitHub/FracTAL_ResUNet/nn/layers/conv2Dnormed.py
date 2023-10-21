import torch

class Conv2DNormed(torch.nn.Module):
    def __init__(self, channels, kernel_size, stride=(1,1),
                 padding=(0, 0), dilation=(1, 1), activation=None,
                 weight_initializer=None, _norm_type='BatchNorm', norm_groups=None, axis=1, groups=1,
                 ):
        super().__init__()

        self.conv2d = torch.nn.LazyConv2d(channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, bias=False,
                                      groups=groups)

        if _norm_type == 'BatchNorm':
            self.norm_layer = torch.nn.BatchNorm2d(channels)
        else:
            self.norm_layer = torch.nn.GroupNorm(num_groups=norm_groups, num_channels=channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm_layer(x)
        return x
