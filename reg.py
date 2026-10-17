import torch
from torch import nn
import torch.nn.functional as F
from functools import partial


def smooth_loss(offset_map):
    dx = torch.abs(offset_map[:, :, :, 1:] - offset_map[:, :, :, :-1])
    dy = torch.abs(offset_map[:, :, 1:, :] - offset_map[:, :, :-1, :])
    dx = dx * dx
    dy = dy * dy
    grad = torch.mean(dx) + torch.mean(dy)
    return grad


def get_initial(act_func, init_func):
    a = 0.0
    gain = 0.02
    if act_func == "leaky_relu":
        a = 0.2
    if isinstance(init_func, str):
        if init_func == "kaiming":
            act_func = "relu" if act_func is None else act_func  
            return partial(nn.init.kaiming_normal_, a=a, nonlinearity=act_func, mode='fan_in')
        elif init_func == "normal":
            return partial(nn.init.normal_, mean=0.0, std=gain)
        elif init_func == "zeros":
            return partial(torch.nn.init.normal_, mean=0.0, std=1e-5)
    elif init_func is None:
        if act_func in ["relu", "leaky_relu"]:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=act_func)
        elif act_func in ["tanh", "sigmoid"]:
            gain = torch.nn.init.calculate_gain(nonlinearity=act_func, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_func


def get_activate(act_func):
    if act_func == "relu":
        return nn.ReLU(inplace=False)
    elif act_func == "leaky_relu":
        negative_slope = 0.2
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    else:
        return None


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                      nn.InstanceNorm2d(dim),
                      nn.ReLU(True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                      nn.InstanceNorm2d(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        x = x + self.conv_block(x)  # skip connection
        return x


class ResLayer(nn.Module):
    def __init__(self, dim, n_blocks, init_func):
        super(ResLayer, self).__init__()
        res_layer = []
        for i in range(n_blocks):
            res_layer += [ResBlock(dim)]
        self.res_layer = nn.Sequential(*res_layer)

        initial = get_initial("relu", init_func)

        def init_weights(model):
            if type(model) == nn.Conv2d:
                initial(model.weight)
                if model.bias is not None:
                    model.bias.data.zero_()
            if type(model) == nn.BatchNorm2d:
                nn.init.normal_(model.weight.data, 1.0, 0.02)
                nn.init.constant_(model.bias.data, 0.0)

        self.res_layer.apply(init_weights)

    def forward(self, x):
        return self.res_layer(x)


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, bias=True, act_func="relu",
                 init_func="kaiming", use_norm=False, use_resnet=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.res_layer = ResLayer(output_nc, n_blocks=1, init_func=init_func) if use_resnet else None
        self.norm = nn.InstanceNorm2d(output_nc) if use_norm else None
        self.activate = get_activate(act_func)
        # init
        initial = get_initial(act_func, init_func)
        initial(self.conv.weight)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm2d):
            nn.init.normal_(self.norm.weight.data, 0.0, 1.0)
            nn.init.constant_(self.norm.bias.data, 0.0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        if self.res_layer is not None:
            x = self.res_layer(x)
        return x


class DownLayer(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, bias=True, act_func="leaky_relu",
                 init_func="kaiming", use_norm=False, use_resnet=True, skip=True, refine=False,
                 pool=True, pool_size=2):
        super(DownLayer, self).__init__()
        self.conv0 = ConvBlock(input_nc, output_nc, kernel_size, stride, padding, bias, act_func, init_func,
                               use_norm, use_resnet)
        self.conv1 = None
        if refine:
            self.conv1 = ConvBlock(output_nc, output_nc, kernel_size, stride, padding, bias, act_func, init_func,
                                   use_norm, use_resnet)
        self.skip = skip
        self.pool = None
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = skip = self.conv0(x)
        if self.conv1 is not None:
            x = skip = self.conv1(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.skip:
            return x, skip
        else:
            return x


# ResUNet
class ResUNet(nn.Module):
    def __init__(self, input_nc_a, input_nc_b, init_func):
        super(ResUNet, self).__init__()
        self.down_block_nc = [32, 64, 64, 64, 64, 64, 64]
        self.up_block_nc = [64, 64, 64, 64, 64, 64, 32]
        skip_nc = {}

        input_nc = input_nc_a + input_nc_b
        act_func = "leaky_relu"

        index = 1
        for output_nc in self.down_block_nc:
            setattr(self, "down_layer{}".format(index),
                    DownLayer(input_nc, output_nc, 3, 1, 1, act_func=act_func, init_func=init_func, bias=True,
                              use_resnet=True, use_norm=False))
            skip_nc["down_layer{}".format(index)] = output_nc
            input_nc = output_nc
            index += 1
        self.bottom_conv0 = ConvBlock(input_nc, 2 * input_nc, 1, 1, 0, act_func=act_func, init_func=init_func,
                                      bias=True,
                                      use_resnet=False, use_norm=False)
        self.bottom_res_layer = ResLayer(2 * input_nc, 3, init_func)
        self.bottom_conv1 = ConvBlock(2 * input_nc, input_nc, 1, 1, 0, act_func=act_func, init_func=init_func,
                                      bias=True,
                                      use_resnet=False, use_norm=False)
        index -= 1
        for output_nc in self.up_block_nc:
            setattr(self, "up_layer{}".format(index),
                    ConvBlock(input_nc + skip_nc["down_layer{}".format(index)], output_nc, 3, 1, 1, bias=True,
                              act_func=act_func, init_func=init_func, use_resnet=False, use_norm=False))
            input_nc = output_nc
            index -= 1

        self.refine = nn.Sequential(ResLayer(input_nc, 1, init_func),
                                    ConvBlock(input_nc, input_nc, 1, 1, 0, init_func=init_func,
                                              act_func=act_func, use_resnet=False, use_norm=False))
        self.output_layer = ConvBlock(input_nc, 2, 3, 1, 1, init_func="zeros", act_func=None,
                                      bias=True, use_resnet=False, use_norm=False)

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_temp = {}
        index = 1

        # down path
        while index <= len(self.down_block_nc):
            x, skip = getattr(self, "down_layer{}".format(index))(x)
            skip_temp["down_layer{}".format(index)] = skip
            index += 1
        # bottom
        x = self.bottom_conv0(x)
        x = self.bottom_res_layer(x)
        x = self.bottom_conv1(x)
        # up
        index -= 1
        while index > (len(self.down_block_nc) - len(self.up_block_nc)):
            skip = skip_temp["down_layer{}".format(index)]
            x = F.interpolate(x, (skip.size(2), skip.size(3)), mode="bilinear")
            x = torch.cat([x, skip], dim=1)
            x = getattr(self, "up_layer{}".format(index))(x)
            index -= 1
        x = self.refine(x)
        x = self.output_layer(x)
        return x


# RegNet
class RegNet(nn.Module):
    def __init__(self, height, width, input_nc_a, input_nc_b):
        """
        :param height: int -> 输入图片的尺寸
        :param width: int -> 输入图片的尺寸
        :param input_nc_a: int -> 输入图像A的channel数
        :param input_nc_b: int -> 输入图像B的channel数
        """
        super(RegNet, self).__init__()
        self.height = height
        self.width = width
        self.input_nc_a = input_nc_a
        self.input_nc_b = input_nc_b
        self.offset_map = ResUNet(self.input_nc_a, self.input_nc_b, init_func="kaiming")
        self.grid = self.get_grid()

    def get_grid(self):
        x = torch.linspace(-1.0, 1.0, self.width)
        y = torch.linspace(-1.0, 1.0, self.height)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)
        grid = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return grid

    def forward(self, img_a, img_b):
        deformations = self.offset_map(img_a, img_b)
        return deformations


# Apply Reg Result to Output Image
class ApplyRegTrans(nn.Module):
    def __init__(self):
        super(ApplyRegTrans, self).__init__()

    def forward(self, img, offset_map):
        b = offset_map.shape[0]
        h = offset_map.shape[2]
        w = offset_map.shape[3]
        size = (h, w)

        vectors = [torch.arange(0, s) for s in size]

        grids = torch.meshgrid(vectors) 
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        device = img.device
        grid = grid.repeat(b, 1, 1, 1).to(device)
        
        new_locs = grid + offset_map
        shape = offset_map.shape[2:]  # [h, w]
        
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)  # (b, c, h, w) -> (b, h, w, c)
        new_locs = new_locs[..., [1, 0]]  # (y, x) -> (x, y)
        
        img_resampled = F.grid_sample(img, new_locs, align_corners=True, padding_mode="border")
        return img_resampled


if __name__ == '__main__':
    x = torch.randn(1, 1, 256, 256)
    y = torch.randn(1, 1, 256, 256)
    reg = RegNet(256, 256, 1, 1)
    out = reg(x, y)
    print(out.shape)


