import torch
import torch.nn as nn

from collections import OrderedDict
from functools import partial

device = "cuda" if torch.cuda.is_available() else "cpu"

class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, do_scale, n_layers=1, activation_class=nn.ReLU, dropout_p=0.3, reverse=False, change_size=0):
        super(ResidualBlock, self).__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.activation_class = activation_class
        self.dropout_p = dropout_p
        self.reverse = reverse
        self.do_scale = do_scale
        self.change_size = change_size
        
        if self.reverse:
            self.conv_class = nn.ConvTranspose2d
            self.scale_class = partial(
                nn.ConvTranspose2d,
                in_channels=self.c_out,
                out_channels=self.c_out,
                stride=2,
            )
            self.conv_name_prefix = "deconv"
            self.scale_name = "upscale"
            self.change_name = "add_size"
        else:
            self.conv_class = nn.Conv2d
            self.scale_class = nn.MaxPool2d
            self.conv_name_prefix = "conv"
            self.scale_name = "downscale"
            self.change_name = "sub_size"
        
        layers = [
            nn.Sequential(OrderedDict([
                (
                    self.conv_name_prefix,
                    self.conv_class(
                        in_channels = self.c_in if i == 0 else self.c_out,
                        out_channels = self.c_out,
                        kernel_size = self.kernel_size,
                        padding = (self.kernel_size - 1) // 2,
                        bias=False,
                    ),
                ),
                ("bnorm", nn.BatchNorm2d(self.c_out)),
                ("act", self.activation_class()),
                ("drop", nn.Dropout2d(self.dropout_p)),
            ])) for i in range(self.n_layers)
        ]
        
        self.layers = nn.Sequential(OrderedDict([(f"layer_{i}", layer) for i, layer in enumerate(layers)]))
        
        self.adapt_residual = nn.Identity() if self.c_in == self.c_out else self.conv_class(self.c_in, self.c_out, 1)
        
        scale = [
            (self.change_name, self.conv_class(self.c_out, self.c_out, self.change_size + 1) if self.change_size > 0 else nn.Identity()),
            (self.scale_name, self.scale_class(kernel_size=2) if self.do_scale else nn.Identity()),
        ]
        self.scale = nn.Sequential(OrderedDict(scale))


    def forward(self, x):
        return self.scale(self.layers(x) + self.adapt_residual(x))

class Discriminator(nn.Module):
    def __init__(self, channels=8, layers=3, kernel_size=3, image_size=256, hidden_size=100):
        super(Discriminator, self).__init__()
        self.blocks = []
        self.blocks += [("initial", ResidualBlock(3, channels, kernel_size, do_scale=False))]

        last_filters_mult = 1
        for i in range(1, layers + 1):
            filters_mult = 2**i
            self.blocks += [(f"block_{i}", ResidualBlock(channels * last_filters_mult, channels * filters_mult, kernel_size, do_scale=True))]
            image_size //= 2
            last_filters_mult = filters_mult

        self.blocks += [("final", ResidualBlock(channels * last_filters_mult, 1, kernel_size, do_scale=True))]
        image_size //= 2
        self.blocks = nn.Sequential(OrderedDict(self.blocks))

        self.pipeline = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("linear1", nn.Linear(image_size * image_size, hidden_size)),
            ("relu", nn.ReLU()),
            ("linear2", nn.Linear(hidden_size, 1)),
            ("sigmoid", nn.Sigmoid()),
        ]))
    
    def forward(self, images):
        result = self.pipeline(self.blocks(images))
        return result

class Generator(nn.Module):
    def __init__(self, channels=8, kernel_size=3, scales=2, blocks=3):
        super(Generator, self).__init__()
        self.blocks = []
        self.blocks += [("initial", ResidualBlock(3, channels, kernel_size, do_scale=False))]

        last_filters_mult = 1
        for i in range(1, scales + 1):
            filters_mult = 2 * last_filters_mult
            self.blocks += [(f"downscale_{i}", ResidualBlock(channels * last_filters_mult, channels * filters_mult, kernel_size, do_scale=True))]
            last_filters_mult = filters_mult
        
        for i in range(1, blocks + 1):
            self.blocks += [(f"block_{i}", ResidualBlock(channels * last_filters_mult, channels * last_filters_mult, kernel_size, do_scale=False))]
            last_filters_mult = filters_mult
        
        for i in range(1, scales + 1):
            filters_mult = last_filters_mult // 2
            self.blocks += [(f"upscale_{i}", ResidualBlock(channels * last_filters_mult, channels * filters_mult, kernel_size, reverse=True, do_scale=True))]
            last_filters_mult = filters_mult
        
        self.blocks += [("final", ResidualBlock(channels * last_filters_mult, 3, kernel_size, do_scale=False))]
        self.blocks += [("de_pixelize", nn.Conv2d(3, 3, kernel_size * 2 + 1, padding=kernel_size))]

        self.blocks = nn.Sequential(OrderedDict(self.blocks))
        self.norm = nn.Tanh()
    
    def forward(self, images):
        return self.norm(self.blocks(images))

class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.gen_a_b = Generator()
        self.gen_b_a = Generator()
        self.discr_a = Discriminator()
        self.discr_b = Discriminator()
    
    def forward(self, images_a, images_b):
        pred_true_a = self.discr_a(images_a)
        pred_true_b = self.discr_b(images_b)
        
        (images_a, images_b) = self.gen_b_a(images_b), self.gen_a_b(images_a)

        pred_fake_a = self.discr_a(images_a)
        pred_fake_b = self.discr_b(images_b)

        (cycled_a, cycled_b) = self.gen_b_a(images_b), self.gen_a_b(images_a)

        return pred_true_a, pred_true_b, pred_fake_a, pred_fake_b, cycled_a, cycled_b

    def get_fake_a(self, images_b):
        with torch.no_grad():
            return self.gen_a_b(images_b)
    
    def get_fake_b(self, images_a):
        with torch.no_grad():
            return self.gen_b_a(images_a)
