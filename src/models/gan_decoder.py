# Progressive Growing GAN Decoder based on this paper: https://arxiv.org/abs/1710.10196


# GAN Generator component of the Progressive Growing GAN
# Takes in 512 feature latent vector from UNet-based encoder and upscales it to 2048x1024x3 texture map

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

latent_size = 512
starting_resolution = 4

def nf(stage, fmap_base, fmap_decay, fmap_max):
            # Number of feature maps at the given stage
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


def pixel_norm(x, epsilon=1e-8):
    # Normalize the feature vector per pixel
    return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + epsilon)


def update_smoothed_generator(Gs, G, beta=0.999):
    with torch.no_grad():
        param_dict_G = dict(G.named_parameters())
        for name, param in Gs.named_parameters():
            param.data = beta * param.data + (1.0 - beta) * param_dict_G[name].data


class EqualizedLearningRateLayer(nn.Module):
    # Custom layer for equalized learning rate
    def __init__(self, layer, gain=np.sqrt(2)):
        super().__init__()
        self.layer = layer
        self.scale = np.sqrt(gain / np.sqrt(np.prod(layer.weight.shape[1:])))
        nn.init.normal_(self.layer.weight)
        if self.layer.bias is not None:
            nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        return self.layer(x) * self.scale
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, gain=np.sqrt(2), use_wscale=True, use_leaky_relu=True):
        super().__init__()
        layers = []

        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        if use_wscale:
            conv1 = EqualizedLearningRateLayer(conv1, gain=gain)
        layers.append(conv1)
        if use_leaky_relu:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU())

        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        if use_wscale:
            conv2 = EqualizedLearningRateLayer(conv2, gain=gain)
        layers.append(conv2)
        if use_leaky_relu:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)
    

class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels=3, gain=1.0, use_wscale=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if use_wscale:
            self.conv = EqualizedLearningRateLayer(self.conv, gain=gain)
    
    def forward(self, x):
        return self.conv(x)


class GAN_Generator(nn.Module):
    def __init__(self, num_channels=3, resolution=2048, fmap_base=8192, fmap_decay=1.0, fmap_max=512, latent_size=512, normalize_latents=False, use_wscale=True, use_leaky_relu=True):
        super().__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution))
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.latent_size = latent_size if latent_size is not None else min(fmap_base, fmap_max)
        self.normalize_latents = normalize_latents
        
        self.layers = nn.ModuleDict()
        self.to_rgb_layers = nn.ModuleDict()

        # Initial block for 8x4 resolution
        self.layers['512'] = nn.Sequential(
            EqualizedLearningRateLayer(nn.Linear(latent_size, 
                                                 nf(1, self.fmap_base, self.fmap_decay, self.fmap_max) * 16 * 4 * 8), 
                                                 gain=np.sqrt(2)/4),
            nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU()
        )
        self.to_rgb_layers['512'] = ToRGB(nf(1, self.fmap_base, self.fmap_decay, self.fmap_max), 
                                          num_channels, 
                                          use_wscale=use_wscale)

        # Conv layers for each resolution
        current_width, current_height = 8, 4
        for res in range(3, self.resolution_log2 + 1):
            layer_name = f'{current_width}x{current_height}'
            print('Layer name:', layer_name)
            print('nf:', nf(res-1, self.fmap_base, self.fmap_decay, self.fmap_max))
            self.layers[layer_name] = ConvBlock(nf(res-1, self.fmap_base, self.fmap_decay, self.fmap_max), 
                                                nf(res, self.fmap_base, self.fmap_decay, self.fmap_max), 
                                                use_wscale=use_wscale, 
                                                use_leaky_relu=use_leaky_relu)
            self.to_rgb_layers[layer_name] = ToRGB(nf(res-1, self.fmap_base, self.fmap_decay, self.fmap_max), 
                                                   num_channels, 
                                                   use_wscale=use_wscale)
            current_width *= 2
            current_height *= 2
        
        
    def forward(self, latents_in):
        x = latents_in
        if self.normalize_latents:
            x = pixel_norm(x)

        # Transform input into a 8x4 feature map
        x = self.layers['512'](x)
        x = x.view(-1, self.fmap_max, 4, 8)
        current_width, current_height = 8, 4
        for res in range(3, self.resolution_log2 + 1):
            layer_name = f'{current_width}x{current_height}'
            print('Layer name:', layer_name)
            print(x.shape)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.layers[layer_name](x)
            if res == self.resolution_log2:
                x = self.to_rgb_layers[layer_name](x)
            current_width *= 2
            current_height *= 2
        
        return x
    

# GAN Discriminator component of the Progressive Growing GAN
# Takes in 2048x1024x3 texture map and downscales it to a 512 feature latent vector

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gain=np.sqrt(2), use_wscale=True):
        super(Conv2D, self).__init__()
        self.use_wscale = use_wscale
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * gain)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if self.use_wscale:
            wscale = torch.sqrt(torch.mean(self.weight.data ** 2))
            return F.conv2d(x, self.weight / wscale, self.bias, padding=1)
        else:
            return F.conv2d(x, self.weight, self.bias, padding=1)

class MinibatchStddevLayer(nn.Module):
    def __init__(self, group_size=4, eps=1e-8):
        super(MinibatchStddevLayer, self).__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = min(self.group_size, N)  # Number of groups
        F = C // G
        x = x.view(G, -1, F, H, W)
        mean = torch.mean(x, dim=0, keepdim=True)
        y = torch.mean((x - mean) ** 2, dim=0)
        y = torch.sqrt(y + self.eps)
        y = y.mean(dim=[2, 3, 4], keepdims=True).squeeze(4).squeeze(3).squeeze(2)
        y = y.repeat(G, 1, H, W)
        return torch.cat([x.view(N, C, H, W), y], dim=1)
    
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_wscale=True, use_leaky_relu=True):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = Conv2D(in_channels, in_channels, 3, use_wscale=use_wscale)
        self.conv2 = Conv2D(in_channels, out_channels, 3, use_wscale=use_wscale)
        self.act = nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2)
        return x

class FromRGB(nn.Module):
    def __init__(self, in_channels, out_channels, use_wscale=True, use_leaky_relu=True):
        super(FromRGB, self).__init__()
        self.conv = Conv2D(in_channels, out_channels, 1, use_wscale=use_wscale)
        self.act = nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))
    
class GAN_Discriminator(nn.Module):
    def __init__(self, num_channels=3, resolution=(2048, 1024), fmap_base=8192, fmap_decay=1.0, fmap_max=512, use_wscale=True, use_leaky_relu=True):
        super(GAN_Discriminator, self).__init__()
        
        self.resolution_log2_w = int(np.log2(resolution[0]))
        self.resolution_log2_h = int(np.log2(resolution[1]))
        
        self.from_rgb = FromRGB(num_channels, nf(self.resolution_log2_h-1, fmap_base, fmap_decay, fmap_max), use_wscale, use_leaky_relu)

        self.blocks = nn.ModuleList()
        min_res_log2 = min(self.resolution_log2_w, self.resolution_log2_h)
        for res in range(min_res_log2, 2, -1):
            in_channels = nf(res - 1, fmap_base, fmap_decay, fmap_max)
            out_channels = nf(res - 2, fmap_base, fmap_decay, fmap_max)
            self.blocks.append(DiscriminatorBlock(in_channels, out_channels, use_wscale, use_leaky_relu))

        self.final_block = nn.Sequential(
            MinibatchStddevLayer(),
            Conv2D(nf(1, fmap_base, fmap_decay, fmap_max) + 1, nf(1, fmap_base, fmap_decay, fmap_max), 3, use_wscale=use_wscale),
            nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU(),
            nn.Flatten(),
            nn.Linear(nf(1, fmap_base, fmap_decay, fmap_max) * 4 * 4, 1)
        )

    def forward(self, images):
        x = self.from_rgb(images)
        print(x.shape)
        for block in self.blocks:
            x = block(x)
            print(x.shape)
        return self.final_block(x)