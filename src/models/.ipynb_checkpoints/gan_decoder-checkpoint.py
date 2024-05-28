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

class SimpleUpscaleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, gain=np.sqrt(2), use_wscale=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gain = gain
        self.use_wscale = use_wscale

        # Define weight and bias for a 3x3 convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Calculate scaling factor
        if self.use_wscale:
            scale = self.gain / np.sqrt(self.in_channels * 3 * 3)
        else:
            scale = self.gain
        weight = self.weight * scale
        
        # Upsample the input using nearest neighbor interpolation
        x_upsampled = F.interpolate(x, scale_factor=2, mode='nearest')
        
        # Perform convolution with kernel size 3 and padding 1
        output = F.conv2d(x_upsampled, weight, self.bias, stride=1, padding=1)
        return output
    
# Not working as intended (channels)
class FusedUpscaleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gain=np.sqrt(2), use_wscale=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.gain = gain
        self.use_wscale = use_wscale

        # Define weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if self.use_wscale:
            scale = self.gain / np.sqrt(self.in_channels * (self.kernel_size ** 2))
        else:
            scale = self.gain
        weight = self.weight * scale
        
        # Padding to ensure the output size is doubled
        weight = F.pad(weight, [1, 1, 1, 1], mode='constant', value=0)
        
        # Upsample the weight using bilinear interpolation
        weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] + weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]) / 4
        
        # Perform convolution
        output = F.conv_transpose2d(x, weight, self.bias, stride=2, padding=1)
        return output


class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels=3, gain=1.0, use_wscale=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if use_wscale:
            self.conv = EqualizedLearningRateLayer(self.conv, gain=gain)
    
    def forward(self, x):
        return self.conv(x)
    

class AlphaBlendLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, low_res, high_res, alpha):
        return low_res * (1.0 - alpha) + high_res * alpha


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
        self.use_wscale = use_wscale
        self.use_leaky_relu = use_leaky_relu
        
        self.layers = nn.ModuleDict()
        self.to_rgb_layers = nn.ModuleDict()
        self.alpha_blend_layers = nn.ModuleDict()

        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.normalize_latents = normalize_latents

        init_modules = [
            nn.Linear(latent_size, nf(1, fmap_base, fmap_decay, fmap_max) * 4 * 8),
            nn.Unflatten(1, (nf(1, fmap_base, fmap_decay, fmap_max), 4, 8)),
            nn.Conv2d(nf(2, fmap_base, fmap_decay, fmap_max), nf(2, fmap_base, fmap_decay, fmap_max), 3, padding=1),
        ]
        if use_wscale:
            init_modules[0] = EqualizedLearningRateLayer(init_modules[0], gain=np.sqrt(2)/4)
            init_modules[2] = EqualizedLearningRateLayer(init_modules[2])
        # if normalize_latents:
            # init_modules.append(PixelNorm())
        init_modules.insert(2, nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU())
        init_modules.append(nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU())
        print(init_modules)
        self.layers['8x4'] = nn.Sequential(*init_modules)
        self.to_rgb_layers['8x4'] = ToRGB(nf(2, self.fmap_base, self.fmap_decay, self.fmap_max), 
                                          num_channels, 
                                          use_wscale=use_wscale)
    
    def add_block(self, resolution, device):
        resolution_log2_w = int(np.log2(resolution[0]))
        resolution_log2_h = int(np.log2(resolution[1]))
        resolution_log2 = min(resolution_log2_w, resolution_log2_h)
        in_channels = nf(resolution_log2-1, self.fmap_base, self.fmap_decay, self.fmap_max)
        out_channels = nf(resolution_log2, self.fmap_base, self.fmap_decay, self.fmap_max)
        print(f'Resolution: {resolution}, In Channels: {in_channels}, Out Channels: {out_channels}')
        
        new_to_rgb = ToRGB(out_channels, self.num_channels, use_wscale=True).to(device)
        new_upscale_conv = SimpleUpscaleConv2d(in_channels, out_channels, 3, use_wscale=True)
        modules = []
        modules.append(new_upscale_conv)
        if self.use_leaky_relu:
            modules.append(nn.LeakyReLU(0.2))
        else:
            modules.append(nn.ReLU())
        
        if self.use_wscale:
            modules[0] = EqualizedLearningRateLayer(modules[0])
        
        new_block = nn.Sequential(*modules).to(device)

        self.layers[f'{resolution[0]}x{resolution[1]}'] = new_block
        self.to_rgb_layers[f'{resolution[0]}x{resolution[1]}'] = new_to_rgb
        self.alpha_blend_layers[f'{resolution[0]}x{resolution[1]}'] = AlphaBlendLayer().to(device)
        
    def forward(self, latents_in, alpha=1.0):
        x = latents_in
        print('Latents:', x.shape)
        if self.normalize_latents:
            x = pixel_norm(x)
        
        print(self.layers.keys())
        for i, layer_name in enumerate(self.layers.keys()):
            if alpha < 1.0 and len(self.layers) > 1 and i >= len(self.layers) - 1:
                previous_layer_name = list(self.layers.keys())[i-1]
                print('Performing alpha blending...')
                print('Input shape:', x.shape)
                x_low = self.layers[layer_name](x)
                print('X_low output shape', x_low.shape)
                x_high = F.interpolate(x, scale_factor=2, mode='nearest')
                print('X_high output shape', x_high.shape)
                x_low = self.to_rgb_layers[layer_name](x_low)
                x_high = self.to_rgb_layers[previous_layer_name](x_high)
                current_rgb = self.alpha_blend_layers[layer_name](x_low, x_high, alpha)
                print(current_rgb.shape)
                return current_rgb

            print('Layer name:', layer_name)
            print('Input shape:', x.shape)
            
            x = self.layers[layer_name](x)
            print('Output shape:', x.shape)
        
        current_rgb = self.to_rgb_layers[layer_name](x)
        return current_rgb
    

# GAN Discriminator component of the Progressive Growing GAN
# Takes in 2048x1024x3 texture map and downscales it to a 512 feature latent vector

class DiscAlphaBlendLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, high_res, low_res, alpha):
        # Blend the high resolution and low resolution outputs based on alpha
        return high_res * alpha + low_res * (1.0 - alpha)


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, gain=np.sqrt(2), use_wscale=True):
        super(Conv2D, self).__init__()
        self.use_wscale = use_wscale
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * gain)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding

        self.gain = gain
        nn.init.normal_(self.weight, 0.0, self.gain / np.sqrt(in_channels * kernel_size ** 2))
        if use_wscale:
            self.scale = gain / np.sqrt(in_channels * kernel_size ** 2)
        else:
            self.scale = 1.0

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)

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
        y = y.mean(dim=[1, 2, 3], keepdims=True)
        y = y.expand(G, -1, H, W)
        y = y.repeat(N // G, 1, 1, 1)
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
        self.conv = Conv2D(in_channels, out_channels, 1, padding=0, use_wscale=use_wscale)
        self.act = nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))
    
class GAN_Discriminator(nn.Module):
    def __init__(self, num_channels=3, max_resolution=(2048, 1024), fmap_base=8192, fmap_decay=1.0, fmap_max=512, use_wscale=True, use_leaky_relu=True):
        super(GAN_Discriminator, self).__init__()
        self.resolution_log2_w = int(np.log2(max_resolution[0]))
        self.resolution_log2_h = int(np.log2(max_resolution[1]))
        self.min_res_log2 = min(self.resolution_log2_w, self.resolution_log2_h)

        self.num_channels = num_channels
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.use_wscale = use_wscale
        self.use_leaky_relu = use_leaky_relu
        
        self.initial_block = nn.Sequential(
            MinibatchStddevLayer(),
            Conv2D(nf(1, fmap_base, fmap_decay, fmap_max) + 1, nf(1, fmap_base, fmap_decay, fmap_max), 3, use_wscale=use_wscale),
            nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU(),
            Conv2D(nf(1, fmap_base, fmap_decay, fmap_max), nf(1, fmap_base, fmap_decay, fmap_max), 4, padding=0, use_wscale=use_wscale),
            nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU(),
            nn.Flatten(),
            nn.Linear(nf(1, fmap_base, fmap_decay, fmap_max) * 5, 1)
        )

        self.alpha_blend_layers = nn.ModuleDict()
        self.from_rgb_layers = nn.ModuleDict()
        self.from_rgb_layers['layer2'] = FromRGB(num_channels, nf(2, fmap_base, fmap_decay, fmap_max), use_wscale, use_leaky_relu)
        
        self.blocks = nn.ModuleList()
        self.blocks.append(self.initial_block)
    
    def add_block(self, resolution, device):
        resolution_log2_w = int(np.log2(resolution[0]))
        resolution_log2_h = int(np.log2(resolution[1]))
        res_log2 = min(resolution_log2_w, resolution_log2_h)

        if resolution == (2048, 1024):
            new_block = nn.Sequential(
                Conv2D(nf(self.resolution_log2_h, self.fmap_base, self.fmap_decay, self.fmap_max), 
                       nf(self.resolution_log2_h, self.fmap_base, self.fmap_decay, self.fmap_max), 
                       3, 
                       use_wscale=self.use_wscale),
                nn.LeakyReLU(0.2) if self.use_leaky_relu else nn.ReLU(),
                Conv2D(nf(self.resolution_log2_h, self.fmap_base, self.fmap_decay, self.fmap_max), 
                       nf(self.resolution_log2_h-1, self.fmap_base, self.fmap_decay, self.fmap_max), 
                       3, 
                       use_wscale=self.use_wscale),
                nn.LeakyReLU(0.2) if self.use_leaky_relu else nn.ReLU(),
                nn.AvgPool2d(2)
            ).to(device)
            self.blocks.insert(0, new_block)
            self.alpha_blend_layers[f'layer{res_log2}'] = DiscAlphaBlendLayer().to(device)
            self.from_rgb_layers[f'layer{res_log2}'] = FromRGB(self.num_channels, 
                        nf(self.resolution_log2_h, self.fmap_base, self.fmap_decay, self.fmap_max), 
                        self.use_wscale, 
                        self.use_leaky_relu).to(device)
            
            print(f'Final Block: {res_log2}')
            
            return

        in_channels = nf(res_log2, self.fmap_base, self.fmap_decay, self.fmap_max)
        out_channels = nf(res_log2-1, self.fmap_base, self.fmap_decay, self.fmap_max)

        new_from_rgb = FromRGB(self.num_channels, in_channels, self.use_wscale, self.use_leaky_relu).to(device)
        new_block = DiscriminatorBlock(in_channels, out_channels, self.use_wscale, self.use_leaky_relu).to(device)

        self.blocks.insert(0, new_block)
        self.alpha_blend_layers[f'layer{res_log2}'] = DiscAlphaBlendLayer().to(device)
        self.from_rgb_layers[f'layer{res_log2}'] = new_from_rgb

    def forward(self, images, resolution, alpha=1.0):
        print(f'Training at resolution: {resolution}')
        print(f'Image Shape: {images.shape}')
        resolution_log2_w = int(np.log2(images.shape[2]))
        resolution_log2_h = int(np.log2(images.shape[3]))
        resolution_log2 = min(resolution_log2_w, resolution_log2_h)
        x = images
        blocks = self.blocks
        if len(self.blocks) > 1 and alpha < 1.0:
            x_low = F.avg_pool2d(x, kernel_size=2)
            x_low = self.from_rgb_layers[f'layer{resolution_log2-1}'](x_low)
            # print(x_low.shape)

            x_high = self.from_rgb_layers[f'layer{resolution_log2}'](x)
            x_high = self.blocks[0](x_high)
            x = self.alpha_blend_layers[f'layer{resolution_log2}'](x_high, x_low, alpha)
            blocks = self.blocks[1:]
        else:
            x = self.from_rgb_layers[f'layer{resolution_log2}'](x)
        
        for i, block in enumerate(blocks):
            print(f'Block: {i}')
            x = block(x)
            print(x.shape)
        return x