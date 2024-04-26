import os
import torch as torch
#import tensorflow as tf
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from config.config import get_config
from PIL import Image
from datasets.my_dataset import MyDataset


class dataloader:
    def __init__(self, config):
        self.config = get_config()
        self.root = self.config.train_data_root
        self.batch_table = {4:32, 8:32, 16:32, 32:16, 64:16, 128:16, 256:12, 512:3, 1024:1} # change this according to available gpu memory.
        self.batchsize = int(self.batch_table[4])        # we start from 2^2=4
        self.imsize = 4
        self.num_workers = 4

        self.train_dataloader = None
        self.val_dataloader = None
        self.setup_dataloaders(self.imsize)
    
    def setup_dataloaders(self, imsize):
        # Define the transformations
        transform_x = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        transform_y = transforms.Compose([
            transforms.Resize(size=(imsize,imsize), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        # Create the dataset
        self.dataset = MyDataset(
            image_dir=os.path.join(self.config.train_data_root, "x/"),
            mask_dir=os.path.join(self.config.train_data_root, "y/"),
            transform=transform_x,
            target_transform=transform_y
        )

        if torch.cuda.is_available():
            generator = torch.Generator(device='cuda')
        else:
            generator = torch.Generator(device='cpu') 
        
        # Split the dataset into training and validation sets
        torch.manual_seed(self.config.random_seed) # ensures that each time renew is called, the same split is used
        n_val = int(len(self.dataset) * self.config.val_split)
        n_train = len(self.dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(self.dataset, [n_train, n_val], generator=generator)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=self.num_workers,
            generator=generator
        )
        
    def renew(self, resl):
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))
        
        self.batchsize = int(self.batch_table[pow(2,resl)])
        self.imsize = int(pow(2,resl))
        self.setup_dataloaders(self.imsize)

    def __iter__(self):
        return iter(self.train_dataloader)
    
    def __next__(self):
        return next(self.train_dataloader)

    def __len__(self):
        return len(self.train_dataloader.dataset)

       
    def get_train_batch(self):
        data_iter = iter(self.train_dataloader)
        x, y = next(data_iter)
        # Adjust the pixel range of images and masks from [0, 1] to [-1, 1]
        x = x.mul(2).add(-1)  # scales [0,1] -> [-1,1]
        y = y.mul(2).add(-1)  # scales [0,1] -> [-1,1]
        return x, y

    def get_val_batch(self):
        data_iter = iter(self.val_dataloader)
        x, y = next(data_iter)
        # Adjust the pixel range of images and masks from [0, 1] to [-1, 1]
        x = x.mul(2).add(-1)  # scales [0,1] -> [-1,1]
        y = y.mul(2).add(-1)  # scales [0,1] -> [-1,1]
        return x, y


        