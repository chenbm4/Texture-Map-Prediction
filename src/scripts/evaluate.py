import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src/')
from models.unet_model import UNet
from datasets.my_dataset import MyDataset

# Configuration
config = {
    "dataset_path": "data/processed/full/test/",
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "checkpoints/full/256_mse/best_checkpoint.pth",
    "output_folder": "predictions/256_mse/"
}

# Load the test dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

test_dataset = MyDataset(
    image_dir=os.path.join(config["dataset_path"], "x/"),
    mask_dir=os.path.join(config["dataset_path"], "y/"),
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Model setup
model = UNet(n_channels=3, n_classes=3, bilinear=False)

# Load the saved model
checkpoint = torch.load(config["checkpoint_path"], map_location=config["device"])
model.load_state_dict(checkpoint['state_dict'])
model.to(config["device"])
model.eval()

# Evaluate the model
def evaluate():
    running_loss = 0.0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            for j, output in enumerate(outputs):
                save_image(output, f"{i * config['batch_size'] + j}.png")

    average_loss = running_loss / len(test_loader)
    print(f"Test Loss: {average_loss}")

def save_image(tensor, filename):
    img = transforms.ToPILImage()(tensor.cpu().data)
    img.save(os.path.join(config["output_folder"], filename))

# Run evaluation
evaluate()
