import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import copy
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.append('src/')
from models.unet_model import UNet
from datasets.my_dataset import MyDataset


config = {
    "epochs": 100,
    "batch_size": 4,
    "learning_rate": 0.001,
    "val_split": 0.1,
    "dataset_path": "data/processed/small/",
    "checkpoint_dir": "checkpoints/",
    # "early_stopping_patience": 3,
    "log_dir": "logs/",
    "log_interval": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print("Configuration:", config)

# Initialize TensorBoard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join(config["log_dir"], current_time, 'train')
val_log_dir = os.path.join(config["log_dir"], current_time, 'val')
train_summary_writer = SummaryWriter(train_log_dir)
val_summary_writer = SummaryWriter(val_log_dir)

# Create the dataset
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

dataset = MyDataset(
    image_dir=os.path.join(config["dataset_path"], "x/"),
    mask_dir=os.path.join(config["dataset_path"], "y/"),
    transform=transform
)

# Split the dataset into training and validation
n_val = int(len(dataset) * config["val_split"])
n_train = len(dataset) - n_val

print(f"Total dataset size: {len(dataset)}")
print(f"Training dataset size: {n_train}")
print(f"Validation dataset size: {n_val}")

train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

print("Train and validation data loaders created.")

# Model setup
model = UNet(n_channels=3, n_classes=3, bilinear=False).to(config["device"])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

print("Model initialized:", model)

# Training and validation functions
def train_epoch(epoch_index):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # TensorBoard logging
        for name, param in model.named_parameters():
            train_summary_writer.add_histogram(name, param, epoch_index)
            if param.grad is not None:
                train_summary_writer.add_histogram(f'{name}.grad', param.grad, epoch_index)

        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_summary_writer.add_scalar('Training Loss', epoch_loss, epoch_index)

    # TensorBoard logging
    if epoch_index % config["log_interval"] == 0:
        inputs_batch = inputs.cpu().data
        labels_batch = labels.cpu().data
        outputs_batch = outputs.cpu().data
        train_summary_writer.add_images('Input', inputs_batch, epoch_index)
        train_summary_writer.add_images('Labels', labels_batch, epoch_index)
        train_summary_writer.add_images('Outputs', outputs_batch, epoch_index)
    
    current_lr = optimizer.param_groups[0]['lr']
    train_summary_writer.add_scalar('Learning Rate', current_lr, epoch_index)
    return epoch_loss

def val_epoch(epoch_index):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    epoch_loss = running_loss / len(val_loader)
    val_summary_writer.add_scalar('Validation Loss', epoch_loss, epoch_index)

    # TensorBoard logging
    if epoch_index % config["log_interval"] == 0:
        inputs_batch = inputs.cpu().data
        labels_batch = labels.cpu().data
        outputs_batch = outputs.cpu().data
        val_summary_writer.add_images('Input', inputs_batch, epoch_index)
        val_summary_writer.add_images('Labels', labels_batch, epoch_index)
        val_summary_writer.add_images('Outputs', outputs_batch, epoch_index)

    return epoch_loss

# Checkpoint saving
def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, filename)

# TensorBoard Graph
dummy_input = torch.randn(1, 3, 512, 512).to(config["device"])
train_summary_writer.add_graph(model, dummy_input)

# Main training loop
best_loss = np.inf
epochs_no_improve = 0

print("Training started")
for epoch in tqdm(range(config["epochs"])):
    train_loss = train_epoch(epoch)
    val_loss = val_epoch(epoch)
    scheduler.step(val_loss)

    print(f"Epoch: {epoch + 1}/{config['epochs']}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        save_checkpoint(epoch, model, optimizer, os.path.join(config["checkpoint_dir"], 'best_checkpoint.pth'))
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == config["early_stopping_patience"]:
            print("Early stopping triggered")
            break

model.load_state_dict(best_model_wts)
train_summary_writer.close()
val_summary_writer.close()
print("Training completed")

# Log hyperparameters and final metrics
hparams = {'lr': config['learning_rate'], 'batch_size': config['batch_size']}
final_metrics = {'loss': best_loss}
train_summary_writer.add_hparams(hparams, final_metrics)