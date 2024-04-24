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
import numpy as np
from tqdm import tqdm

sys.path.append('src/')
from models.unet_encoder import UNet
from models.gan_decoder import GAN_Generator, GAN_Discriminator, update_smoothed_generator
from models.loss import generator_loss, discriminator_loss
from datasets.my_dataset import MyDataset


config = {
    "epochs": 100,
    "batch_size": 4,
    "learning_rate": 0.001,
    "val_split": 0.1,
    "dataset_path": "data/processed/full/test/",
    "checkpoint_dir": "checkpoints/",
    # "early_stopping_patience": 3,
    "log_dir": "logs/",
    "log_interval": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# print("Configuration:", config)

# Create the directories
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(config["log_dir"], current_time)
os.makedirs(config["checkpoint_dir"], exist_ok=True)
os.makedirs(config["log_dir"], exist_ok=True)

# Initialize TensorBoard
log_writer = SummaryWriter(log_dir)

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
encoder_model = UNet(n_channels=3, n_classes=3).to(config["device"])
generator_model = GAN_Generator().to(config["device"])
discriminator_model = GAN_Discriminator().to(config["device"])

# smoothed generator model
generator_smoothed_model = copy.deepcopy(generator_model).to(config["device"])

optimizer_E = optim.Adam(encoder_model.parameters(), lr=config["learning_rate"])
optimizer_G = optim.Adam(generator_model.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator_model.parameters(), lr=0.001, betas=(0.5, 0.999))


def train_epoch(epoch_index):
    encoder_model.train()
    generator_model.train()
    discriminator_model.train()

    running_EG_loss = 0.0   # Encoder-Generator loss
    running_D_loss = 0.0    # Discriminator loss

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        
        latents = encoder_model(inputs)
        outputs = generator_model(latents)
        
        EG_loss = generator_loss(discriminator_model, outputs, labels)
        EG_loss.backward(retain_graph=True) # Retain graph for discriminator loss
        running_EG_loss += EG_loss.item()

        # Update encoder and generator
        optimizer_E.step()
        optimizer_G.step()

        # Train Discriminator
        D_loss = discriminator_loss(discriminator_model, labels, outputs.detach(), device=config["device"])
        D_loss.backward()
        running_D_loss += D_loss.item()
        
        optimizer_D.step()

        # TensorBoard logging
        for name, param in encoder_model.named_parameters():
            log_writer.add_histogram(name, param, epoch_index)
            if param.grad is not None:
                log_writer.add_histogram(f'{name}.grad', param.grad, epoch_index)

     # Compute average losses
    avg_EG_loss = running_EG_loss / len(train_loader)
    avg_D_loss = running_D_loss / len(train_loader)

    # Log losses to TensorBoard
    log_writer.add_scalar('Loss/Encoder-Generator', avg_EG_loss, epoch_index)
    log_writer.add_scalar('Loss/Discriminator', avg_D_loss, epoch_index)

    # TensorBoard logging
    if epoch_index % config["log_interval"] == 0:
        inputs_batch = inputs.cpu().data
        labels_batch = labels.cpu().data
        outputs_batch = outputs.cpu().data
        log_writer.add_images('Input', inputs_batch, epoch_index)
        log_writer.add_images('Labels', labels_batch, epoch_index)
        log_writer.add_images('Outputs', outputs_batch, epoch_index)
    
    current_lr = optimizer_E.param_groups[0]['lr']
    log_writer.add_scalar('Learning Rate', current_lr, epoch_index)
    return avg_EG_loss, avg_D_loss

def val_epoch(epoch_index):
    encoder_model.eval()
    generator_model.eval()
    discriminator_model.eval()
    EG_running_loss = 0.0
    D_running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            latents = encoder_model(inputs)
            outputs = generator_model(latents)
            EG_loss = generator_loss(discriminator_model, outputs, labels)
            D_loss = discriminator_loss(discriminator_model, labels, outputs, device=config["device"])
            EG_running_loss += EG_loss.item()
            D_running_loss += D_loss.item()

    EG_epoch_loss = EG_running_loss / len(val_loader)
    log_writer.add_scalar('Validation Loss (Encoder/Generator)', EG_epoch_loss, epoch_index)
    D_epoch_loss = D_running_loss / len(val_loader)
    log_writer.add_scalar('Validation Loss (Discriminator)', D_epoch_loss, epoch_index)

    # TensorBoard logging
    if epoch_index % config["log_interval"] == 0:
        inputs_batch = inputs.cpu().data
        labels_batch = labels.cpu().data
        outputs_batch = outputs.cpu().data
        log_writer.add_images('Input', inputs_batch, epoch_index)
        log_writer.add_images('Labels', labels_batch, epoch_index)
        log_writer.add_images('Outputs', outputs_batch, epoch_index)

    return EG_epoch_loss, D_epoch_loss

# Checkpoint saving
def save_checkpoint(epoch, E, G, D, opt_E, opt_G, opt_D, filename):
    checkpoint = {
        'epoch': epoch + 1,
        'E_state_dict': E.state_dict(),
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'opt_E_state_dict': opt_E.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_state_dict': opt_D.state_dict(),
    }

    torch.save(checkpoint, filename)

# TensorBoard Graph
encoder_dummy_input = torch.randn(1, 3, 512, 512).to(config["device"])
generator_dummy_input = torch.randn(1, 512).to(config["device"])
discriminator_dummy_input = torch.randn(1, 3, 1024, 2048).to(config["device"])
log_writer.add_graph(encoder_model, encoder_dummy_input)
print('Encoder model graph added to TensorBoard')
log_writer.add_graph(generator_model, generator_dummy_input)
print('Generator model graph added to TensorBoard')
log_writer.add_graph(discriminator_model, discriminator_dummy_input)
print('Discriminator model graph added to TensorBoard')

# Main training loop
best_model_wts = {'encoder': copy.deepcopy(encoder_model.state_dict()),
                  'generator': copy.deepcopy(generator_model.state_dict()),
                  'discriminator': copy.deepcopy(discriminator_model.state_dict())}
best_loss = np.inf

print("Training started")
for epoch in tqdm(range(config["epochs"])):
    train_loss = train_epoch(epoch)
    EG_val_loss, D_val_loss = val_epoch(epoch)

    update_smoothed_generator(generator_smoothed_model, generator_model)

    print(f"Epoch: {epoch + 1}/{config['epochs']}, Train Loss: {train_loss}, EG Val Loss: {EG_val_loss}, D Val Loss: {D_val_loss}") 

    if EG_val_loss < best_loss:
        best_loss = EG_val_loss
        best_model_wts['encoder'] = copy.deepcopy(encoder_model.state_dict())
        best_model_wts['generator'] = copy.deepcopy(generator_smoothed_model.state_dict())
        best_model_wts['discriminator'] = copy.deepcopy(discriminator_model.state_dict())
        save_checkpoint(epoch, 
                        encoder_model, 
                        generator_smoothed_model, 
                        discriminator_model, 
                        optimizer_E, 
                        optimizer_G, 
                        optimizer_D, 
                        os.path.join(config["checkpoint_dir"], 'best_checkpoint.pth'))

encoder_model.load_state_dict(best_model_wts['encoder'])
generator_model.load_state_dict(best_model_wts['generator'])
discriminator_model.load_state_dict(best_model_wts['discriminator'])

# Log hyperparameters and final metrics before closing the writer
hparams = {'lr': config['learning_rate'], 'batch_size': config['batch_size']}
final_metrics = {'loss': best_loss}
log_writer.add_hparams(hparams, final_metrics)

# Close the TensorBoard writer
log_writer.close()
print("Training completed")