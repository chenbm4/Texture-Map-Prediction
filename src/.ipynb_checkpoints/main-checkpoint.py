from models.unet_model import UNet
import torch

def main():
    print(torch.cuda.is_available())
    # Create an instance of the UNet model
    model = UNet(n_channels=3, n_classes=3, bilinear=False)

    # Dummy input
    input_shape = (3, 512, 512)
    dummy_input = torch.randn(1, *input_shape)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Print input and output shape
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()