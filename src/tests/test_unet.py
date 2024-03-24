import unittest
import sys
import torch
sys.path.append('../')

from models.unet_model import UNet

class TestUNetModel(unittest.TestCase):

    def test_forward_pass(self):
        model = UNet(n_channels=3, n_classes=3, bilinear=False)
        input_shape = (3, 512, 512)
        dummy_input = torch.randn(1, *input_shape)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(dummy_input.shape, (1, *input_shape))
        self.assertEqual(output.shape, (1, 3, 512, 512))

if __name__ == '__main__':
    unittest.main()