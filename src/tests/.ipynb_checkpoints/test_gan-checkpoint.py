import unittest
import torch
from models.import_gan import Generator, Discriminator  # Adjust this import based on your actual module paths

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.config = {'nz': 100, 'ngf': 64, 'nc': 3, 'flag_bn': True, 'flag_pixelwise': True,
                       'flag_wn': True, 'flag_leaky': True, 'flag_tanh': True, 'flag_norm_latent': True}
        self.generator = Generator(self.config)

    def test_initialization(self):
        """ Test initialization of generator. """
        self.assertIsInstance(self.generator, Generator)

    def test_output_shape(self):
        """ Test the output shape of the generator. """
        input_z = torch.randn(4, self.config['nz'], 1, 1)  # Batch size of 4
        output = self.generator(input_z)
        self.assertEqual(output.shape, (4, self.config['nc'], 16, 16))  # Assume 16x16 is the output spatial dimension

    def test_parameter_update(self):
        """ Ensure parameters are being updated. """
        input_z = torch.randn(4, self.config['nz'], 1, 1)
        output = self.generator(input_z)
        initial_params = [p.clone() for p in self.generator.parameters()]
        output.mean().backward()
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.01)
        optimizer.step()
        for initial, updated in zip(initial_params, self.generator.parameters()):
            self.assertFalse(torch.equal(initial, updated))

class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.config = {'ndf': 64, 'nc': 3, 'nz': 100, 'flag_bn': True, 'flag_pixelwise': True,
                       'flag_wn': True, 'flag_leaky': True, 'flag_sigmoid': True}
        self.discriminator = Discriminator(self.config)

    def test_initialization(self):
        """ Test initialization of discriminator. """
        self.assertIsInstance(self.discriminator, Discriminator)

    def test_output_shape(self):
        """ Test the output shape of the discriminator. """
        input_img = torch.randn(4, self.config['nc'], 16, 16)  # Batch size of 4, assuming 16x16 images
        output = self.discriminator(input_img)
        self.assertEqual(output.shape, (4, 1))  # Assuming output is a probability per image

    def test_parameter_update(self):
        """ Ensure parameters are being updated. """
        input_img = torch.randn(4, self.config['nc'], 16, 16)
        output = self.discriminator(input_img)
        initial_params = [p.clone() for p in self.discriminator.parameters()]
        output.mean().backward()
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.01)
        optimizer.step()
        for initial, updated in zip(initial_params, self.discriminator.parameters()):
            self.assertFalse(torch.equal(initial, updated))

if __name__ == '__main__':
    unittest.main()
