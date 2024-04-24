import unittest
import torch
from ..models.gan_decoder import GAN_Generator, GAN_Discriminator, AlphaBlendLayer

class TestGANComponents(unittest.TestCase):
    def setUp(self):
        self.num_channels = 3
        self.resolution = 2048
        self.fmap_base = 8192
        self.fmap_decay = 1.0
        self.fmap_max = 512
        self.latent_size = 512
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.generator = GAN_Generator(num_channels=self.num_channels,
                                       resolution=self.resolution,
                                       fmap_base=self.fmap_base,
                                       fmap_decay=self.fmap_decay,
                                       fmap_max=self.fmap_max,
                                       latent_size=self.latent_size,
                                       normalize_latents=False,
                                       use_wscale=True,
                                       use_leaky_relu=True).to(self.device)
        
        self.discriminator = GAN_Discriminator(num_channels=self.num_channels,
                                               max_resolution=(self.resolution, self.resolution//2),
                                               fmap_base=self.fmap_base,
                                               fmap_decay=self.fmap_decay,
                                               fmap_max=self.fmap_max,
                                               use_wscale=True,
                                               use_leaky_relu=True).to(self.device)

    def test_generator_output_shape(self):
        # Test the generator output shape
        mock_latents = torch.randn(4, self.latent_size).to(self.device)  # Batch size of 4
        for i in range(4, 12):
            self.generator.add_block((pow(2, i), pow(2, i)//2), self.device)
            print(f"Added block with resolution {pow(2, i), pow(2, i)//2}")
        output = self.generator(mock_latents, alpha=0.5)
        expected_shape = (4, self.num_channels, self.resolution//2, self.resolution)
        self.assertEqual(output.shape, expected_shape)

    # def test_discriminator_output_shape(self):
    #     mock_images = torch.randn(4, self.num_channels, 2048, 1024).to(self.device)
    #     for i in range(4, 12):
    #         self.discriminator.add_block((pow(2, i), pow(2, i)//2), self.device)
    #         print(f"Added block with resolution {pow(2, i), pow(2, i)//2}")
    #     output = self.discriminator(mock_images, resolution=(2048, 1024), alpha=0.5)
    #     self.assertTrue(output.shape == (4, 1))

    # def test_alpha_blend(self):
    #     alpha_layer = AlphaBlendLayer().to('cpu')
    #     low_res = torch.full((1, 3, 64, 64), fill_value=0.5)
    #     high_res = torch.full((1, 3, 64, 64), fill_value=1.0)

    #     # Test for alpha = 0 (Should be exactly low_res)
    #     alpha = 0.0
    #     output = alpha_layer(low_res, high_res, alpha)
    #     self.assertTrue(torch.equal(output, low_res), f"Failed at alpha={alpha}")

    #     # Test for alpha = 1 (Should be exactly high_res)
    #     alpha = 1.0
    #     output = alpha_layer(low_res, high_res, alpha)
    #     self.assertTrue(torch.equal(output, high_res), f"Failed at alpha={alpha}")

    #     # Test for alpha = 0.5 (Should be average of low_res and high_res)
    #     alpha = 0.5
    #     expected_output = (low_res + high_res) / 2
    #     output = alpha_layer(low_res, high_res, alpha)
    #     torch.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-8, msg=f"Failed at alpha={alpha}")

    #     alphas = [0.25, 0.75]
    #     for alpha in alphas:
    #         output = alpha_layer(low_res, high_res, alpha)
    #         expected_output = low_res * (1 - alpha) + high_res * alpha
    #         torch.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-8, msg=f"Failed at alpha={alpha}")

if __name__ == '__main__':
    unittest.main()
