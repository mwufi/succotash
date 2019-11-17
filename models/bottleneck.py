import torch
from torch import nn
from torch.nn import functional as F

from models import ModelBase


class Bottleneck(ModelBase):
	name = 'bottleneck'

	def __init__(self, n_channels, growth_rate):
		super().__init__()
		hidden_channels = 4 * growth_rate
		self.bn1 = nn.BatchNorm2d(n_channels)
		self.conv1 = nn.Conv2d(n_channels, hidden_channels, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(hidden_channels)
		self.conv2 = nn.Conv2d(hidden_channels, growth_rate, kernel_size=3, padding=1, bias=False)
		self._post_init()

	def _forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = self.conv2(F.relu(self.bn2(out)))
		out = torch.cat((x, out), 1)
		return out


class DownStack(ModelBase):
	name = 'conv discriminator'

	def __init__(self, n_data_channels, n_base_filters):
		super().__init__()
		self.main = nn.Sequential(
			# input is (n_data_channels) x 64 x 64
			nn.Conv2d(n_data_channels, n_base_filters, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (n_base_filters) x 32 x 32
			nn.Conv2d(n_base_filters, n_base_filters * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(n_base_filters * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (n_base_filters*2) x 16 x 16
			nn.Conv2d(n_base_filters * 2, n_base_filters * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(n_base_filters * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (n_base_filters*4) x 8 x 8
			nn.Conv2d(n_base_filters * 4, n_base_filters * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(n_base_filters * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (n_base_filters*8) x 4 x 4
			nn.Conv2d(n_base_filters * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def _forward(self, input):
		return self.main(input)


class UpStack(ModelBase):
	name = 'conv generator'

	def __init__(self, z_dim, n_output_channels, n_base_filters):
		super().__init__()
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(z_dim, n_base_filters * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(n_base_filters * 8),
			nn.ReLU(True),
			# state size. (n_base_filters*8) x 4 x 4
			nn.ConvTranspose2d(n_base_filters * 8, n_base_filters * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(n_base_filters * 4),
			nn.ReLU(True),
			# state size. (n_base_filters*4) x 8 x 8
			nn.ConvTranspose2d(n_base_filters * 4, n_base_filters * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(n_base_filters * 2),
			nn.ReLU(True),
			# state size. (n_base_filters*2) x 16 x 16
			nn.ConvTranspose2d(n_base_filters * 2, n_base_filters, 4, 2, 1, bias=False),
			nn.BatchNorm2d(n_base_filters),
			nn.ReLU(True),
			# state size. (n_base_filters) x 32 x 32
			nn.ConvTranspose2d(n_base_filters, n_output_channels, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (n_output_channels) x 64 x 64
		)

	def _forward(self, input):
		return self.main(input)
