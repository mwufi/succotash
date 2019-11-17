import torch
from torch.nn import functional as F
from torch import nn

from models.basic import ModelBase


class Bottleneck(ModelBase):
	name = 'bottleneck'

	def _init(self, n_channels, growth_rate):
		hidden_channels = 4 * growth_rate
		self.bn1 = nn.BatchNorm2d(n_channels)
		self.conv1 = nn.Conv2d(n_channels, hidden_channels, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(hidden_channels)
		self.conv2 = nn.Conv2d(hidden_channels, growth_rate, kernel_size=3, padding=1, bias=False)

	def _forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = self.conv2(F.relu(self.bn2(out)))
		out = torch.cat((x, out), 1)
		return out
