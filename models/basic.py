from typing import Any

from torch import nn


class MLP(nn.Module):
	def __init__(self, layer_sizes):
		if len(layer_sizes) < 1:
			raise ValueError('You have to have an input dimension')
		if len(layer_sizes) < 2:
			raise ValueError('You have to have an output dimension')

		self.layers = nn.Sequential(*[
			nn.Linear(layer_sizes[i], layer_sizes[i + 1])
			for i in range(len(layer_sizes) - 1)
		])

	def forward(self, tensor: Any, **kwargs: Any):
		return self.layers(tensor)
