import logging
from typing import Any

from numpy import prod
from torch import nn

format_strings = {
	'date': '%(asctime)s:%(msecs)03d - %(name)s - %(message)s',
	'basic': ':%(msecs)03d - %(name)s - %(message)s'
}

logging.basicConfig(
	format=format_strings['basic'],
	level=logging.INFO,
	datefmt='%m/%d/%Y %I:%M:%S'
)


class ModelBase(nn.Module):
	def __init__(self):
		super().__init__()
		self.logger = logging.getLogger(self.name)

	@property
	def trainable_parameters(self):
		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		params = sum([prod(p.size()) for p in model_parameters])
		return params

	def preprocess(self, *input):
		for t in input:
			self.logger.info(f'input size: {t.size()}')
		return input

	def postprocess(self, tensor_outputs):
		self.logger.info(f'output size: {tensor_outputs.size()}')
		return tensor_outputs

	def forward(self, *input: Any, **kwargs: Any):
		tensor_inputs = self.preprocess(*input)
		tensor_outputs = self._forward(*tensor_inputs, **kwargs)
		return self.postprocess(tensor_outputs)


class MLP(ModelBase):
	name = 'MLP'

	def __init__(self, layer_sizes):
		super().__init__()

		if len(layer_sizes) < 1:
			raise ValueError('You have to have an input dimension')
		if len(layer_sizes) < 2:
			raise ValueError('You have to have an output dimension')

		f = [
			nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i + 1])
			for i in range(len(layer_sizes) - 1)
		]

		self.layers = nn.Sequential(*f)

		print()
		print(f'(init)'.center(40, '-'))
		print(f'{self.name}')
		print(f'{self.trainable_parameters} trainable parameters')
		print('(end of init)'.center(40, '-'))
		print()

	def _forward(self, tensor: Any):
		return self.layers(tensor)
