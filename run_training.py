import argparse

import torch

from utils import Config

B = 13
C = 3
H = 40
W = 2


def print_diagnostics():
	print('Welcome to Dixit GAN!')
	print('Pytorch version:', torch.__version__)


def create_model(args):
	print('model', args)
	return 4


def create_optimiser(args):
	print('optimiser', args)
	return 4


def create_dataloader(args):
	print('data', args)
	return 4


def train(args):
	print('training', args)

	for epoch in range(args.epochs):
		print('Epoch {}'.format(epoch))
		print('Done!')


if __name__ == "__main__":
	print_diagnostics()

	parser = argparse.ArgumentParser('Dixit GAN generator')
	parser.add_argument('--config', type=str, help='Path to config file', required=True)
	args = parser.parse_args()

	c = Config(args.config)

	data = create_dataloader(c.data)
	model = create_model(c.model)
	train(c.training)

# x = torch.rand((B, C, H, W))
# z = x.reshape((B, H * W * C))
# t = model(z)
#
# p = Bottleneck(3, 2)
# output = p(x)
