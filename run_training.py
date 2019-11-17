import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.nn.parallel
import torch.utils.data
import torch.utils.data
import torchvision.utils as vutils

from data import create_dataset
from models import MLP
from utils import Config

# %matplotlib inline

B = 13
C = 3
H = 40
W = 2


def print_diagnostics():
	print('Welcome to Dixit GAN!')
	print('Pytorch version:', torch.__version__)


def create_model(args):
	print('model', args)
	return MLP([H * W * C, 100, 10])


def create_optimiser(args):
	print('optimiser', args)
	return 4


def create_dataloader(args):
	print('data', args)
	dataset = create_dataset(args)

	# Create the dataloader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
											 shuffle=True,
											 num_workers=args.workers)
	return dataloader


def set_random_seed(args):
	"""Set a seed for reproducibility"""
	if isinstance(args.random_seed, int):
		manualSeed = args.random_seed
	else:
		manualSeed = random.randint(1, 10000)

	print("Random Seed: ", manualSeed)
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)


def train(args):
	print('training', args)
	set_random_seed(args)

	for epoch in range(args.epochs):
		print('Epoch {}'.format(epoch))
		print('Done!')


if __name__ == "__main__":
	print_diagnostics()

	parser = argparse.ArgumentParser('Dixit GAN generator')
	parser.add_argument('--config', type=str, help='Path to config file', required=True)
	args = parser.parse_args()

	c = Config(args.config)

	dataloader = create_dataloader(c.data)
	model = create_model(c.model)

	# Decide which device we want to run on
	device = torch.device("cuda:0" if (torch.cuda.is_available() and c.training.gpus > 0) else "cpu")

	# Plot some training images
	print('data loading')
	real_batch = next(iter(dataloader))
	plt.figure(figsize=(8, 8))
	plt.axis("off")
	plt.title("Training Images")
	plt.imshow(
		np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
	plt.show()
