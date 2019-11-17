import torch

from utils import Config
import argparse
from models.basic import MLP
from models.bottleneck import Bottleneck

B = 13
C = 3
H = 40
W = 2


def print_diagnostics():
	print('Welcome to Dixit GAN!')
	print('Pytorch version:', torch.__version__)


def create_model(args):
	print(args)
	x = MLP([C*W*H, 10])
	return x


if __name__ == "__main__":
	print_diagnostics()

	parser = argparse.ArgumentParser('Dixit GAN generator')
	parser.add_argument('--config', type=str, help='Path to config file', required=True)
	args = parser.parse_args()

	c = Config(args.config)

	model = create_model(c.model)

	x = torch.rand((B, C, H, W))
	z = x.reshape((B, H*W*C))
	t = model(z)

	p = Bottleneck(3, 2)
	output = p(x)

