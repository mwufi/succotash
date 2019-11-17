import torch

from models.basic import MLP
from utils import test_config

H = 40
W = 40


def print_diagnostics():
	print('Welcome to Dixit GAN!')
	print('Pytorch version:', torch.__version__)


def create_model():
	x = MLP([H * W, 100, 10])
	print(x)
	return x


if __name__ == "__main__":
	print_diagnostics()
	test_config()
	# x = torch.rand((H, W), names=('H', 'W'))
	# model = create_model()
	# print(x)
	# print(model(x))
