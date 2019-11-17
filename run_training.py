import torch

from models.basic import MLP
from models.bottleneck import Bottleneck

B = 13
C = 3
H = 40
W = 2


def print_diagnostics():
	print('Welcome to Dixit GAN!')
	print('Pytorch version:', torch.__version__)


def create_model():
	x = MLP([C*W*H, 10])
	return x


if __name__ == "__main__":
	print_diagnostics()

	x = torch.rand((B, C, H, W))
	z = x.reshape((B, H*W*C))
	model = create_model()
	
	p = Bottleneck(3, 2)
	output = p(x)

	t = model(z)
