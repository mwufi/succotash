import torch

from models.basic import MLP

B = 13
H = 40
W = 2


def print_diagnostics():
	print('Welcome to Dixit GAN!')
	print('Pytorch version:', torch.__version__)


def create_model():
	x = MLP([W*H, 10])
	return x


if __name__ == "__main__":
	print_diagnostics()

	x = torch.rand((B, H, W), names=('batch_size', 'height', 'width'))
	z = x.reshape((B, H*W))
	model = create_model()

	t = model(z)
