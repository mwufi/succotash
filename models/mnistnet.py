import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
	input_size = [1,28,28]

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout2d(0.25)
		self.dropout2 = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1)
		return output


class myNet(nn.Module):
	input_size = None

	def __init__(self):
		super().__init__()
		self.conv = nn.Conv2d(3, 10, 2, stride=2)
		self.relu = nn.ReLU()
		self.flatten = lambda x: x.view(-1)
		self.fc1 = nn.Linear(160, 5)
		self.seq = nn.Sequential(nn.Linear(5, 3), nn.Linear(3, 2))

	def forward(self, x):
		x = self.relu(self.conv(x))
		x = self.fc1(self.flatten(x))
		x = self.seq(x)
