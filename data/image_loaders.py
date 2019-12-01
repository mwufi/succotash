from torchvision import datasets, transforms


def create_dataset(args):
	if hasattr(args, 'name') and args.name == 'mnist':
			dataset = datasets.MNIST('../examples/mnist_data', train=True, download=True,
						   transform=transforms.Compose([
							   transforms.ToTensor(),
							   transforms.Normalize((0.1307,), (0.3081,))
						   ]))
			return dataset

	if hasattr(args, 'path'):
		dataset = datasets.ImageFolder(root=args.path, transform=transforms.Compose([
			transforms.Resize(args.image_size),
			transforms.CenterCrop(args.image_size),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		]))
		return dataset

