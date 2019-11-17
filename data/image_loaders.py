import torchvision.datasets as dset
import torchvision.transforms as transforms


def create_dataset(args):
	dataset = dset.ImageFolder(root=args.path, transform=transforms.Compose([
		transforms.Resize(args.image_size),
		transforms.CenterCrop(args.image_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	]))
	return dataset
#
