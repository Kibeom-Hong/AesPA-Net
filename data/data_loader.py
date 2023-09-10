import os
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torch
from .dataset_util import *
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split

def get_loader(image_dir, crop_size=0, image_size=0, batch_size=16, normalize=True, noise=True, split='train', num_workers=2):
	transform = []
	if split == 'train':
		transform.append(transforms.RandomHorizontalFlip())
	if crop_size > 0:
		transform.append(transforms.CenterCrop([crop_size, crop_size]))
	if image_size > 0:
		transform.append(transforms.Resize([image_size, image_size]))
	transform.append(transforms.ToTensor())
	if normalize:
		transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	if noise:
		transform.append(transforms.Lambda(lambda x: x + 1./128 * torch.randn(x.size())))
	# TODO: add mirror augmentation (stylegan)
	# TODO: (generaory) change down / up sampling method (stylegan)
	transform = transforms.Compose(transform)

	dataset = datasets.ImageFolder(image_dir, transform=transform)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader

def get_loaders(dir_dataset, crop_size=0, image_size=0, batch_size=16, num_workers=2):
	modes = ['train', 'val', 'test']
	loaders = {}
	for mode in modes:
		dir_imagefolder = os.path.join(dir_dataset, mode)
		if not os.path.exists(dir_imagefolder):
			continue
		loader = get_loader(dir_imagefolder, crop_size, image_size, batch_size, mode, num_workers)
		loaders[mode] = loader
	return loaders

def get_video_loader(name, root_dir, vid_length=32, image_size=0, batch_size=16, normalize=True, noise=False, split='train', num_workers=2, mode=None):
	transform = []
	if image_size > 0:
		if name in ['UCF101', 'UCF101_mini']:
			transform.append(transforms.Resize([image_size, int(image_size*1.33)]))
			transform.append(transforms.CenterCrop([image_size, image_size]))
		else:
			transform.append(transforms.Resize([image_size, image_size]))
	transform.append(transforms.ToTensor())
	if normalize: # check one more
		transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	if noise:
		transform.append(transforms.Lambda(lambda x: x + 1./128 * torch.randn(x.size())))

	transform = transforms.Compose(transform)


	root_dir = '/media/cvpr-bu/6TB_1/dataset/'
	dir_path = os.path.join(root_dir, name)

	if name == 'HMDB':
		dataset = HMDB(root_dir=dir_path, transform=transform, T=vid_length)

	elif name == 'UCFsports':
		dataset = UCFsports(root_dir = dir_path, transform = transform, T=vid_length)

	elif name == 'HumanAction':
		dataset = HumanAction(root_dir = dir_path, transform = transform, T=vid_length)

	elif name == 'UCF101':
		dataset = UCF101(root_dir = dir_path, transform = transform, T=vid_length)

	elif name == 'UCF101_mini':
		dataset = UCF101_mini(root_dir = dir_path , transform = transform, T=vid_length)

	elif name == 'AoT':
		dir_path = os.path.join(root_dir, "UCFsports")
		#num_train = 13318
		#num_train = 83
		num_train = 143
		split_ = int(np.floor(0.2*num_train))
		train_idx, valid_idx = num_train-split_, split_
		#indices = list(range(num_train))
		#train_idx, valid_idx = indices[split:], indices[:split]
		#train_sampler = SubsetRandomSampler(train_idx)
		#valid_sampler = SubsetRandomSampler(valid_idx)
		train_dataset, valid_dataset = random_split(UCF101_AoT(root_dir = dir_path , transform = transform, T=vid_length), [train_idx, valid_idx])
		#train_dataset = UCFsports_AoT_Train(root_dir = dir_path , transform = transform, T=vid_length)
		#valid_dataset = UCFsports_AoT_Valid(root_dir = dir_path , transform = transform, T=vid_length)

		if split == 'train':
			return data.DataLoader(dataset=train_dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  pin_memory=True,
								  num_workers=num_workers, drop_last=True)
		else:
			return data.DataLoader(dataset=valid_dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  pin_memory=True,
								  num_workers=num_workers, drop_last=True)

	
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  pin_memory=True,
								  num_workers=num_workers, drop_last=True)

	return data_loader
	

def get_video_loaders(name, root_dir, vid_length=32, crop_size=0, image_size=0, batch_size=16, num_workers=2):
	modes = ['train', 'val', 'test']
	loaders = {}
	for mode in modes:
		dir_imagefolder = os.path.join(dir_dataset, mode)
		if not os.path.exists(dir_imagefolder):
			continue
		loader = get_loader(name, root_dir, vid_length, crop_size, image_size, batch_size, mode, num_workers)
		loaders[mode] = loader
	return loaders