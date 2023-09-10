import os, sys, random, cv2, pdb, csv

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import imageio
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import natsort
Image.MAX_IMAGE_PIXELS = 1000000000
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import pdb
def xdog(_image, epsilon=0.5, phi=10, k=1.4, tau=1, sigma=0.5):
	_image = gaussian_filter(_image, 0.7)
	gauss1 = gaussian_filter(_image, sigma)
	gauss2 = gaussian_filter(_image, sigma*k)
	D = gauss1 - tau*gauss2
	U = D/255

	for i in range(0, len(U)):
		for j in range(0, len(U[0])):
			U[i][j] = abs(1-U[i][j])

	for i in range(0, len(U)):
		for j in range(0, len(U[0])):
			if U[i][j] >= epsilon:
				U[i][j] = 1
			else:
				ht = np.tanh(phi*(U[i][j]-epsilon))
				U[i][j] = 1+ht
	
	return U*255

class MSCOCO(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False):
		super(MSCOCO, self).__init__()

		#self.file_names = sorted(os.listdir(root_path))
		self.root_path = root_path
		self.file_names = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(self.root_path,'train2017')) for f in files if f.endswith('jpg') or f.endswith('png')])
		self.file_names += sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(self.root_path,'val2017')) for f in files if f.endswith('jpg') or f.endswith('png')])
		self.transform = _transformer(imsize, cropsize, cencrop)

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			image = Image.open(self.file_names[index]).convert("RGB")
		except:
			print(self.file_names[index])

		return self.transform(image)

class WiKiART(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False):
		super(WiKiART, self).__init__()

		#self.file_names = sorted(os.listdir(root_path))
		#self.file_names = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(self.root_path) for f in files if f.endswith('jpg') or f.endswith('png')])
		self.root_path = root_path
		self.file_names = []
		self.transform = _transformer(imsize, cropsize, cencrop)
		art_path = '../../dataset/wikiart_csv'
		self.csv_files = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(art_path) for f in files if (f.split('_')[-1]).split('.')[0] == 'train' ]) 
		for csv_file in self.csv_files:
			f = open(csv_file, 'r', encoding='utf-8')
			rdr = csv.reader(f)
			for line in rdr:
				self.file_names.append(os.path.join(self.root_path, line[0]))

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			image = Image.open(self.file_names[index]).convert("RGB")
		except:
			print(self.file_names[index])
		return self.transform(image)

class Webtoon(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False):
		super(Webtoon, self).__init__()

		#self.file_names = sorted(os.listdir(root_path))
		self.root_path = root_path
		#self.file_names = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(self.root_path,'train2017')) for f in files if f.endswith('jpg') or f.endswith('png')])
		#self.file_names += sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(self.root_path,'val2017')) for f in files if f.endswith('jpg') or f.endswith('png')])
		self.file_names = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(self.root_path,'faces')) for f in files if f.endswith('jpg') or f.endswith('png')])
		self.transform = _transformer(imsize, cropsize, cencrop)

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			image = Image.open(self.file_names[index]).convert("RGB")
		except:
			print(self.file_names[index])

		return self.transform(image)


class TestDataset(torch.utils.data.Dataset):
	def __init__(self, imsize=None, cropsize=None, cencrop=False):
		super(TestDataset, self).__init__()

		self.transform = _transformer(imsize, cropsize, cencrop)

		photo_path = '../../dataset/MSCoCo'
		self.photo_file_names = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(os.path.join(photo_path, 'test2017')) for f in files if f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg') ])
		

		art_root_path = '../../dataset/wikiart'
		self.art_file_names = []
		art_path = '../../dataset/wikiart_csv'
		self.csv_files = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(art_path) for f in files if (f.split('_')[-1]).split('.')[0] == 'val' ]) 
		for csv_file in self.csv_files:
			f = open(csv_file, 'r', encoding='utf-8')
			rdr = csv.reader(f)
			for line in rdr:
				self.art_file_names.append(os.path.join(art_root_path, line[0]))
		
	def __len__(self):
		return len(self.photo_file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			photo_image = Image.open(self.photo_file_names[index]).convert("RGB")
			art_image = Image.open(self.art_file_names[index]).convert("RGB")
		except:
			print(self.photo_file_names[index])
			print(self.art_file_names[index])
		return self.transform(photo_image), self.transform(art_image)


class Art_Transfer_TestDataset(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False):
		super(Art_Transfer_TestDataset, self).__init__()

		self.transform = _transformer()
		art_root_path = '../../dataset/wikiart'
		self.art_file_names = []
		art_path = '../../dataset/wikiart_csv'
		self.csv_files = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(art_path) for f in files if (f.split('_')[-1]).split('.')[0] == 'val' ]) 
		for csv_file in self.csv_files:
			f = open(csv_file, 'r', encoding='utf-8')
			rdr = csv.reader(f)
			for line in rdr:
				self.art_file_names.append(os.path.join(art_root_path, line[0]))
		
	def __len__(self):
		return len(self.art_file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			art_image = Image.open(self.art_file_names[index]).convert("RGB")
		except:
			print(self.art_file_names[index])
		return self.transform(art_image)

class Transfer_TestDataset(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False, type='photo', is_test=False):
		super(Transfer_TestDataset, self).__init__()

		#self.file_names = sorted(os.listdir(root_path))
		self.root_path = root_path
		if is_test:
			self.transform = _transformer()#_transformer()#_transformer()# #여기도 나중에 코드정리할 때 분리해주기
			#self.transform = _transformer(imsize)#_transformer()# #여기도 나중에 코드정리할 때 분리해주기
		else:
			#self.transform = _transformer(imsize, cropsize, cencrop)#_transformer()# #여기도 나중에 코드정리할 때 분리해주기
			self.transform = _transformer(imsize)
			#self.transform = _transformer(imsize, cropsize, cencrop)
		
		if type =='photo':
			self.file_names = (sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(self.root_path) for f in files if f.endswith('jpg') or f.endswith('png') or f.endswith('JPG') or f.endswith('jpeg')]))
		else:
			self.file_names = natsort.natsorted(sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(self.root_path) for f in files if f.endswith('jpg') or f.endswith('png') or f.endswith('JPG') or f.endswith('jpeg')]))
		
	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			image = Image.open(self.file_names[index]).convert("RGB")
		except:
			print(self.file_names[index])
			image = Image.open(self.file_names[index-1]).convert("RGB")
		return self.transform(image)
		

class Analysis_TestDataset(torch.utils.data.Dataset):
	def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False, type='photo', is_test=False):
		super(Analysis_TestDataset, self).__init__()

		#self.file_names = sorted(os.listdir(root_path))
		self.root_path = root_path
		if is_test:
			#self.transform = _transformer()#_transformer()#_transformer()# #여기도 나중에 코드정리할 때 분리해주기
			self.transform = _transformer(imsize)#_transformer()# #여기도 나중에 코드정리할 때 분리해주기
		else:
			#self.transform = _transformer(imsize, cropsize, cencrop)#_transformer()# #여기도 나중에 코드정리할 때 분리해주기
			self.transform = _transformer(imsize, cropsize, cencrop)
		
		if type =='photo':
			self.file_names = (sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(self.root_path) for f in files if f.endswith('jpg') or f.endswith('png') or f.endswith('JPG') or f.endswith('jpeg')]))
		else:
			self.file_names = natsort.natsorted(sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(self.root_path) for f in files if f.endswith('jpg') or f.endswith('png') or f.endswith('JPG') or f.endswith('jpeg')]))
		
	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, index):
		#image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
		try:
			image = Image.open(self.file_names[index]).convert("RGB")
		except:
			print(self.file_names[index])
			image = Image.open(self.file_names[index-1]).convert("RGB")
		return self.transform(image), self.file_names[index]


def lastest_arverage_value(values, length=100):
	if len(values) < length:
		length = len(values)
	return sum(values[-length:])/length


def _normalizer(denormalize=False):
	# set Mean and Std of RGB channels of IMAGENET to use pre-trained VGG net
	MEAN = [0.485, 0.456, 0.406]
	STD = [0.229, 0.224, 0.225]    
	
	if denormalize:
		MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
		STD = [1/std for std in STD]
	
	return transforms.Normalize(mean=MEAN, std=STD)

def _transformer(imsize=None, cropsize=None, cencrop=False):
	normalize = _normalizer()
	transformer = []
	w, h = imsize, imsize
	if imsize:
		transformer.append(transforms.Resize(imsize))
	if cropsize:
		if cencrop:
			transformer.append(transforms.CenterCrop(cropsize))
		else:
			transformer.append(transforms.RandomCrop(cropsize))

	transformer.append(transforms.ToTensor())
	transformer.append(normalize)
	return transforms.Compose(transformer)

def imsave(tensor, path, nrow=4, npadding=0):
	denormalize = _normalizer(denormalize=True)
	if tensor.is_cuda:
		tensor = tensor.cpu()
	tensor = torchvision.utils.make_grid(tensor, nrow=nrow, padding=npadding)
	torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
	return None

def imsave_no_norm(tensor1, tensor2, path, nrow=4, npadding=0):
	normalize = _normalizer(denormalize=False)
	denormalize = _normalizer(denormalize=True)
	if tensor1.is_cuda:
		tensor1 = tensor1.cpu()
		tensor2 = tensor2.cpu()
	tensor1 = torchvision.utils.make_grid(tensor1, nrow=nrow, padding=npadding)
	tensor2 = torchvision.utils.make_grid(tensor2, nrow=nrow, padding=npadding)
	torchvision.utils.save_image(denormalize(tensor1).clamp_(0.0, 1.0)+normalize(tensor2), path)
	#import pdb;pdb.set_trace()
	return None

def denorm(tensor, nrow=4, npadding=0):
	denormalize = _normalizer(denormalize=True)
	if tensor.is_cuda:
		tensor = tensor.cpu()
	tensor = torchvision.utils.make_grid(tensor, nrow=nrow, padding=npadding)
	return (denormalize(tensor).clamp_(0.0, 1.0))

def imload(path, imsize=None, cropsize=None, cencrop=False):
	transformer = _transformer(imsize, cropsize, cencrop)
	return transformer(Image.open(path).convert("RGB")).unsqueeze(0)

def imshow(tensor):
	denormalize = _normalizer(denormalize=True)    
	if tensor.is_cuda:
		tensor = tensor.cpu()    
	tensor = torchvision.utils.make_grid(denormalize(tensor.squeeze(0)))
	image = transforms.functional.to_pil_image(tensor.clamp_(0.0, 1.0))
	return image

def maskload(path):
	mask = Image.open(path).convert('L')
	return transforms.functional.to_tensor(mask).unsqueeze(0)

