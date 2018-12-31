import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				images.append(path)

	return images


def random_choice(trimap, crop_size=(320, 320)):
	crop_height, crop_width = crop_size

	(h, w) = trimap.size
	x = np.random.randint(int(crop_height / 2), h - int(crop_height / 2))
	y = np.random.randint(int(crop_width / 2), w - int(crop_width / 2))
	return x, y


def safe_crop(img, x, y):
	region = (x - 160, y - 160, x + 160, y + 160)
	crop_img = img.crop(region)

	return crop_img


def RGB_np2Tensor(imgIn, imgTar):
	ts = (2, 0, 1)
	imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
	imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)
	return imgIn, imgTar


def augment(imgIn, imgTar):
	if random.random() < 0.3:
		imgIn = imgIn[:, ::-1, :]
		imgTar = imgTar[:, ::-1, :]
	if random.random() < 0.3:
		imgIn = imgIn[::-1, :, :]
		imgTar = imgTar[::-1, :, :]
	return imgIn, imgTar


def getPatch(imgIn, imgTar, args, scale):
	(ih, iw, c) = imgIn.shape
	(th, tw) = (scale * ih, scale * iw)
	tp = args.patchSize
	ip = tp // scale
	ix = random.randrange(0, iw - ip + 1)
	iy = random.randrange(0, ih - ip + 1)
	(tx, ty) = (scale * ix, scale * iy)
	imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
	imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]
	return imgIn, imgTar


class DIV2K(data.Dataset):
	def __init__(self, args):
		self.args = args
		self.scale = args.scale
		apath = args.dataDir
		dirHR = 'HR'
		dirLR = 'LR'
		self.dirIn = os.path.join(apath, dirLR)
		self.dirTar = os.path.join(apath, dirHR)
		self.fileList = os.listdir(self.dirHR)
		self.nTrain = len(self.fileList)

	def __getitem__(self, idx):
		scale = self.scale
		nameIn, nameTar = self.getFileName(idx)
		imgIn = cv2.imread(nameIn)
		imgTar = cv2.imread(nameTar)
		if self.args.need_patch:
			imgIn, imgTar = getPatch(imgIn, imgTar, self.args, scale)
		imgIn, imgTar = augment(imgIn, imgTar)
		return RGB_np2Tensor(imgIn, imgTar)

	def __len__(self):
		return self.nTrain

	def getFileName(self, idx):
		name = self.fileList[idx]
		nameTar = os.path.join(self.dirTar, name)
		name = name[0:-4] + 'x3' + '.png'
		nameIn = os.path.join(self.dirIn, name)
		return nameIn, nameTar


class DetailNetDataLoader(object):
	def __init__(self, opt):
		self.dataset = InputDataset(opt.dataroot)
		self.dataloader = torch.utils.data.DataLoader(
			self.dataset,
			shuffle=True,
			batch_size=opt.batch_size,
			num_workers=4,
			drop_last=True
		)

	def load_data(self):
		return self

	def __len__(self):
		return len(self.dataset)

	def __iter__(self):
		for i, data in enumerate(self.dataloader):
			yield data


class InputDataset(data.Dataset):
	def __init__(self, dataroot):
		super(InputDataset, self).__init__()
		self.root = dataroot
		self.dir_input = os.path.join(dataroot, 'merged')
		self.dir_trimap = os.path.join(dataroot, 'trimap')
		self.dir_alpha = os.path.join(dataroot, 'alpha')

		self.input_paths = make_dataset(self.dir_input)
		self.trimap_paths = make_dataset(self.dir_trimap)
		self.alpha_paths = make_dataset(self.dir_alpha)

		self.input_paths = sorted(self.input_paths)
		self.trimap_paths = sorted(self.trimap_paths)

		self.alpha_paths = sorted(self.alpha_paths)

		self.input_size = len(self.input_paths)
		self.trimap_size = len(self.trimap_paths)
		self.alpha_size = len(self.alpha_paths)

		# self.dir_bg = os.path.join(dataroot, 'bg')
		# self.dir_fg = os.path.join(dataroot, 'fg')
		# self.bg_paths = make_dataset(self.dir_bg)
		# self.fg_paths = make_dataset(self.dir_fg)
		# self.bg_paths = sorted(self.bg_paths)
		# self.fg_paths = sorted(self.fg_paths)

		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def __getitem__(self, index):
		# print(self.trimap_paths)

		# print(index, index//100, self.input_paths[index])

		input_path = self.input_paths[index]
		trimap_path = self.trimap_paths[index // 100]
		alpha_path = self.alpha_paths[index // 100]

		# print("input: " + input_path + "\n", "trimap: " + trimap_path + "\n", "alpha: " + alpha_path + "\n")
		input_img = Image.open(input_path).convert('RGB')
		trimap_img = Image.open(trimap_path)
		alpha_img = Image.open(alpha_path)

		x, y = random_choice(trimap_img)

		input_img = safe_crop(input_img, x, y)
		trimap_img = safe_crop(trimap_img, x, y)
		alpha_img = safe_crop(alpha_img, x, y)

		I = self.transform(input_img)
		T = self.transform(trimap_img)
		A = self.transform(alpha_img)

		# bg_path = self.bg_paths[index]
		# fg_path = self.fg_paths[index // 100]
		# bg_img = Image.open(bg_path).convert('RGB')
		# fg_img = Image.open(fg_path).convert('RGB')
		# bg_img = safe_crop(bg_img, x, y)
		# fg_img = safe_crop(fg_img, x, y)
		# B = self.transform(bg_img)
		# F = self.transform(fg_img)

		# return {'I': I, 'T': T, 'A': A,
		# 		'B': B, 'F': F}

		return {'I': I, 'T': T, 'A': A}

	def __len__(self):
		return self.input_size
