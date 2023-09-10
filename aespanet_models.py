from quopri import decodestring
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
import math
from utils import *
from style_decorator import StyleDecorator

def size_arrange(x):
	x_w, x_h = x.size(2), x.size(3)

	if (x_w%2) != 0:
		x_w = (x_w//2)*2
	if (x_h%2) != 0:
		x_h = (x_h//2)*2

	if ( x_h > 1024 ) or (x_w > 1024) :
		old_x_w = x_w
		x_w = x_w//2
		x_h = int(x_h*x_w/old_x_w)
	
	return F.interpolate(x, size=(x_w, x_h))

def get_LL_HH(x):
	pooled = torch.nn.functional.avg_pool2d(x, 2)
	#up_pooled = torch.nn.functional.interpolate(pooled, scale_factor=2, mode='nearest')
	up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode='nearest')
	HH = x - up_pooled
	LL = up_pooled
	return HH, LL

def split_image(x, scale=1):
	origin_w, origin_h = x.size(2), x.size(3)
	down_sample = torch.nn.functional.avg_pool2d(x, 2**scale)
	up_sample = torch.nn.functional.interpolate(down_sample, (origin_w, origin_h), mode='bilinear')
	return get_LL_HH(up_sample)


def calc_mean_std(feat, eps=1e-5):
	# eps is a small value added to the variance to avoid divide-by-zero.
	size = feat.size()
	assert (len(size) == 4)
	N, C = size[:2]
	feat_var = feat.view(N, C, -1).var(dim=2) + eps
	feat_std = feat_var.sqrt().view(N, C, 1, 1)
	feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
	return feat_mean, feat_std

def mean_variance_norm(feat):
	size = feat.size()
	mean, std = calc_mean_std(feat)
	normalized_feat = (feat - mean.expand(size)) / std.expand(size)
	return normalized_feat


class Noise(nn.Module):
	def __init__(self, use_noise, sigma=0.2):
		super(Noise, self).__init__()
		self.use_noise = use_noise
		self.sigma = sigma

	def forward(self, x):
		if self.use_noise:
			return x + self.sigma * torch.autograd.Variable(torch.cuda.FloatTensor(x.size()).normal_(), requires_grad=False)
		else:
			return x


class AdaIN(nn.Module):
	def __init__(self):
		super(AdaIN, self).__init__()
	
	def forward(self, content, style, style_strength=1.0, eps=1e-5):
		b, c, h, w = content.size()
		
		content_std, content_mean = torch.std_mean(content.view(b, c, -1), dim=2, keepdim=True)
		style_std, style_mean = torch.std_mean(style.view(b, c, -1), dim=2, keepdim=True)
	
		normalized_content = (content.view(b, c, -1) - content_mean)/(content_std+eps)
		
		stylized_content = (normalized_content * style_std) + style_mean

		output = (1-style_strength)*content + style_strength*stylized_content.view(b, c, h, w)
		return output

class StylizedSANet(nn.Module):
	def __init__(self, in_planes):
		super(StylizedSANet, self).__init__()
		self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
		self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
		self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
		self.sm = nn.Softmax(dim = -1)
		self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
		self.style_transform = AdaIN()

	def forward(self, content, style):
		F = self.f(mean_variance_norm(content))
		G = self.g(mean_variance_norm(style))
		H = self.h(style)
		b, c, h, w = F.size()
		F = F.view(b, -1, w * h).permute(0, 2, 1)
		b, c, h, w = G.size()
		G = G.view(b, -1, w * h)
		S = torch.bmm(F, G)
		S = self.sm(S)
		b, c, h, w = H.size()
		H = H.view(b, -1, w * h)
		O = torch.bmm(H, S.permute(0, 2, 1))
		b, c, h, w = content.size()
		O = O.view(b, c, h, w)
		O = self.out_conv(O)
		#stylized_content = self.style_transform(content, style)
		#O += stylized_content
		O += content
		return O
		#아니면 여기 adaptive하게 설정하는데 일단은 0.5 0.5 로 해볼까?

class Transform(nn.Module):
	def __init__(self, in_planes):
		super(Transform, self).__init__()
		self.sanet4_1 = StylizedSANet(in_planes = in_planes)
		self.sanet5_1 = StylizedSANet(in_planes = in_planes)
		self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
		self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
		
	def forward(self, content4_1, style4_1, content5_1, style5_1):
		#return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))
		return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + nn.functional.interpolate(self.sanet5_1(content5_1, style5_1), size=(self.sanet4_1(content4_1, style4_1).size(2), self.sanet4_1(content4_1, style4_1).size(3)))))


class AdaAttN(nn.Module):
	def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
		super(AdaAttN, self).__init__()
		if key_planes is None:
			key_planes = in_planes
		self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
		self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
		self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
		self.sm = nn.Softmax(dim=-1)
		self.max_sample = max_sample

	def forward(self, content, style, content_key, style_key, seed=None):
		F = self.f(content_key)
		G = self.g(style_key)
		H = self.h(style)
		b, _, h_g, w_g = G.size()
		G = G.view(b, -1, w_g * h_g).contiguous()
		if w_g * h_g > self.max_sample:
			if seed is not None:
				torch.manual_seed(seed)
			index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
			G = G[:, :, index]
			style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
		else:
			style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
		b, _, h, w = F.size()
		F = F.view(b, -1, w * h).permute(0, 2, 1)
		S = torch.bmm(F, G)
		# S: b, n_c, n_s
		S = self.sm(S)
		# mean: b, n_c, c
		mean = torch.bmm(S, style_flat)
		# std: b, n_c, c
		std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
		# mean, std: b, c, h, w
		mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
		std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
		return std * mean_variance_norm(content) + mean, S

class AdaAttn_Transformer(nn.Module):
	def __init__(self, in_planes, key_planes=None, shallow_layer=False):
		super(AdaAttn_Transformer, self).__init__()
		self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
		self.attn_adain_5_1 = AdaAttN(in_planes=in_planes,
										key_planes=key_planes + 512 if shallow_layer else key_planes)
		self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
		self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
		self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

	def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None):
		feature_4_1, attn_4_1 = self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed)
		feature_5_1, attn_5_1 = self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed)
		#stylized_results = self.merge_conv(self.merge_conv_pad(feature_4_1 +  self.upsample5_1(feature_5_1)))
		stylized_results = self.merge_conv(self.merge_conv_pad(feature_4_1 +  nn.functional.interpolate(feature_5_1, size=(feature_4_1.size(2), feature_4_1.size(3) ) ) ) )
		return stylized_results, feature_4_1, feature_5_1
	


class AdaptiveMultiAdaAttN(nn.Module):
	def __init__(self, in_planes, max_sample=256 * 256, query_planes=None, key_planes=None):
		super(AdaptiveMultiAdaAttN, self).__init__()
		if key_planes is None:
			key_planes = in_planes
		self.f = nn.Conv2d(query_planes, key_planes, (1, 1))
		self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
		self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
		self.sm = nn.Softmax(dim=-1)
		self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
		self.max_sample = max_sample

	def forward(self, content, style, content_key, style_key, seed=None):
		F = self.f(content_key)
		G = self.g(style_key)
		H = self.h(style)
		b, _, h_g, w_g = G.size()
		G = G.view(b, -1, w_g * h_g).contiguous()
		if w_g * h_g > self.max_sample:
			if seed is not None:
				torch.manual_seed(seed)
			index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
			G = G[:, :, index]
			style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
		else:
			style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
		b, _, h, w = F.size()
		F = F.view(b, -1, w * h).permute(0, 2, 1)
		S = torch.bmm(F, G)
		# S: b, n_c, n_s
		S = self.sm(S)
		# mean: b, n_c, c
		mean = torch.bmm(S, style_flat)
		# std: b, n_c, c
		std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
		# mean, std: b, c, h, w
		mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
		std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
		return std * mean_variance_norm(content) + mean, S

class AdaptiveMultiAttn_Transformer(nn.Module):
	def __init__(self, in_planes, query_planes=None, key_planes=None, shallow_layer=False):
		super(AdaptiveMultiAttn_Transformer, self).__init__()
		self.attn_adain_4_1 = AdaptiveMultiAdaAttN(in_planes=in_planes, query_planes=query_planes, key_planes=key_planes)
		self.attn_adain_5_1 = AdaptiveMultiAdaAttN(in_planes=in_planes, query_planes=query_planes, key_planes=key_planes+512)
		self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
		self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
		self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

	def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None):
		feature_4_1, attn_4_1 = self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed)
		feature_5_1, attn_5_1 = self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed)
		#stylized_results = self.merge_conv(self.merge_conv_pad(feature_4_1 +  self.upsample5_1(feature_5_1)))
		stylized_results = self.merge_conv(self.merge_conv_pad(feature_4_1 +  nn.functional.interpolate(feature_5_1, size=(feature_4_1.size(2), feature_4_1.size(3) ) ) ) )
		return stylized_results, feature_4_1, feature_5_1, attn_4_1, attn_5_1



class AdaptiveMultiAdaAttN_v2(nn.Module):
	def __init__(self, in_planes, out_planes, max_sample=256 * 256, query_planes=None, key_planes=None):
		super(AdaptiveMultiAdaAttN_v2, self).__init__()
		if key_planes is None:
			key_planes = in_planes
		self.f = nn.Conv2d(query_planes, key_planes, (1, 1))
		self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
		self.h = nn.Conv2d(in_planes, out_planes, (1, 1))
		self.sm = nn.Softmax(dim=-1)
		self.out_conv = nn.Conv2d(in_planes, out_planes, (1, 1))
		self.max_sample = max_sample

	def forward(self, content, style, content_key, style_key, seed=None):
		F = self.f(content_key)
		G = self.g(style_key)
		H = self.h(style)
		b, _, h_g, w_g = G.size()
		G = G.view(b, -1, w_g * h_g).contiguous()
		b, _, h, w = F.size()
		F = F.view(b, -1, w * h).permute(0, 2, 1)
		S = torch.bmm(F, G)
		# S: b, n_c, n_s
		S = self.sm(S)
		
		b, style_c, style_h, style_w = H.size()
		H = torch.nn.functional.interpolate(H, (h_g, w_g), mode='bicubic')
		if h_g * w_g > self.max_sample:
			if seed is not None:
				torch.manual_seed(seed)
			index = torch.randperm(h_g * w_g).to(content.device)[:self.max_sample]
			G = G[:, :, index]
			style_flat = H.view(b, -1, h_g * w_g)[:, :, index].transpose(1, 2).contiguous()
		else:
			style_flat = H.view(b, -1, h_g * w_g).transpose(1, 2).contiguous()
		
		
		# mean: b, n_c, c
		mean = torch.bmm(S, style_flat)
		# std: b, n_c, c
		std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
		# mean, std: b, c, h, w
		_, _, ch, cw = content.size()
		#mean = torch.nn.functional.interpolate(mean.view(b, style_h, style_w, style_c).permute(0, 3, 1, 2).contiguous(), (ch, cw))
		#std = torch.nn.functional.interpolate(std.view(b, style_h, style_w, style_c).permute(0, 3, 1, 2).contiguous(), (ch, cw))
		mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
		std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
		return std * mean_variance_norm(content) + mean, S

class AdaptiveMultiAttn_Transformer_v2(nn.Module):
	def __init__(self, in_planes, out_planes, query_planes=None, key_planes=None, shallow_layer=False):
		super(AdaptiveMultiAttn_Transformer_v2, self).__init__()
		self.attn_adain_4_1 = AdaptiveMultiAdaAttN_v2(in_planes=in_planes, out_planes=out_planes, query_planes=query_planes, key_planes=key_planes)
		self.attn_adain_5_1 = AdaptiveMultiAdaAttN_v2(in_planes=in_planes, out_planes=out_planes, query_planes=query_planes, key_planes=key_planes+512)
		#self.attn_adain_5_1 = AdaptiveMultiAdaAttN_v2(in_planes=in_planes, out_planes=out_planes, query_planes=query_planes+512, key_planes=key_planes+512)
		self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
		self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
		self.merge_conv = nn.Conv2d(out_planes, out_planes, (3, 3))

	def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None):
		feature_4_1, attn_4_1 = self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed)
		feature_5_1, attn_5_1 = self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed)
		#stylized_results = self.merge_conv(self.merge_conv_pad(feature_4_1 +  self.upsample5_1(feature_5_1)))
		
		stylized_results = self.merge_conv(self.merge_conv_pad(feature_4_1 +  nn.functional.interpolate(feature_5_1, size=(feature_4_1.size(2), feature_4_1.size(3) ) ) ) )
		return stylized_results, feature_4_1, feature_5_1, attn_4_1, attn_5_1


class VGGEncoder(nn.Module):
	def __init__(self, vgg):
		super(VGGEncoder, self).__init__()

		self.pad = nn.ReflectionPad2d(1)
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.AvgPool2d(2)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = False)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = False)
		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = False)
		self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices = False)

		###Level0###
		self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
		self.conv0.weight = nn.Parameter(torch.FloatTensor(vgg.modules[0].weight))
		self.conv0.bias = nn.Parameter(torch.FloatTensor(vgg.modules[0].bias))

		###Level1###
		self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
		self.conv1_1.weight = nn.Parameter(torch.FloatTensor(vgg.modules[2].weight))
		self.conv1_1.bias = nn.Parameter(torch.FloatTensor(vgg.modules[2].bias))
		
		self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
		self.conv1_2.weight = nn.Parameter(torch.FloatTensor(vgg.modules[5].weight))
		self.conv1_2.bias = nn.Parameter(torch.FloatTensor(vgg.modules[5].bias))
		
		###Level2###
		self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
		self.conv2_1.weight = nn.Parameter(torch.FloatTensor(vgg.modules[9].weight))
		self.conv2_1.bias = nn.Parameter(torch.FloatTensor(vgg.modules[9].bias))
		
		self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
		self.conv2_2.weight = nn.Parameter(torch.FloatTensor(vgg.modules[12].weight))
		self.conv2_2.bias = nn.Parameter(torch.FloatTensor(vgg.modules[12].bias))
		
		###Level3###
		self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
		self.conv3_1.weight = nn.Parameter(torch.FloatTensor(vgg.modules[16].weight))
		self.conv3_1.bias = nn.Parameter(torch.FloatTensor(vgg.modules[16].bias))

		self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_2.weight = nn.Parameter(torch.FloatTensor(vgg.modules[19].weight))
		self.conv3_2.bias = nn.Parameter(torch.FloatTensor(vgg.modules[19].bias))

		self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_3.weight = nn.Parameter(torch.FloatTensor(vgg.modules[22].weight))
		self.conv3_3.bias = nn.Parameter(torch.FloatTensor(vgg.modules[22].bias))

		self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_4.weight = nn.Parameter(torch.FloatTensor(vgg.modules[25].weight))
		self.conv3_4.bias = nn.Parameter(torch.FloatTensor(vgg.modules[25].bias))

		
		###Level4###
		self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
		self.conv4_1.weight = nn.Parameter(torch.FloatTensor(vgg.modules[29].weight))
		self.conv4_1.bias = nn.Parameter(torch.FloatTensor(vgg.modules[29].bias))

		self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 0)
		self.conv4_2.weight = nn.Parameter(torch.FloatTensor(vgg.modules[32].weight))
		self.conv4_2.bias = nn.Parameter(torch.FloatTensor(vgg.modules[32].bias))

		self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 0)
		self.conv4_3.weight = nn.Parameter(torch.FloatTensor(vgg.modules[35].weight))
		self.conv4_3.bias = nn.Parameter(torch.FloatTensor(vgg.modules[35].bias))

		self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)
		self.conv4_4.weight = nn.Parameter(torch.FloatTensor(vgg.modules[38].weight))
		self.conv4_4.bias = nn.Parameter(torch.FloatTensor(vgg.modules[38].bias))

		###Level5###
		self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 0)
		self.conv5_1.weight = nn.Parameter(torch.FloatTensor(vgg.modules[42].weight))
		self.conv5_1.bias = nn.Parameter(torch.FloatTensor(vgg.modules[42].bias))

	def forward(self, x):
		skips = {}
		for level in [1, 2, 3, 4]:
			x = self.encode(x, skips, level)
		return x

	def encode(self, x, skips):
		is_maxpool=False

		out = self.conv0(x)
		out = self.relu(self.conv1_1(self.pad(out)))
		skips['conv1_1'] = out
		
		out = self.relu(self.conv1_2(self.pad(out)))
		skips['conv1_2'] = out
		resize_w, resize_h = out.size(2), out.size(3)
		pooled_feature = self.pool(out)
		
		#HH = out - F.interpolate(pooled_feature, scale_factor=2, mode='nearest')
		HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
		skips['pool1'] = HH
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool1(out)

		out = self.relu(self.conv2_1(self.pad(pooled_feature)))
		skips['conv2_1'] = out

		out = self.relu(self.conv2_2(self.pad(out)))
		skips['conv2_2'] = out
		resize_w, resize_h = out.size(2), out.size(3)
		pooled_feature = self.pool(out)
		
		#HH = out - F.interpolate(pooled_feature, scale_factor=2, mode='nearest')
		HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
		skips['pool2'] = HH
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool2(out)

		out = self.relu(self.conv3_1(self.pad(pooled_feature)))
		skips['conv3_1'] = out

		out = self.relu(self.conv3_2(self.pad(out)))
		out = self.relu(self.conv3_3(self.pad(out)))
		out = self.relu(self.conv3_4(self.pad(out)))
		skips['conv3_4'] = out
		resize_w, resize_h = out.size(2), out.size(3)
		pooled_feature = self.pool(out)
		#HH = out - F.interpolate(pooled_feature, scale_factor=2, mode='nearest')
		HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
		skips['pool3'] = HH

		##################################
		if is_maxpool:
			pooled_feature = self.maxpool3(out)
		
		out = self.relu(self.conv4_1(self.pad(pooled_feature)))
		skips['conv4_1'] = out

		out = self.relu(self.conv4_2(self.pad(out)))
		out = self.relu(self.conv4_3(self.pad(out)))
		out = self.relu(self.conv4_4(self.pad(out)))
		skips['conv4_4'] = out
		resize_w, resize_h = out.size(2), out.size(3)
		pooled_feature = self.pool(out)
		HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
		skips['pool4'] = HH
		
		####################################
		if is_maxpool:
			pooled_feature = self.maxpool4(out)
		out = self.relu(self.conv5_1(self.pad(pooled_feature)))
		skips['conv5_1'] = out

		return out

	def get_features(self, x, level):
		is_maxpool = False

		out = self.conv0(x)
		out = self.relu(self.conv1_1(self.pad(out)))
		if level == 1:
			return out
		
		out = self.relu(self.conv1_2(self.pad(out)))
		pooled_feature = self.pool(out)
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool1(out)
		out = self.relu(self.conv2_1(self.pad(pooled_feature)))
		if level == 2:
			return out

		out = self.relu(self.conv2_2(self.pad(out)))
		pooled_feature = self.pool(out)
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool2(out)
		out = self.relu(self.conv3_1(self.pad(pooled_feature)))
		if level == 3:
			return out

		out = self.relu(self.conv3_2(self.pad(out)))
		out = self.relu(self.conv3_3(self.pad(out)))
		out = self.relu(self.conv3_4(self.pad(out)))
		pooled_feature = self.pool(out)
		##################################
		if is_maxpool:
			pooled_feature = self.maxpool3(out)
		out = self.relu(self.conv4_1(self.pad(pooled_feature)))
		if level == 4:
			return out
		
		out = self.relu(self.conv4_2(self.pad(out)))
		out = self.relu(self.conv4_3(self.pad(out)))
		out = self.relu(self.conv4_4(self.pad(out)))
		pooled_feature = self.pool(out)
		####################################
		if is_maxpool:
			pooled_feature = self.maxpool4(out)
		out = self.relu(self.conv5_1(self.pad(pooled_feature)))
		if level == 5:
			return out

class VGGDecoder(nn.Module):
	def __init__(self):
		super(VGGDecoder, self).__init__()
		
		self.pad = nn.ReflectionPad2d(1)
		self.relu = nn.ReLU(inplace=False)
		self.adain = AdaIN()
		self.styledecorator = StyleDecorator()

		#self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 0)
		#self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)
		#self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 0)
		#self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 0)
		self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
		self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
		self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
		self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
		self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
		self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
		self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)


		init_weights(self)
		
	def forward(self, x, skips):
		x = self.decode(x, skips)
		

	def decode(self, stylized_feat, content_skips, style_skips):		
		out = self.relu(self.conv4_1(self.pad(stylized_feat)))
		resize_w, resize_h = content_skips['conv3_4'].size(2), content_skips['conv3_4'].size(3)
		unpooled_feat = F.interpolate(out, size=[resize_w, resize_h], mode='nearest')	
		out = unpooled_feat

		##여기에 skip_conneciton 넣어서 stylization boost 하기 (local pattern)
		out = self.relu(self.conv3_4(self.pad(out)))
		out = self.relu(self.conv3_3(self.pad(out)))
		out =  self.relu(self.conv3_2(self.pad(out)))
		#out = feature_wct_simple(out, style_skips['conv3_1'])

		out = self.relu(self.conv3_1(self.pad(out)))
		resize_w, resize_h = content_skips['conv2_2'].size(2), content_skips['conv2_2'].size(3)
		unpooled_feat = F.interpolate(out, size=[resize_w, resize_h], mode='nearest')
		out = unpooled_feat
		
		out = self.relu(self.conv2_2(self.pad(out)))
		#out = feature_wct_simple(out, style_skips['conv2_1'])

		out = self.relu(self.conv2_1(self.pad(out)))
		resize_w, resize_h = content_skips['conv1_2'].size(2), content_skips['conv1_2'].size(3)
		unpooled_feat = F.interpolate(out, size=[resize_w, resize_h], mode='nearest')
		out = unpooled_feat

		out = self.relu(self.conv1_2(self.pad(out)))
		#out = feature_wct_simple(out, style_skips['conv1_1'])
		out = self.conv1_1(self.pad(out))
		return out

	def reconstruct(self, x):
		out = self.relu(self.conv4_1(self.pad(x)))
		out = F.interpolate(out, size=[out.size(2), out.size(3)], mode='nearest')
		out = self.relu(self.conv3_4(self.pad(out)))
		out = self.relu(self.conv3_3(self.pad(out)))
		out =  self.relu(self.conv3_2(self.pad(out)))
		out = self.relu(self.conv3_1(self.pad(out)))
		out = F.interpolate(out, size=[out.size(2), out.size(3)], mode='nearest')
		out = self.relu(self.conv2_2(self.pad(out)))
		out = self.relu(self.conv2_1(self.pad(out)))
		out = F.interpolate(out, size=[out.size(2), out.size(3)], mode='nearest')
		out = self.relu(self.conv1_2(self.pad(out)))
		out = self.conv1_1(self.pad(out))
		return out


class Baseline_net(nn.Module):
	def __init__(self, pretrained_vgg=None):
		super(Baseline_net, self).__init__()

		self.encoder = VGGEncoder(pretrained_vgg)
		self.decoder = VGGDecoder()
		#self.transformer = Transform(in_planes=512)
		self.adain = AdaIN()
		self.decorator = StyleDecorator()
		#query_channels = 512
		#key_channels = 512 + 256 + 128 + 64
		
		#query_channels = 512 
		#key_channels = 64 #512 

		#for v2 transformer
		#query_channels = 512
		query_channels = 512#+256+128+64
		key_channels = 512+256+128+64

		#self.transformer = AdaAttn_Transformer(in_planes=512, key_planes=channels, shallow_layer=True)
		#self.transformer = AdaptiveMultiAttn_Transformer(in_planes=512, query_planes=query_channels, key_planes=key_channels, shallow_layer=True)
		self.transformer = AdaptiveMultiAttn_Transformer_v2(in_planes=512, out_planes=512, query_planes=query_channels, key_planes=key_channels)
		
	
	def get_keys(self, feat_skips, last_layer_idx, type='content', need_shallow=True):
		results = []
		target_conv_layer = 'conv'+str(last_layer_idx)+'_1'
		_, _, h, w = feat_skips[target_conv_layer].shape
		for i in range(1,last_layer_idx+1):
			target_conv_layer = 'conv'+str(i)+'_1'
			if i == last_layer_idx:
				results.append(mean_variance_norm(feat_skips[target_conv_layer]))	
			else:
				results.append(mean_variance_norm(nn.functional.interpolate(feat_skips[target_conv_layer], (h, w))))

		return torch.cat(results, dim=1)

	def adaptive_get_keys(self, feat_skips, start_layer_idx, last_layer_idx, target_feat):
		B, C, th, tw = target_feat.shape
		results = []
		target_conv_layer = 'conv'+str(last_layer_idx)+'_1'
		_, _, h, w = feat_skips[target_conv_layer].shape
		for i in range(start_layer_idx, last_layer_idx+1):
			target_conv_layer = 'conv'+str(i)+'_1'
			if i == last_layer_idx:
				results.append(mean_variance_norm(feat_skips[target_conv_layer]))	
			else:
				results.append(mean_variance_norm(nn.functional.interpolate(feat_skips[target_conv_layer], (h, w))))
		
		return nn.functional.interpolate(torch.cat(results, dim=1), (th, tw) )
		

	def encode(self, x, skips):
		return self.encoder.encode(x, skips)

	def decode(self, stylized_feat, content_skips, style_skips):
		return self.decoder.decode(stylized_feat, content_skips, style_skips)

	def forward(self, content, style, adaptive_alpha, gray_content=None, gray_style=None):		
		
		content_ = size_arrange(content)
		style_ = size_arrange(style)

		####Input####
		content_feat, content_skips = content_, {}
		style_feat, style_skips = style_, {}
		
		####Encode####
		content_feat = self.encode(content_feat, content_skips)
		style_feat = self.encode(style_feat, style_skips)
		
		if gray_content is not None:
			gray_content_ = size_arrange(gray_content)
			gray_content_feat, gray_content_skips = gray_content_, {}
			gray_content_feat = self.encode(gray_content_feat, gray_content_skips)
		if gray_style is not None:
			gray_style_ = size_arrange(gray_style)
			gray_style_feat, gray_style_skips = gray_style_, {}
			gray_style_feat = self.encode(gray_style_feat, gray_style_skips)


		####Style transfer####
		#transformed_feature = self.transformer(content_skips['conv4_1'], style_skips['conv4_1'], content_skips['conv5_1'], style_skips['conv5_1'])

		#Test_Multi_style_level_1
		local_transformed_feature, attn_style_4_1, attn_style_5_1, attn_map_4_1, attn_map_5_1 = self.transformer(content_skips['conv4_1'], style_skips['conv4_1'], content_skips['conv5_1'], style_skips['conv5_1'],
								self.adaptive_get_keys(content_skips, 4, 4, target_feat=content_skips['conv4_1']),
								self.adaptive_get_keys(style_skips, 1, 4, target_feat=style_skips['conv4_1']),
								self.adaptive_get_keys(content_skips, 5, 5, target_feat=content_skips['conv5_1']),
								self.adaptive_get_keys(style_skips, 1, 5, target_feat=style_skips['conv5_1']))
		
		if gray_content is not None:
			global_transformed_feat = feature_wct_simple(gray_content_skips['conv4_1'], gray_style_skips['conv4_1'])
			#global_transformed_feat = self.adain(gray_content_skips['conv4_1'], gray_style_skips['conv4_1'])
			#global_transformed_feat = self.decorator(gray_content_skips['conv4_1'], gray_style_skips['conv4_1'])
		else:	
			global_transformed_feat = feature_wct_simple(content_skips['conv4_1'], style_skips['conv4_1'])
			#global_transformed_feat = self.adain(content_skips['conv4_1'], style_skips['conv4_1'])
			#global_transformed_feat = self.decorator(content_skips['conv4_1'], style_skips['conv4_1'])

		
		transformed_feature = global_transformed_feat*(1-adaptive_alpha.unsqueeze(-1).unsqueeze(-1)) + adaptive_alpha.unsqueeze(-1).unsqueeze(-1)*local_transformed_feature
		#transformed_feature = global_transformed_feat + local_transformed_feature
		
		## Ablation study ##
		#transformed_feature = local_transformed_feature

		####Decode####
		stylized_image = self.decode(transformed_feature, content_skips, style_skips)
		
		return stylized_image, attn_style_4_1, attn_style_5_1, attn_map_4_1, attn_map_5_1


#################################
#####MultiScaleDiscriminator#####
#################################
class MultiScaleImageDiscriminator(nn.Module):
	def __init__(self, nc=3, ndf=64):
		super(MultiScaleImageDiscriminator, self).__init__()
		self.nc = nc
		self.output_dim = 1

		self.conv1 = nn.Sequential(
			#256->128
			spectral_norm(nn.Conv2d(self.nc, ndf, 4, 2, 1, bias=False)),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.skip_out1 = nn.Sequential(
			nn.Conv2d(ndf, 32, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AdaptiveAvgPool2d(8),
			nn.Conv2d(32, self.output_dim, 3, 1, 1, bias=False)
		)
		
		self.conv2 = nn.Sequential(
			#128->64
			spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
			#nn.BatchNorm2d(ndf*2),
			nn.InstanceNorm2d(ndf*2),
			nn.LeakyReLU(0.2, inplace=True),
		)
		
		self.conv3 = nn.Sequential(
			#64->32
			spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
			#nn.BatchNorm2d(ndf*4),
			nn.InstanceNorm2d(ndf*4),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.skip_out2 = nn.Sequential(
			nn.Conv2d(ndf*4, 32, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.AdaptiveAvgPool2d(8),
			nn.Conv2d(32, self.output_dim, 3, 1, 1, bias=False)
		)
		
		self.conv4 = nn.Sequential(
			#32->16
			spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),
			#nn.BatchNorm2d(ndf*8),
			nn.InstanceNorm2d(ndf*8),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.conv5 = nn.Sequential(
			#16->8
			spectral_norm(nn.Conv2d(ndf*8, self.output_dim, 4, 2, 1, bias=False)),
		)

		init_weights(self)

	def forward(self, input):
		out = self.conv1(input)
		skip_out1 = self.skip_out1(out)
		out = self.conv3(self.conv2(out))
		skip_out2 = self.skip_out2(out)
		out = self.conv5(self.conv4(out))
		out = ((out+skip_out1+skip_out2)*1/3).squeeze()
		#out = ((out+skip_out1+skip_out2)).squeeze()
		
		return out
