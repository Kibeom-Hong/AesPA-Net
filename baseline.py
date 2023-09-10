import os, math, random, pdb, time, timeit, cv2
#from requests import patch

import torch, torchfile
import torchvision.utils as utils
import torch.nn.functional as F
from torch.distributions.beta import Beta

from aespanet_models import Baseline_net
from aespanet_models import MultiScaleImageDiscriminator
from aespanet_models import *

from utils import *
from utils import attn_visualization_all
from contextual_utils import contextual_loss_v2
from data.dataset_util import *
from hist_loss import RGBuvHistBlock
import wandb


def calc_histogram_loss(A, B, histogram_block):
	input_hist = histogram_block(A)
	target_hist = histogram_block(B)
	histogram_loss = (1/np.sqrt(2.0) * (torch.sqrt(torch.sum(
		torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) / 
		input_hist.shape[0])

	return histogram_loss

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

def get_HH_LL(x):
	pooled = torch.nn.functional.avg_pool2d(x, 2)
	#up_pooled = torch.nn.functional.interpolate(pooled, scale_factor=2, mode='nearest')
	up_pooled = torch.nn.functional.interpolate(pooled, scale_factor=2, mode='bilinear')
	HH = x - up_pooled
	LL = up_pooled
	return HH, LL




class Baseline(object):
	def __init__(self, args):
		super(Baseline, self).__init__()
		self.imsize = args.imsize #(512,1024)
		self.batch_size = args.batch_size
		self.cencrop = args.cencrop
		self.cropsize = args.cropsize
		self.num_workers = args.num_workers
		self.content_dir = args.content_dir
		self.style_dir = args.style_dir
		self.lr = args.lr
		self.train_result_dir = args.train_result_dir
		self.comment = args.comment
		self.max_iter = args.max_iter
		self.check_iter = args.check_iter
		self.args = args

		self.hist = RGBuvHistBlock(insz=64, h=256, intensity_scale=True, method='inverse-quadratic').cuda()

		#######################################
		####          Model Load           ####
		#######################################
		pretrained_vgg = torchfile.load('./baseline_checkpoints/vgg_normalised_conv5_1.t7')
		self.network = Baseline_net(pretrained_vgg=pretrained_vgg)
		self.discriminator = MultiScaleImageDiscriminator()
		
		self.network.cuda()
		self.discriminator.cuda()
		
		#######################################
		####   Loss function, Optimizer    ####
		#######################################
		for param in self.network.encoder.parameters():
			param.requires_grad = False
		
		betas=(0.5, 0.999)
		self.optim = torch.optim.Adam(
			[{'params': self.network.decoder.parameters()},{'params': self.network.transformer.parameters()}],
			lr = self.lr,
			betas=betas
			)

		self.dis_optim = torch.optim.Adam(
			filter(lambda p: p.requires_grad, self.discriminator.parameters()),
			lr = self.lr,
			betas=betas,
			weight_decay=0.00001
			)

		self.d_reg_every = 16
		self.d_reg_ratio = self.d_reg_every / (self.d_reg_every+1)

		self.MSE_loss = torch.nn.MSELoss().cuda()
		self.MSE_instance_loss = torch.nn.MSELoss(reduction='none').cuda()
		self.bce_loss = torch.nn.BCEWithLogitsLoss().cuda()
		self.cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda()

		self.tv_weight = 1e-6 #1

		self.result_st_dir = os.path.join(self.train_result_dir, self.comment, 'log')
		os.makedirs(self.result_st_dir, exist_ok=True)

	
	def invert_gray(self, input):
		return torchvision.transforms.functional.rgb_to_grayscale(input).repeat(1,3,1,1)

	def calc_content_loss(self, input, target, norm = False):
		if(norm == False):
			return self.MSE_loss(input, target)
		else:
			return self.MSE_loss(mean_variance_norm(input), mean_variance_norm(target))

	def calc_style_loss(self, input, target, dim=False):
		input_mean, input_std = calc_mean_std(input)
		target_mean, target_std = calc_mean_std(target)
		if dim==True:
			return torch.mean(self.MSE_instance_loss(input_mean, target_mean) + self.MSE_instance_loss(input_std, target_std), dim=(1,2))
		return self.MSE_loss(input_mean, target_mean) + self.MSE_loss(input_std, target_std)

	def calc_style_loss_centered_gram(self, input, target, dim=False):
		return self.MSE_loss(gram_matrix(input-torch.mean(input)), gram_matrix(target-torch.mean(target)))

	def adjust_learning_rate(self, optimizer, iteration_count):
		lr = self.lr / (1.0 + 5e-5 * iteration_count)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def extract_image_patches(self, x, kernel, stride=1):
		b,c,h,w = x.shape
		
		# Extract patches
		patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
		patches = patches.contiguous().view(b, c, -1, kernel, kernel)
		patches = patches.permute(0,2,1,3,4).contiguous()
		
		return patches.view(b, -1, c, kernel, kernel)
		#return patches.view(b, number_of_patches, c, h, w)

	def local_global_gram_loss(self, image, level, ratio, path):
		if level == 0:
			encoded_features = image
		else:
			encoded_features = self.network.encoder.get_features(image, level) # B x C x W x H
		global_gram = gram_matrix(encoded_features)

		B, C, w, h = encoded_features.size()
		target_w, target_h = w//ratio, h//ratio
		
		stride = target_w
		patches = self.extract_image_patches(encoded_features, target_w, stride)
		_, patches_num, _, _, _ = patches.size()
		cos = torch.nn.CosineSimilarity(eps=1e-6)
		cos_gram_statistic = []
		
		for idx in range(B):
			if patches_num < 2:
				continue
			cos_gram = []
			for patch in range(0, patches_num):
				cos_gram.append( cos( global_gram, gram_matrix(patches[idx][patch].unsqueeze(0))).mean() )
			if idx == 0:
				cos_gram_statistic = torch.tensor(cos_gram)
			elif idx == 1:
				cos_gram_statistic = torch.stack([cos_gram_statistic, torch.tensor(cos_gram)])
			else:
				cos_gram_statistic = torch.cat([cos_gram_statistic, torch.tensor(cos_gram).unsqueeze(0)])
				
		return cos_gram_statistic

	def inter_gram_loss(self, image, level, ratio, path):
		if level == 0:
			encoded_features = image
		else:
			encoded_features = self.network.encoder.get_features(image, level) # B x C x W x H
		global_gram = gram_matrix(encoded_features)

		B, C, w, h = encoded_features.size()
		target_w, target_h = w//ratio, h//ratio
		#assert target_w==target_h
		
		patches = self.extract_image_patches(encoded_features, target_w, target_h)
		_, patches_num, _, _, _ = patches.size()
		cos = torch.nn.CosineSimilarity(eps=1e-6)
		
		cos_gram_statistic = {}
		comb = torch.combinations(torch.arange(patches_num), r=2)
		if patches_num >= 10:
			##You can control the probability of sampling pathces##
			##Please refer to the supplementary##
			sampling_num = int(comb.size(0)*0.1) #10%
			#sampling_num = int(comb.size(0)*0.3) #30%
			#sampling_num = int(comb.size(0)*0.7) #70%
		else:
			sampling_num = comb.size(0)
		for idx in range(B):
			if patches_num < 2:
				continue
			cos_gram = []
			
			for idxes in random.choices(list(comb), k=sampling_num):
				cos_gram.append( cos( gram_matrix(patches[idx][idxes[0]].unsqueeze(0)), gram_matrix(patches[idx][idxes[1]].unsqueeze(0))).mean().item() )	
				#cos_gram.append( cos( global_gram, gram_matrix(patches[idx][patch].unsqueeze(0))).mean().item() )	
			cos_gram_statistic[idx] = cos_gram
			
		
		return cos_gram_statistic


	def adaptive_gram_weight(self, image, level, ratio):
		if level == 0:
			encoded_features = image
		else:
			encoded_features = self.network.encoder.get_features(image, level) # B x C x W x H
		global_gram = gram_matrix(encoded_features)

		B, C, w, h = encoded_features.size()
		target_w, target_h = w//ratio, h//ratio
		#assert target_w==target_h
		
		patches = self.extract_image_patches(encoded_features, target_w, target_h)
		_, patches_num, _, _, _ = patches.size()
		cos = torch.nn.CosineSimilarity(eps=1e-6)
		
		intra_gram_statistic = []
		inter_gram_statistic = []
		comb = torch.combinations(torch.arange(patches_num), r=2)
		if patches_num >= 10:
			sampling_num = int(comb.size(0)*0.05)
		else:
			sampling_num = comb.size(0)
		for idx in range(B):
			if patches_num < 2:
				continue
			cos_gram = []
			
			for patch in range(0, patches_num):
				
				cos_gram.append(cos( global_gram, gram_matrix(patches[idx][patch].unsqueeze(0))).mean().item() )
				
			intra_gram_statistic.append(torch.tensor(cos_gram))
			
			cos_gram = []
			for idxes in random.choices(list(comb), k=sampling_num):
				cos_gram.append( cos( gram_matrix(patches[idx][idxes[0]].unsqueeze(0)), gram_matrix(patches[idx][idxes[1]].unsqueeze(0))).mean().item() )	
				
			inter_gram_statistic.append(torch.tensor(cos_gram))
			
		intra_gram_statistic = torch.stack(intra_gram_statistic).mean(dim=1)
		inter_gram_statistic = torch.stack(inter_gram_statistic).mean(dim=1)
		results = (intra_gram_statistic + inter_gram_statistic)/2
		
		##For boosting value
		results = ( 1/(1+torch.exp(-10*(results-0.6))) )

		return results


	def patch_wise_cos(self, image, level, patch_size, stride, path):
		encoded_features = self.network.encoder.get_features(image, level) # B x C x W x H
		patches = self.extract_image_patches(encoded_features, patch_size, stride)
		B, patches_num, C, w, h = patches.size()
		cos = torch.nn.CosineSimilarity(eps=1e-6)
		total_gram = {}
		for idx in range(B):
			fix_patch = patches[idx][0]
			gram_statistic = []
			if patches_num < 2:
				continue
			for patch in range(1, patches_num):
				gram_statistic.append(cos( gram_matrix(fix_patch.unsqueeze(0)), gram_matrix(patches[idx][patch].unsqueeze(0))).mean().item() )
			
			plt.figure()
			plt.hist(gram_statistic)
			plt.savefig(os.path.join(path, str(level)+'_'+str(idx)+'_histogram.png'))

			plt.figure()
			plt.plot(gram_statistic)
			plt.savefig(os.path.join(path, str(level)+'_'+str(idx)+'_plot.png'))

			total_gram[idx] = ( sum(gram_statistic) / len(gram_statistic) )
		
		return total_gram


	
	def proposed_local_gram_loss_v2(self, stylization, style, alpha):
		local_style_loss = 0
		
		B, C, th, tw = style.size()
		for batch in range(B):
			window_size = min(int(2 ** int((9/8-alpha[batch])*8+4)), 256)
			for level in [4, 5]:

				stylization_patches = self.network.encoder.get_features(self.extract_image_patches(stylization[batch:batch+1], window_size, window_size).squeeze(0), level)
				style_patches = self.network.encoder.get_features(self.extract_image_patches(style[batch:batch+1], window_size, window_size).squeeze(0), level)
				
				gram_stylization_patches = gram_matrix(stylization_patches -torch.mean(stylization_patches))
				gram_style_patches = gram_matrix(style_patches -torch.mean(style_patches))

				local_style_loss += self.MSE_loss(gram_stylization_patches, gram_style_patches)
				
		return local_style_loss / B / 2

	
	def styleaware_regularizer(self, stylized_results, style):
		level=4
		ratios = [2, 4, 8, 16, 32]
		reg = 0
		for ratio in ratios:
			stylized_std = torch.nn.functional.log_softmax( self.local_global_gram_loss(stylized_results, level, ratio, '') )
			style_std = torch.nn.functional.log_softmax( self.local_global_gram_loss(style, level, ratio, '') )
			reg += nn.KLDivLoss(reduction='sum', log_target=True)(stylized_std, style_std)
		
		return reg


	def contrastive_styleaware_regularizer(self, content, style, style_adaptive_alpha, gray_content, gray_style):
		level=4
		ratios = [2, 4, 8, 16, 32]
		reg = 0
		tau = 0.2
		for B in range(self.batch_size):
			style_ = torch.vstack([style, style[B].unsqueeze(0).repeat(self.batch_size-1, 1, 1, 1)])
			content_ = torch.vstack([content, torch.cat((content[:B],content[B+1:]))])
			style_adaptive_alpha_ = torch.vstack([style_adaptive_alpha, style_adaptive_alpha[B].unsqueeze(0).repeat(self.batch_size-1, 1)])
			ground_truth = torch.nn.functional.one_hot(torch.arange(B,B+1), num_classes=self.batch_size)
			ground_truth_ = torch.hstack([ground_truth, torch.ones_like(ground_truth)])[:,:-1].squeeze()
			stylized_results, _, _, _, _ = self.network(content_, style_, style_adaptive_alpha_, self.invert_gray(content_), self.invert_gray(style_))
			
			for ratio in ratios:
				stylized_std =  self.local_global_gram_loss(stylized_results, level, ratio, '') 
				style_std =  self.local_global_gram_loss(style, level, ratio, '') 
				#out = torch.mm(stylized_std, style_std.transpose(1,0)) / tau
				out = torch.mm(torch.mean(stylized_std, 1, True), torch.mean(style_std, 1, True).transpose(1,0)) / tau
				reg +=self.cross_entropy_loss(out, ground_truth_.squeeze())

			
			del content_, style_, style_adaptive_alpha_, ground_truth_, stylized_results
			torch.cuda.empty_cache()
		
		return reg

	
	def calc_adaptive_alpha(self, style, gray_style):
		color_style_alpha_8 = (((self.adaptive_gram_weight(style, 1, 8)+self.adaptive_gram_weight(style, 2, 8)+self.adaptive_gram_weight(style, 3, 8) ) /3 ).unsqueeze(1).cuda() )
		gray_style_alpha_8 = ((self.adaptive_gram_weight(gray_style, 1, 8)+self.adaptive_gram_weight(gray_style, 2, 8)+self.adaptive_gram_weight(gray_style, 3, 8) ) /3 ).unsqueeze(1).cuda()

		color_style_alpha_16 = (((self.adaptive_gram_weight(style, 1, 16)+self.adaptive_gram_weight(style, 2, 16)+self.adaptive_gram_weight(style, 3, 16) ) /3 ).unsqueeze(1).cuda() )
		gray_style_alpha_16 = ((self.adaptive_gram_weight(gray_style, 1, 16)+self.adaptive_gram_weight(gray_style, 2, 16)+self.adaptive_gram_weight(gray_style, 3, 16) ) /3 ).unsqueeze(1).cuda()

		return (color_style_alpha_16 + gray_style_alpha_16 + color_style_alpha_8 + gray_style_alpha_8 )/4


	def train(self):

		########################
		wandb.init(project="New_style_transfer")
		wandb.run.name = self.comment
		wandb.config.update(self.args)
		########################

		content_set = MSCOCO(self.content_dir, (self.imsize, self.imsize), self.cropsize, self.cencrop)
		art_reference_set = WiKiART(self.style_dir, (self.imsize, self.imsize), self.cropsize, self.cencrop)
		
		content_loader = torch.utils.data.DataLoader(content_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
		art_reference_loader = torch.utils.data.DataLoader(art_reference_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)


		N = content_set.__len__()
		self.content_iter = iter(content_loader)
		self.art_iter = iter(art_reference_loader)

		annealing_factor = (np.linspace(0, 1.5, self.max_iter))

		for iteration in range(self.max_iter):
			#self.adjust_learning_rate(self.optim, iteration_count=iteration)
			empty_segment = np.asarray([])
			try:
				content = next(self.content_iter).cuda()
				style = next(self.art_iter).cuda()
				content = size_arrange(content)
				style = size_arrange(style)
			except:
				self.content_iter = iter(content_loader)
				self.art_iter = iter(art_reference_loader)
				content = next(self.content_iter).cuda()
				style = next(self.art_iter).cuda()
				content = size_arrange(content)
				style = size_arrange(style)
				

			###***********************###
			## Step 0 : Get Patterned value of given style images ##
			###***********************###
			#adaptive_alpha = self.adaptive_gram_weight(style, 3, 8).unsqueeze(1).cuda()
			#content_adaptive_alpha = ((self.adaptive_gram_weight(content, 1, 8)+self.adaptive_gram_weight(content, 2, 8)+self.adaptive_gram_weight(content, 3, 8) ) /3 ).unsqueeze(1).cuda()
			#style_adaptive_alpha = ((self.adaptive_gram_weight(style, 1, 8)+self.adaptive_gram_weight(style, 2, 8)+self.adaptive_gram_weight(style, 3, 8) ) /3 ).unsqueeze(1).cuda()


			#########################
			##Abalation 1 : Origin ##
			#########################
			#content_recon,_,_,_,_ = self.network(content, content, empty_segment, empty_segment)
			#style_recon,_,_,_,_ = self.network(style, style, empty_segment, empty_segment)
			

			#########################
			##Abalation 2 : Gray   ## <- This is the best
			#########################
			gray_content = torchvision.transforms.functional.rgb_to_grayscale(content).repeat(1,3,1,1)
			gray_style = torchvision.transforms.functional.rgb_to_grayscale(style).repeat(1,3,1,1)

			style_adaptive_alpha = (((self.adaptive_gram_weight(style, 1, 8)+self.adaptive_gram_weight(style, 2, 8)+self.adaptive_gram_weight(style, 3, 8) ) /3 ).unsqueeze(1).cuda() +\
			 ((self.adaptive_gram_weight(gray_style, 1, 8)+self.adaptive_gram_weight(gray_style, 2, 8)+self.adaptive_gram_weight(gray_style, 3, 8) ) /3 ).unsqueeze(1).cuda() )/2

			content_recon,_,_,_,_ = self.network(gray_content, content, torch.ones_like(style_adaptive_alpha), gray_content, content)
			style_recon,_,_,_,_ = self.network(gray_style, style, style_adaptive_alpha, gray_style, style)


			#############################
			##Abalation 3 : Gray+noise ##
			#############################
			#noise_gray_content = torchvision.transforms.functional.rgb_to_grayscale(content).repeat(1,3,1,1)+torch.randn_like(content)
			#noise_gray_style = torchvision.transforms.functional.rgb_to_grayscale(style).repeat(1,3,1,1)+torch.randn_like(style)

			#content_recon,_,_,_,_ = self.network(noise_gray_content, content, empty_segment, empty_segment)
			#style_recon,_,_,_,_ = self.network(noise_gray_style, style, empty_segment, empty_segment)

			#############################
			##Abalation 4 : HH         ##
			#############################
			#content_recon,_,_,_,_ = self.network(get_HH_LL(size_arrange(content))[0], content, empty_segment, empty_segment)
			#style_recon,_,_,_,_ = self.network(get_HH_LL(size_arrange(style))[0], style, empty_segment, empty_segment)

			#############################
			##Abalation 5 : Gray+HH    ##
			#############################
			#gray_content = torchvision.transforms.functional.rgb_to_grayscale(content).repeat(1,3,1,1)
			#gray_style = torchvision.transforms.functional.rgb_to_grayscale(style).repeat(1,3,1,1)
			#content_recon,_,_,_,_ = self.network(get_HH_LL(size_arrange(gray_content))[0], content, empty_segment, empty_segment)
			#style_recon,_,_,_,_ = self.network(get_HH_LL(size_arrange(gray_style))[0], style, empty_segment, empty_segment)

			
			stylization, attn_style_4_1, attn_style_5_1, attn_map_4_1, attn_map_5_1 = self.network(content, style, style_adaptive_alpha, gray_content, style)
			
			###*********************************###
			#####Step 2: Calculate  Loss  #########
			###*********************************###

			#########################	
			####Adversarial parts####
			#########################
			origin_gan_output = self.discriminator(style)
			style_gan_output = self.discriminator(stylization.detach())

			D_loss = self.bce_loss(origin_gan_output, ones_like(origin_gan_output)) + self.bce_loss(style_gan_output, zeros_like(style_gan_output))

			self.dis_optim.zero_grad()
			D_loss.backward()
			self.dis_optim.step()

			#####################################
			##### Style loss + Adversarial G ####
			#####################################

			#style_loss = 0
			content_loss = 0
			global_style_loss = 0
			local_style_loss = 0
			feature_recon_loss = []
			identity_loss1 = 0
			identity_loss2 = 0
			cx_loss = 0
			color_loss=0
			
			local_style_loss = self.proposed_local_gram_loss_v2(stylization, style, style_adaptive_alpha)
			
			###### Content & Style loss #####
			for level in [2,3,4,5]:
				stylized_feat = self.network.encoder.get_features(stylization, level)
				style_feat = self.network.encoder.get_features(style, level)
				content_feat = self.network.encoder.get_features(content, level)
				recon_style_feat = self.network.encoder.get_features(style_recon, level)
				recon_content_feat = self.network.encoder.get_features(content_recon, level)
				
				identity_loss2 += (self.calc_content_loss(recon_content_feat, content_feat) + self.calc_content_loss(recon_style_feat, style_feat))

				if level in [4,5]:
					#global_style_loss += self.calc_style_loss(stylized_feat, style_feat, dim=True)
					global_style_loss += self.calc_style_loss_centered_gram(stylized_feat, style_feat, dim=True)
					content_loss += self.calc_content_loss(stylized_feat, content_feat, norm=True)
					if level==4:
						cx_loss += ( (contextual_loss_v2(content_feat, recon_content_feat)) + (contextual_loss_v2(style_feat, recon_style_feat)) )
				del style_feat, content_feat, stylized_feat, recon_content_feat, recon_style_feat
				torch.cuda.empty_cache()
			
			
			attnded_style_loss = torch.mean(
								(self.MSE_instance_loss(gram_matrix(attn_style_4_1), gram_matrix(self.network.encoder.get_features(stylization, 4))) +\
								self.MSE_instance_loss(gram_matrix(attn_style_5_1), gram_matrix(self.network.encoder.get_features(stylization, 5))) ), dim=(1,2)
			)
			identity_loss1 = self.calc_content_loss(content_recon, content) + self.calc_content_loss(style_recon, style)
			tv_loss = TVloss(content_recon, self.tv_weight) + TVloss(style_recon, self.tv_weight)
			
			reg = self.styleaware_regularizer(stylization, style)
			#reg = self.contrastive_styleaware_regularizer(content, style, style_adaptive_alpha, gray_content, gray_style)
			
			color_loss = calc_histogram_loss(stylization, style, self.hist)

			#########################	
			####Adversarial parts####
			#########################
			style_gan_output_ = self.discriminator(stylization)

			G_loss = self.bce_loss(style_gan_output_, ones_like(style_gan_output_))


			total_loss = 1.0*cx_loss + 1.0*content_loss + 1.0*identity_loss2 + 0.2*local_style_loss + 100.0*torch.mean(attnded_style_loss*style_adaptive_alpha) + \
						 10.0*torch.mean(global_style_loss*(1-style_adaptive_alpha)) + 1.5*color_loss + tv_loss # + 0.1*G_loss #Final_v9_adv_t2_no_adv

			if torch.isnan(total_loss):
				del content_recon, style_recon, stylization
				del total_loss				
				torch.cuda.empty_cache()
				continue

			self.optim.zero_grad()
			total_loss.backward()
			self.optim.step()
			
			
			
			################################ 
			####      Checkpoints       ####
			################################
			wandb.log({
					"L/content_loss" : content_loss.item(),
					"L/attnded_style_loss" : torch.mean(attnded_style_loss*style_adaptive_alpha).item(),
					"L/global_style_loss" : torch.mean(global_style_loss*(1-style_adaptive_alpha)).item(),
					"L/identity_loss1" : identity_loss1.item(),
					"L/identity_loss2" : identity_loss2.item(),
					#"L/local_style_loss" : torch.mean(local_style_loss).item(),
					"L/local_style_loss" : local_style_loss.item(),
					"L/total_loss" : total_loss.item(),
					"L/cx_loss" : cx_loss.item(),
					"L/reg" : reg.item(),
				})

			if ((iteration) == 100) or ((iteration) % self.check_iter) == 0:
				print("%s: Iteration: [%d/%d]\tC_loss: %2.4f"%(time.ctime(), iteration, self.max_iter, total_loss.item()))
				wandb.log({"Recon_results": wandb.Image(denorm(tensor=torch.cat([content, content_recon, style, style_recon, stylization]), nrow=self.batch_size))})
				
			if (iteration) % 500 == 0:
				torch.save({'iteration': iteration,
					'state_dict': self.network.decoder.state_dict(),},
					os.path.join(self.result_st_dir, 'dec_model_'+str(iteration)+'.pth'))
				
				torch.save({'iteration': iteration,
					'state_dict': self.network.transformer.state_dict(),},
					os.path.join(self.result_st_dir, 'transformer_model_'+str(iteration)+'.pth'))

			del content_recon, style_recon, stylization#, r1_loss_sum
			del total_loss
			torch.cuda.empty_cache()


	def test(self, args):
		self.network.decoder.load_state_dict(torch.load(os.path.join(self.result_st_dir, 'dec_model_'+str(args.test_iter)+'.pth'))['state_dict'])
		self.network.transformer.load_state_dict(torch.load(os.path.join(self.result_st_dir, 'transformer_model_'+str(args.test_iter)+'.pth'))['state_dict'])
		
		content_set = Transfer_TestDataset(self.content_dir, (512, 512), self.cropsize, self.cencrop, type='art', is_test=True)
		art_reference_set = Transfer_TestDataset(self.style_dir, (512, 512), self.cropsize, self.cencrop, type='art', is_test=True)

		dir_path = os.path.join(args.test_result_dir, self.comment, 'stylized_results'+str(args.test_iter)+'_namhyuk')
		os.makedirs(dir_path, exist_ok=True)
		content_loader = torch.utils.data.DataLoader(content_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
		art_reference_loader = torch.utils.data.DataLoader(art_reference_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)

		N = content_set.__len__()
		self.content_iter = iter(content_loader)
		self.art_iter = iter(art_reference_loader)
		
		self.network.train(False)
		self.network.eval()
		
		
		import timeit
		import numpy as np
		inference_time = []
		
		for iteration in range(1,(N//self.batch_size)+1):
			empty_segment = np.asarray([])
			try:
				content = next(self.content_iter).cuda()
				style = next(self.art_iter).cuda()
			except:
				self.content_iter = iter(content_loader)
				self.art_iter = iter(art_reference_loader)
				content = next(self.content_iter).cuda()
				style = next(self.art_iter).cuda()
			
			content = size_arrange(content)
			style = size_arrange(style)
			
			gray_content = torchvision.transforms.functional.rgb_to_grayscale(content).repeat(1,3,1,1)
			gray_style = torchvision.transforms.functional.rgb_to_grayscale(style).repeat(1,3,1,1)

			start = timeit.default_timer()
			style_adaptive_alpha = (((self.adaptive_gram_weight(style, 1, 8)+self.adaptive_gram_weight(style, 2, 8)+self.adaptive_gram_weight(style, 3, 8) ) /3 ).unsqueeze(1).cuda() +\
			 ((self.adaptive_gram_weight(gray_style, 1, 8)+self.adaptive_gram_weight(gray_style, 2, 8)+self.adaptive_gram_weight(gray_style, 3, 8) ) /3 ).unsqueeze(1).cuda() )/2
			
			stylization, attn_style_4_1, attn_style_5_1, attn_map_4_1, attn_map_5_1 = self.network(content, style, style_adaptive_alpha, gray_content, style)
			

			end = timeit.default_timer()
			inference_time.append((end-start))
			imsave(stylization,  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_'+str(style_adaptive_alpha.item())+'.png'), nrow=self.batch_size )
			
			##You can test according to various alpha scale##

			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.0, gray_content, gray_style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_00.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.1, gray_content, gray_style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_01.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.2, gray_content, gray_style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_02.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.3, gray_content, style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_03.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.4, gray_content, style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_04.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.5, gray_content, style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_05.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.6, gray_content, style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_06.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.7, gray_content, style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_07.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.8, gray_content, style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_08.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*0.9, gray_content, style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_09.png'), nrow=self.batch_size )
			#imsave(self.network(content, style, torch.ones_like(style_adaptive_alpha)*1.0, gray_content, style)[0],  os.path.join(dir_path,  'single_art_stylized_'+str(iteration)+'_10.png'), nrow=self.batch_size )
			#imsave(content,  os.path.join(dir_path,  'content'+str(iteration)+'.png'), nrow=self.batch_size )
			#imsave(style,  os.path.join(dir_path,  'style'+str(iteration)+'.png'), nrow=self.batch_size )
			#del content, style, gray_content, gray_style
			torch.cuda.empty_cache()
		print('avg : ', np.mean(inference_time))
		




	def content_fidelity(self, args):
		import torchvision.transforms.functional as trans_F


		self.network.decoder.load_state_dict(torch.load(os.path.join(self.result_st_dir, 'dec_model_'+str(args.test_iter)+'.pth'))['state_dict'])
		self.network.transformer.load_state_dict(torch.load(os.path.join(self.result_st_dir, 'transformer_model_'+str(args.test_iter)+'.pth'))['state_dict'])
		content_set = Transfer_TestDataset(self.content_dir, (self.imsize, self.imsize), self.cropsize, self.cencrop, type='art', is_test=False)
		art_reference_set = Transfer_TestDataset(self.style_dir, (self.imsize, self.imsize), self.cropsize, self.cencrop, type='art', is_test=False)
		
		dir_path = os.path.join(args.test_result_dir, self.comment, 'stylized_results'+str(args.test_iter)+'_newnewnew')
		os.makedirs(dir_path, exist_ok=True)
		content_loader = torch.utils.data.DataLoader(content_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
		art_reference_loader = torch.utils.data.DataLoader(art_reference_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)

		N = content_set.__len__()
		content_iter = iter(content_loader)
		art_iter = iter(art_reference_loader)
		
		self.network.train(False)
		self.network.eval()
		
		cos = torch.nn.CosineSimilarity(eps=1e-6)
		total_CE = 0
		total_SL = 0
		total_style_loss = 0

		for iteration in range(1,(N//self.batch_size)+1):
			empty_segment = np.asarray([])
			try:
				content = next(self.content_iter).cuda()
				style = next(self.art_iter).cuda()
			except:
				self.content_iter = iter(content_loader)
				self.art_iter = iter(art_reference_loader)
				content = next(self.content_iter).cuda()
				style = next(self.art_iter).cuda()

			content = size_arrange(content)
			style = size_arrange(style)
			
			gray_style = torchvision.transforms.functional.rgb_to_grayscale(style).repeat(1,3,1,1)

			style_adaptive_alpha = (((self.adaptive_gram_weight(style, 1, 8)+self.adaptive_gram_weight(style, 2, 8)+self.adaptive_gram_weight(style, 3, 8) ) /3 ).unsqueeze(1).cuda() +\
			((self.adaptive_gram_weight(gray_style, 1, 8)+self.adaptive_gram_weight(gray_style, 2, 8)+self.adaptive_gram_weight(gray_style, 3, 8) ) /3 ).unsqueeze(1).cuda() )/2
			
			stylization = self.network(content, style, style_adaptive_alpha, content, style)[0]
			CE = 0
			for l in [2,3,4,5]:
			#for l in [4,5]: #Only layer 4th and 5th
				CE += cos(self.network.encoder.get_features(torch.nn.functional.interpolate(stylization, size=(256,256)), l), self.network.encoder.get_features(torch.nn.functional.interpolate(content, size=(256,256)), l)).abs().mean().item()
			CE = CE/4
			total_CE += CE
			
			
			cropped_style = trans_F.five_crop(torch.nn.functional.interpolate(style, size=(256,256)), 64)
			cropped_stylization = trans_F.five_crop(torch.nn.functional.interpolate(stylization, size=(256,256)), 64)
			total_gram_loss = []
			for k in range(5):
				gram_loss = []
				for level in [4,5]:
					reference_feat = gram_matrix(self.network.encoder.get_features(cropped_style[k], level))
					
					art_ours_output_feat = gram_matrix(self.network.encoder.get_features(cropped_stylization[k], level))
					gram_loss.append(self.MSE_loss(reference_feat, art_ours_output_feat).item())
					del reference_feat, art_ours_output_feat
					torch.cuda.empty_cache()
				total_gram_loss.append(np.mean(gram_loss))
				
			
			total_style_loss +=  np.mean(total_gram_loss)

			del stylization
			torch.cuda.empty_cache()

		total_CE = total_CE / N
		print("iteration  : " , str(args.test_iter))
		print("Content metric : ", total_CE)
		print("Style metric : " , (total_style_loss / len(temp_list)))


	def eval(self, args):
		self.network.decoder.load_state_dict(torch.load(os.path.join(self.result_st_dir, 'dec_model_'+str(args.test_iter)+'.pth'))['state_dict'])
		self.network.transformer.load_state_dict(torch.load(os.path.join(self.result_st_dir, 'transformer_model_'+str(args.test_iter)+'.pth'))['state_dict'])
		content_set = Transfer_TestDataset(self.content_dir, (self.imsize, self.imsize), self.cropsize, self.cencrop, type='art', is_test=True)
		art_reference_set = Transfer_TestDataset(self.style_dir, (self.imsize, self.imsize), self.cropsize, self.cencrop, type='art', is_test=True)
		content_loader = torch.utils.data.DataLoader(content_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
		art_reference_loader = torch.utils.data.DataLoader(art_reference_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)

		N = art_reference_set.__len__()
		content_iter = iter(content_loader)
		art_iter = iter(art_reference_loader)
		
		self.network.train(False)
		self.network.eval()

		gram_loss = []
		stylized_value = []
		mse_stylized_value = []

		
		for iteration in range(1,(N//self.batch_size)+1):
			empty_segment = np.asarray([])
			try:
				content = next(content_iter).cuda()
				style = next(art_iter).cuda()
			except:
				self.content_iter = iter(content_loader)
				self.art_iter = iter(art_reference_loader)
				content = next(self.content_iter).cuda()
				style = next(self.art_iter).cuda()
				content = size_arrange(content)
				style = size_arrange(style)
			
			gray_content = torchvision.transforms.functional.rgb_to_grayscale(content).repeat(1,3,1,1)
			gray_style = torchvision.transforms.functional.rgb_to_grayscale(style).repeat(1,3,1,1)

			
			style_adaptive_alpha = (((self.adaptive_gram_weight(style, 1, 8)+self.adaptive_gram_weight(style, 2, 8)+self.adaptive_gram_weight(style, 3, 8) ) /3 ).unsqueeze(1).cuda() +\
			 ((self.adaptive_gram_weight(gray_style, 1, 8)+self.adaptive_gram_weight(gray_style, 2, 8)+self.adaptive_gram_weight(gray_style, 3, 8) ) /3 ).unsqueeze(1).cuda() )/2
			
			stylized_results = self.network(content, style, style_adaptive_alpha, gray_content, style)[0]

			gray_stylized = torchvision.transforms.functional.rgb_to_grayscale(stylized_results).repeat(1,3,1,1)
			stylized_adaptive_alpha = (((self.adaptive_gram_weight(stylized_results, 1, 8)+self.adaptive_gram_weight(stylized_results, 2, 8)+self.adaptive_gram_weight(stylized_results, 3, 8) ) /3 ).unsqueeze(1).cuda() +\
			 ((self.adaptive_gram_weight(gray_stylized, 1, 8)+self.adaptive_gram_weight(gray_stylized, 2, 8)+self.adaptive_gram_weight(gray_stylized, 3, 8) ) /3 ).unsqueeze(1).cuda() )/2

			stylized_value.append(torch.nn.functional.l1_loss(stylized_adaptive_alpha, style_adaptive_alpha).item())
			mse_stylized_value.append(((stylized_adaptive_alpha-style_adaptive_alpha)**2).item())

			for level in [2,3,4,5]:
				reference_feat = gram_matrix(self.network.encoder.get_features(style, level))
				art_ours_output_feat = gram_matrix(self.network.encoder.get_features(stylized_results, level))

				gram_loss.append(self.MSE_loss(reference_feat, art_ours_output_feat).item())

				del reference_feat, art_ours_output_feat
				torch.cuda.empty_cache()

		print(" ")
		print(str(args.test_iter) + " GRAM Loss : " ,  np.mean(gram_loss), "  ", np.std(gram_loss))
		print(str(args.test_iter) + " stylized value : " ,  np.mean(stylized_value), "  ", np.std(stylized_value))
		print(str(args.test_iter) + " mse stylized value : " ,  np.mean(mse_stylized_value), "  ", np.std(mse_stylized_value))

	