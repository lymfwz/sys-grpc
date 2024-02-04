# encoding=gbk
import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import My_Net,GaborNet,U_Net
from Enetwork import EU_Net
import csv
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.optim import Adam
from PIL import Image, ImageFilter
import copy

def dice_loss(pred, target):
    dice = 0
    num = pred.size(0)
    # print(pred.size(),target.size())
    for i in range(num):
      inse = torch.sum(pred[i] * target[i])
      l = torch.sum(pred[i] * pred[i])
      r = torch.sum(target[i] * target[i])
      dice = dice + 2 * inse / (l + r)
    return 1 - dice/num

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		# self.criterion = torch.nn.MSELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		model_config = [[2, 3, 2, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 1, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 256, 1, 0.25]]
		"""Build generator and discriminator."""
		if self.model_type =='My_Net':
			self.unet = My_Net(img_ch=3,output_ch=1)
		elif self.model_type =='GaborNet':
			self.unet = GaborNet(img_ch=3,output_ch=1)
		elif self.model_type =='U_Net':
			self.unet = U_Net(img_ch=3,output_ch=1)
		elif self.model_type =='EU_Net':
			self.unet = EU_Net(img_ch=3,output_ch=1,model_cnf=model_config)
		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img





	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
        # U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			Loss_list = []
			for epoch in range(self.num_epochs):
				self.unet.train(True)
				epoch_loss = 0

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT) in enumerate(self.train_loader):
					# GT : Ground Truth
					images = images.to(self.device)
					GT = GT.to(self.device)

					# SR : Segmentation Result
					#print(next(self.unet.parameters()).device)
					SR = self.unet(images).to(self.device)
					# ==================原来start=====================
					SR_probs = F.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)
					# print(GT.size())
					# print(SR_probs.size())
					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					epoch_loss += loss.item()
					# ==================原来end=====================
					# ==================修改损失=====================
					# SR_probs = F.sigmoid(SR)
					# loss = dice_loss(SR_probs,GT)
					# epoch_loss += loss.item()
					# ==================修改损失=====================
					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					one = torch.ones_like(SR_probs)
					zero = torch.zeros_like(SR_probs)
					SR_probs = torch.where(SR > 0.5, one, SR_probs)
					SR_probs = torch.where(SR < 0.5, zero, SR_probs)
					acc += get_accuracy(SR_probs,GT)
					SE += get_sensitivity(SR_probs,GT)
					SP += get_specificity(SR_probs,GT)
					PC += get_precision(SR_probs,GT)
					F1 += get_F1(SR_probs,GT)
					JS += get_JS(SR_probs,GT)
					DC += get_DC(SR_probs,GT)
					length += 1
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				epoch_loss = epoch_loss / length

				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc,SE,SP,PC,F1,JS,DC))
				Loss_list.append(epoch_loss)
				x = range(1, self.num_epochs+1)
				y = Loss_list



			

				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))

				
				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0
				for i, (images, GT) in enumerate(self.valid_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = F.sigmoid(self.unet(images))
					one = torch.ones_like(SR)
					zero = torch.zeros_like(SR)
					SR = torch.where(SR > 0.5, one, SR)
					SR = torch.where(SR < 0.5, zero, SR)
					# SR = cv2.threshold(SR.astype('uint8'), 0.1, 1, cv2.THRESH_BINARY)
					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
						
					length += 1
					
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				unet_score = JS + DC
				# unet_score = acc

				print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))

				# torch.save(self.unet.state_dict(), unet_path)

				#torchvision.utils.save_image(images.data.cpu(),
				#							os.path.join(self.result_path,
				#										'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				#torchvision.utils.save_image(SR.data.cpu(),
				#							os.path.join(self.result_path,
				#										'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				#torchvision.utils.save_image(GT.data.cpu(),
				#							os.path.join(self.result_path,
				#										'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))

				print('The unet_score score is: %.4f' % (unet_score))
				# Save Best U-Net model
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
					torch.save(best_unet,unet_path)

			#===================================== Test ====================================#
			del self.unet
			del best_unet
			self.build_model()
			self.unet.load_state_dict(torch.load(unet_path))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			for i, (images, GT) in enumerate(self.test_loader):

				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = F.sigmoid(self.unet(images))
				one = torch.ones_like(SR)
				zero = torch.zeros_like(SR)
				SR = torch.where(SR > 0.5, one, SR)
				SR = torch.where(SR < 0.5, zero, SR)
				# SR = cv2.threshold(SR.astype('uint8'), 0.1, 1, cv2.THRESH_BINARY)
				acc += get_accuracy(SR,GT)
				SE += get_sensitivity(SR,GT)
				SP += get_specificity(SR,GT)
				PC += get_precision(SR,GT)
				F1 += get_F1(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT)

				torchvision.utils.save_image(images.data.cpu(),
											 os.path.join(self.result_path,
														  '%s_test_%d_image.png' % (self.model_type, length + 1)))
				torchvision.utils.save_image(SR.data.cpu(),
											 os.path.join(self.result_path,
														  '%s_test_%d_SR.png' % (self.model_type, length + 1)))
				torchvision.utils.save_image(GT.data.cpu(),
											 os.path.join(self.result_path,
														  '%s_test_%d_GT.png' % (self.model_type, length + 1)))

				length += 1
					
			acc = acc/length
			#acc = acc.numpy()
			SE = SE/length
			SE = SE.numpy()
			SP = SP/length
			SP = SP.numpy()
			PC = PC/length
			PC = PC.numpy()
			F1 = F1/length
			F1 = F1.numpy()
			JS = JS/length
			DC = DC/length
			# unet_score = JS + DC


			f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
			f.close()
			plt.plot(x, y, '-')
			plt.xlabel('Train loss vs. epoches')
			plt.ylabel('Train loss')
			plt.savefig(os.path.join(self.model_path, "%s-%d-%.4f-%d-%.4f-loss.jpg" %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob)))
			plt.show()

	def test(self):
		unet_path = os.path.join(self.model_path, 'GaborNet-300-0.0010-30-0.6577.pkl' )
		# del self.unet
		# del best_unet
		self.build_model()
		self.unet.load_state_dict(torch.load(unet_path))

		self.unet.train(False)
		self.unet.eval()

		acc = 0.  # Accuracy
		SE = 0.  # Sensitivity (Recall)
		SP = 0.  # Specificity
		PC = 0.  # Precision
		F1 = 0.  # F1 Score
		JS = 0.  # Jaccard Similarity
		DC = 0.  # Dice Coefficient
		length = 0
		i = 0
		for i, (images, GT) in enumerate(self.test_loader):
			images = images.to(self.device)
			GT = GT.to(self.device)
			SR = F.sigmoid(self.unet(images))
			one = torch.ones_like(SR)
			zero = torch.zeros_like(SR)
			SR = torch.where(SR > 0.5, one, SR)
			SR = torch.where(SR < 0.5, zero, SR)
			# SR = cv2.threshold(SR.astype('uint8'), 0.1, 1, cv2.THRESH_BINARY)
			acc += get_accuracy(SR, GT)
			SE += get_sensitivity(SR, GT)
			SP += get_specificity(SR, GT)
			PC += get_precision(SR, GT)
			F1 += get_F1(SR, GT)
			JS += get_JS(SR, GT)
			DC += get_DC(SR, GT)
			# images = images*0.5 +0.5

			torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_256test_%d_image.png'%(self.model_type,length+1)))
			torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_256test_%d_SR.png'%(self.model_type,length+1)))
			torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_256test_%d_GT.png'%(self.model_type,length+1)))

			length += 1

			# visualization------------------------------------------------------------------------------------------
			SR2 = self.unet(images)[1]
			#print(SR2.shape)
			for i in range(SR2.shape[1]):
				feature = SR2[:, i, :, :]
				# print(feature.shape)
				feature = feature.view(feature.shape[1], feature.shape[2])
				# print(feature.shape)
				feature = feature.cpu().detach().numpy()
				#feature = 1.0 / (1 + np.exp(-1 * feature))
				feature = np.round(feature * 255)
				# print(feature[0])
				# feature_img = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
				tmp_file = os.path.join(self.model_path, str(length) + '_' +str(i+1) + '_' + str(256) + '.png')
				# tmp_img = feature_img.copy()
				# tmp_img = cv2.resize(tmp_img, (256, 256), interpolation=cv2.INTER_NEAREST)
				cv2.imwrite(tmp_file, feature)


		acc = acc / length
		# acc = acc.numpy()
		SE = SE / length
		SE = SE.numpy()
		SP = SP / length
		SP = SP.numpy()
		PC = PC / length
		PC = PC.numpy()
		F1 = F1 / length
		F1 = F1.numpy()
		JS = JS / length
		DC = DC / length
		print('[Test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (acc, SE, SP, PC, F1, JS, DC))

		f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow(
			[self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, self.num_epochs, self.num_epochs_decay,
			 self.augmentation_prob])
		f.close()
