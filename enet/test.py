import os, sys
from tkinter.messagebox import NO
from tokenize import Ignore
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import time
from dataloader.generate_groundtruth import generate_groundtruth_render_batch_in
import numpy as np
from PIL import Image
from configs.scannet_constants import *
import torchvision.transforms as transforms

class Test:
	"""Tests the ``model`` on the specified test dataset using the
	data loader, and loss criterion.

	Keyword arguments:
	- model (``nn.Module``): the model instance to test.
	- data_loader (``Dataloader``): Provides single or multi-process
	iterators over the dataset.
	- criterion (``Optimizer``): The loss criterion.
	- metric (```Metric``): An instance specifying the metric to return.
	- device (``torch.device``): An object representing the device on which
	tensors are allocated.

	"""

	def __init__(self, model, data_loader, criterion, metric, device, writer=None, class_encoding=None, log_image_every=1e6):
		self.model = model
		self.data_loader = data_loader
		self.criterion = criterion
		self.metric = metric
		self.device = device
		self.writer = writer
		self.class_encoding = class_encoding
		self.log_image_every = log_image_every
		self.first = True
	def run_epoch(self, iteration_loss=False, epoch_num = 0):
		"""Runs an epoch of validation.

		Keyword arguments:
		- iteration_loss (``bool``, optional): Prints loss at every step.

		Returns:
		- The epoch loss (float), and the values of the specified metrics

		"""
		self.model.eval()
		epoch_loss = 0.0
		self.metric.reset()
		avgTime = 0.0
		numTimeSteps = 0
		first_time = time.time()
		denorm = self.data_loader.dataset.color_mean, self.data_loader.dataset.color_std
		color_mean = None
		color_std = None
		for step, batch_data in enumerate(self.data_loader):
			startTime = time.time()
			inputs, labels = self.preppare_input(batch_data)

			with torch.no_grad():
				# Forward propagation
				outputs = self.model(inputs)

				# Loss computation
				loss = self.criterion(outputs, labels)
   
			# Keep track of loss for current epoch
			losss_item = loss.item()
			epoch_loss += losss_item

			# Keep track of evaluation the metric
			self.metric.add(outputs.detach(), labels.detach())
			endTime = time.time()
			avgTime += (endTime - startTime)
			numTimeSteps += 1

			if iteration_loss > 0 and (step % iteration_loss == 0):
				print("[Step: %d/%d (%3.2f ms)] Iteration loss: %.4f" % (step, len(self.data_loader), \
					1000*(avgTime / (numTimeSteps if numTimeSteps>0 else 1)), loss.item()))
				numTimeSteps = 0
				avgTime = 0.
    
			self.writer.add_scalar('loss_step/val', losss_item, (epoch_num)*len(self.data_loader) + step ) 
			self.writer.add_scalar('time_step/val', ((endTime - startTime)*1000), (epoch_num)*len(self.data_loader) + step)

			if (step%self.log_image_every) == 0 :
				_, predictions = torch.max(outputs.data, 1)
				gt_img = SCANNET_COLOR_MAP_20_array[labels.detach().cpu()].astype('uint8')
				pred_img = SCANNET_COLOR_MAP_20_array[predictions.detach().cpu()].astype('uint8')
				self.writer.add_image(f'step_{step}_'+'pred', pred_img, epoch_num , dataformats='NHWC')
				if self.first:
					if color_mean is None:
						color_mean = np.tile(np.array(self.data_loader.dataset.color_mean).reshape(1,3,1,1), (inputs.shape[0],1,inputs.shape[-2], inputs.shape[-1]))
						color_std = np.tile(np.array(self.data_loader.dataset.color_std).reshape(1,3,1,1), (inputs.shape[0],1,inputs.shape[-2], inputs.shape[-1]))
					slicer = inputs.shape[0]
					rgb = (inputs.detach().cpu().numpy()[:,:3,...	]*color_std[:slicer] + color_mean[:slicer])*255
					rgb = rgb.astype('uint8')
					self.writer.add_image(f'step_{step}_'+'gt', gt_img, epoch_num , dataformats='NHWC')
					self.writer.add_image(f'step_{step}_'+'rgb', rgb, epoch_num , dataformats='NCHW')
					

		iou, miou = self.metric.value()
		avg_loss = epoch_loss / len(self.data_loader)
		self.writer.add_scalar('loss_epoch/val',  avg_loss, (epoch_num+1))
		self.writer.add_scalar('time_epoch/val', ((endTime - first_time)*1000), (epoch_num+1))
		self.writer.add_scalar('miou_epoch/val', miou, (epoch_num+1))
		if self.first:
			self.first = False
		
		for key, iou_ in zip(self.class_encoding, np.nan_to_num(iou,nan=-1)):
			self.writer.add_scalar('val_epoch_iou/'+key, float(iou_), (epoch_num+1))

		return avg_loss, (iou, miou)
	def preppare_input(self, batch):
		if self.data_loader.dataset.use_sh:
			inputs =  torch.cat((batch['rgb'], batch['sh']), 1)
		else:
			inputs = batch['rgb']
		if not self.data_loader.dataset.allow_gen_lables:
			return inputs.to(self.device), batch['labels_2d'].long().to(self.device)
		else:
			intrinsic, poses= batch['intrinsic'], batch['poses']
			mesh, labels, idx_to_ren = batch['mesh'], batch['labels'], batch['idx_to_ren']
			labels_2d = batch['labels_2d'].to(self.device)
			labelss, _ = generate_groundtruth_render_batch_in(image_out_size=[inputs.shape[-2], inputs.shape[-1]], mesh=mesh.to(self.device),
       				intrinsics = intrinsic, labels=labels.to(self.device), poses=poses, device=self.device)
			labels_2d[idx_to_ren] = labelss
			return inputs.to(self.device), labels_2d.long()
