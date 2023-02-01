from email.mime import image
import os, sys
from turtle import begin_fill, pos

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import time
from dataloader.generate_groundtruth import generate_groundtruth_render_batch_in
import torch
import numpy as np
class Train:
	"""Performs the training of ``model`` given a training dataset data
	loader, the optimizer, and the loss criterion.

	Keyword arguments:
	- model (``nn.Module``): the model instance to train.
	- data_loader (``Dataloader``): Provides single or multi-process
	iterators over the dataset.
	- optim (``Optimizer``): The optimization algorithm.
	- criterion (``Optimizer``): The loss criterion.
	- metric (```Metric``): An instance specifying the metric to return.
	- device (``torch.device``): An object representing the device on which
	tensors are allocated.

	"""

	def __init__(self, model, data_loader, optim, criterion, metric, device, writer=None, class_encoding=None):
		self.model = model
		self.data_loader = data_loader
		self.optim = optim
		self.criterion = criterion
		self.metric = metric
		self.device = device
		self.writer = writer
		self.class_encoding = class_encoding
		self.scaler = None
	def run_epoch(self, iteration_loss=0, epoch_num=0):
		"""Runs an epoch of training.

		Keyword arguments:
		- iteration_loss (``bool``, optional): Prints loss at every step.

		Returns:
		- The epoch loss (float).

		"""
		self.model.train()
		epoch_loss = 0.0
		self.metric.reset()
		avgTime = 0.0
		numTimeSteps = 0
		first_time = time.time()
		for step, batch_data in enumerate(self.data_loader):
			startTime = time.time()
			if self.scaler is None:
				self.enabled = self.data_loader.dataset.use_sh
				self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)
			# Get the inputs and labels

			inputs, labels = self.preppare_input(batch_data)
			# Forward propagation
			with torch.autocast(device_type='cuda', dtype=torch.float16, enabled= self.enabled):
				outputs = self.model(inputs)
	
				loss = self.criterion(outputs, labels)

			# Backpropagation
			self.optim.zero_grad()
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optim)
			self.scaler.update()


			# Keep track of loss for current epoch
			losss_item = loss.item()
			epoch_loss += losss_item

			# Keep track of the evaluation metric
			self.metric.add(outputs.detach(), labels.detach())
			endTime = time.time()
			avgTime += (endTime - startTime)
			numTimeSteps += 1

			if iteration_loss > 0 and (step % iteration_loss == 0):
				print("[Step: %d/%d (%3.2f ms)] Iteration loss: %.4f" % (step, len(self.data_loader), \
					1000*(avgTime / (numTimeSteps if numTimeSteps>0 else 1)), loss.item()))
				numTimeSteps = 0
				avgTime = 0.

			self.writer.add_scalar('loss_step/training', losss_item, (epoch_num)*len(self.data_loader) + step ) 
			self.writer.add_scalar('time_step/training', ((endTime - startTime)*1000), (epoch_num)*len(self.data_loader) + step)

		iou, miou = self.metric.value()
		avg_loss = epoch_loss / len(self.data_loader)
		self.writer.add_scalar('loss_epoch/training',  avg_loss, (epoch_num+1))
		self.writer.add_scalar('time_epoch/training', ((endTime - first_time)*1000), (epoch_num+1))
		self.writer.add_scalar('miou_epoch/training', miou, (epoch_num+1))

		
		for key, iou_ in zip(self.class_encoding, np.nan_to_num(iou,nan=-1)):
			self.writer.add_scalar('training_epoch_iou/'+key, float(iou_), (epoch_num+1))
		
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

