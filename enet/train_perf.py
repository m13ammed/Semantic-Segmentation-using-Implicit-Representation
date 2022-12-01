from email.mime import image
import os, sys
from turtle import pos

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import time
from dataloader.generate_groundtruth import generate_groundtruth_render_batch_in
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

	def __init__(self, model, data_loader, optim, criterion, metric, device):
		self.model = model
		self.data_loader = data_loader
		self.optim = optim
		self.criterion = criterion
		self.metric = metric
		self.device = device

	def run_epoch(self, iteration_loss=0):
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
		for step, batch_data in enumerate(self.data_loader):
			startTime = time.time()
			# Get the inputs and labels
			if len(batch_data)==6:
				inputs, intrinsic, poses, mesh, labels, color = batch_data
				color = color.to(self.device)
			else:
				inputs, intrinsic, poses, mesh, labels = batch_data

			
			labels = labels.to(self.device)
			#intrinsic = intrinsic.to(self.device)
			#poses = poses.to(self.device)
			mesh = mesh.to(self.device)
			labels, _ = generate_groundtruth_render_batch_in(image_out_size=[inputs.shape[-2], inputs.shape[-1]], mesh=mesh,
       				intrinsics = intrinsic, labels=labels, poses=poses, device=self.device)
			intrinsic = intrinsic.cpu()
			poses = poses.cpu()
			mesh = mesh.cpu()
			labels = labels.long()
			inputs = inputs.to(self.device)
			# Forward propagation
			outputs = self.model(inputs)
			# Loss computation
			loss = self.criterion(outputs, labels)

			# Backpropagation
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Keep track of loss for current epoch
			epoch_loss += loss.item()

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

		return epoch_loss / len(self.data_loader), self.metric.value()
