import argparse
import torch.utils.data as data
from utils.model_dataset import Perfception as dataset
from PIL import Image
from utils.scannet_constants import *
import numpy as np
import torch

verbose = False
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else: device = torch.device("cpu")

def enet_weighing(dataloader, num_classes, c=1.02):
	"""Computes class weights as described in the ENet paper:

		w_class = 1 / (ln(c + p_class)),

	where c is usually 1.02 and p_class is the propensity score of that
	class:

		propensity_score = freq_class / total_pixels.

	References: https://arxiv.org/abs/1606.02147

	Keyword arguments:
	- dataloader (``data.Dataloader``): A data loader to iterate over the
	dataset.
	- num_classes (``int``): The number of classes.
	- c (``int``, optional): AN additional hyper-parameter which restricts
	the interval of values for the weights. Default: 1.02.

	"""
	class_count = 0
	total = 0
	for _, label in dataloader:
		label = label.cpu().numpy()

		# Flatten label
		flat_label = label.flatten()

		# Sum up the number of pixels of each class and the total pixel
		# counts for each label
		class_count += np.bincount(flat_label, minlength=num_classes)
		total += flat_label.size

	# Compute propensity score and then the weights for each class
	propensity_score = class_count / total
	class_weights = 1 / (np.log(c + propensity_score))

	return class_weights

def load_dataset(dataset):
    if(verbose):
        print("Loading Dataset")
    
    train_set = dataset(mode='train', seg_classes=args.seg_classes, frame_skip=args.frame_skip)
    train_loader = data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True, num_workers=args.workers)

    val_set = dataset(mode='val', seg_classes=args.seg_classes, frame_skip=args.frame_skip)
    val_loader = data.DataLoader(val_set,batch_size=args.batch_size,shuffle=True, num_workers=args.workers)

    test_set = dataset(mode='test', seg_classes=args.seg_classes, frame_skip=args.frame_skip)
    test_loader = data.DataLoader(test_set,batch_size=args.batch_size,shuffle=True, num_workers=args.workers)

    class_encoding = SCANNET_COLOR_MAP_20
    if(args.seg_classes=="SCANNET_200"):
        class_encoding = SCANNET_COLOR_MAP_200
    
    num_classes = len(class_encoding)

    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    if args.mode.lower() == 'test':
        images, labels = next(iter(test_loader))
    else:
        images, labels = next(iter(train_loader))
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    class_weights = enet_weighing(train_loader, num_classes)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index(0)
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader,test_loader), class_weights, class_encoding

if __name__ =="__main__":
    parser = argparse.ArgumentParser(
        prog="ENet Runner for ScanNet and Perfception",
        description="Takes as input the perfception renders and the corresponding ground truth for semantic segmentation"
    )

    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--frame_skip', default=20)
    parser.add_argument('-bs', '--batch_size', default=20) 
    parser.add_argument('-w','--workers', default=1)
    parser.add_argument('-seg','--seg_classes', default="SCANNET_20")
    parser.add_argument('-m','--mode',default="train")
    parser.add_argument('-iul','--ignore_unlabeled',default=True)
    args = parser.parse_args()
    verbose = args.verbose
    frame_skip = args.frame_skip
    batch_size = args.batch_size

    ### Add dataloader part here

    loaders, w_class, class_encoding = load_dataset(dataset)    

