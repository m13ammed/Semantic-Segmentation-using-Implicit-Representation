import argparse
import torch.utils.data as data
from enet.metric.iou import IoU
from enet.model import ENet
from enet.model_dataset import LitPerfception as dataset
from enet.model_dataset import custom_collate
from PIL import Image
from enet.test import Test
from enet.train_perf import Train
from configs.scannet_constants import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from enet.enet_utils import load_checkpoint, save_checkpoint

args = None
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
    import gin
    #gin.parse_config_files_and_bindings(['configs/semantic_perf.gin'],[''])
    train_set = dataset(mode='train')
    train_loader = data.DataLoader(train_set,batch_size=args.batch_size,shuffle=False, num_workers=args.workers, collate_fn=custom_collate)

    val_set = dataset(mode='val')
    val_loader = data.DataLoader(val_set,batch_size=args.batch_size,shuffle=False, num_workers=args.workers, collate_fn=custom_collate)

    test_set = dataset(mode='test')
    test_loader = data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False, num_workers=args.workers, collate_fn=custom_collate)

    class_encoding = SCANNET_COLOR_MAP_20
    if(args.seg_classes=="SCANNET_200"):
        class_encoding = SCANNET_COLOR_MAP_200
    
    num_classes = len(class_encoding)

    class_encoding = SCANNET_COLOR_MAP_20_
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    #if args.mode.lower() == 'test':
    #    images, labels = next(iter(test_loader))
    #else:
    #    images, labels = next(iter(train_loader))
    #print("Image size:", images.size())
    #print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    class_weights = None
    #class_weights = enet_weighing(train_loader, num_classes)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        ignore_index = list(class_encoding).index(0)
        class_weights[ignore_index] = 0

    #print("Class weights:", class_weights)

    return (train_loader, val_loader,test_loader), class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding):
    
    print("\nTraining...\n")
    num_classes = len(class_encoding)
    model = ENet(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
		model.parameters(),
		lr=args.learning_rate, betas=(args.beta0, args.beta1),
		weight_decay=args.weight_decay)

    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
									 args.lr_decay)
    ignore_index = list(class_encoding).index(0)
    metric = IoU(num_classes, ignore_index=ignore_index)
    if args.resume:
        model, optimizer, start_epoch, best_miou = load_checkpoint(model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
    train = Train(model, train_loader, optimizer, criterion, metric, device)
    val = Test(model, val_loader, criterion, metric, device)
    best_miou = 0
    for epoch in range(start_epoch, args.epochs):
        save_checkpoint(model, optimizer, epoch + 1, best_miou,
									  args)
        print(">>>> [Epoch: {0:d}] Training".format(epoch))
        lr_updater.step()
        epoch_loss, (iou, miou) = train.run_epoch(args.print_step)

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
			  format(epoch, epoch_loss, miou))
        
        if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(args.print_step)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
				  format(epoch, loss, miou))

			# Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
            	for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

			# Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                save_checkpoint(model, optimizer, epoch + 1, best_miou,
									  args)

    return model

def predict(model, images, class_encoding):
	images = images.to(device)

	# Make predictions!
	model.eval()
	with torch.no_grad():
		predictions = model(images)

	# Predictions is one-hot encoded with "num_classes" channels.
	# Convert it to a single int using the indices where the maximum (1) occurs
	_, predictions = torch.max(predictions.data, 1)

	# label_to_rgb = transforms.Compose([
	# 	ext_transforms.LongTensorToRGBPIL(class_encoding),
	# 	transforms.ToTensor()
	# ])
	# color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
	# utils.imshow_batch(images.data.cpu(), color_predictions)

def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

	# We are going to use the CrossEntropyLoss loss function as it's most
	# frequentely used in classification problems with multiple classes which
	# fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

	# Evaluation metric

    ignore_index = list(class_encoding).index(0)
    metric = IoU(num_classes, ignore_index=ignore_index)

	# Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

	# Show a batch of samples and labels
    #if args.imshow_batch:
        #print("A batch of predictions from the test set...")
    #images, _ = next(iter(test_loader))
    #predict(model, images, class_encoding)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(
        prog="ENet Runner for ScanNet and Perfception",
        description="Takes as input the perfception renders and the corresponding ground truth for semantic segmentation"
    )

    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--frame_skip', default=0)
    parser.add_argument('-b', '--batch_size', default=4) 
    parser.add_argument('-w','--workers', default=1)
    parser.add_argument('-seg','--seg_classes', default="SCANNET_20")
    parser.add_argument('-m','--mode',default="test")
    parser.add_argument('-iul','--ignore_unlabeled',default=True)
    parser.add_argument('-res','--resume', default=False)
    parser.add_argument('-lr','--learning_rate', default=5e-4)
    parser.add_argument('-ep','--epochs', default=200)
    parser.add_argument('-print','--print_step', default=25)
    parser.add_argument('-b0','--beta0', default=0.9)
    parser.add_argument('-b1','--beta1', default=0.999)
    parser.add_argument('-wd','--weight_decay', default=2e-4)
    parser.add_argument('-lrde','--lr_decay_epochs', default=100)
    parser.add_argument('-lrd','--lr_decay', default=0.5)
    parser.add_argument('-val_every','--validate_every', default=1000)


    args = parser.parse_args()
    verbose = args.verbose
    frame_skip = args.frame_skip
    batch_size = args.batch_size

    ### Add dataloader part here
    
    loaders, w_class, class_encoding = load_dataset(dataset)  
    train_loader, val_loader, test_loader = loaders
    if args.mode.lower() in {'train', 'full'}:
        #print("Weight classes", w_class.shape)
        model = train(train_loader, val_loader, w_class, class_encoding)
        if args.mode.lower() == 'full':
            test(model, test_loader, w_class, class_encoding)
    elif args.mode.lower() == 'test':
        num_classes = len(class_encoding)
        model = ENet(num_classes).to(device)
        optimizer = optim.Adam(model.parameters())
        model = load_checkpoint(model, optimizer, './',
									  '_118.ckpt')[0]
        print(model)
        test(model, test_loader, w_class, class_encoding)
