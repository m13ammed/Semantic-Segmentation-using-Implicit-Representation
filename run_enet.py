import argparse
from distutils.log import debug
from unittest import result
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
from torch.utils.tensorboard import SummaryWriter
import gin
import os
args = None
hparams = None
verbose = False
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else: device = torch.device("cpu")
from dataloader.generate_groundtruth import generate_groundtruth_render_batch_in

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
    if args.cls_weight_pth is None:
        class_count = 0
        total = 0
        for ret in dataloader:
            if not dataloader.dataset.allow_gen_lables:
                label =  ret['labels_2d'].cpu().numpy().astype('int64')
            else:
                intrinsic, poses= ret['intrinsic'], ret['poses']
                mesh, labels, idx_to_ren = ret['mesh'], ret['labels'], ret['idx_to_ren']
                labels_2d = ret['labels_2d'].to(device)
                labelss, _ = generate_groundtruth_render_batch_in(image_out_size=[ret['rgb'].shape[-2], ret['rgb'].shape[-1]], mesh=mesh.to(device),
                        intrinsics = intrinsic, labels=labels.to(device), poses=poses, device=device)
                labels_2d[idx_to_ren] = labelss
                label = labels_2d.cpu().numpy().astype('int64')

            #label = ret['labels_2d'].cpu().numpy().astype('int64')
            # Flatten label
            flat_label = label.flatten()
            # Sum up the number of pixels of each class and the total pixel
            # counts for each label
            class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size
        propensity_score = class_count / total
    else:
        count = np.array(list(np.load(args.cls_weight_pth, allow_pickle=True).tolist().values()))
        # Compute propensity score and then the weights for each class
        propensity_score = count / count.sum()
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights

def load_dataset(dataset):
    if(verbose):
        print("Loading Dataset")
    import gin
    #gin.parse_config_files_and_bindings(['configs/semantic_perf.gin'],[''])
    train_set = dataset(mode='train', use_sh = args.use_sh, allow_gen_lables = args.allow_gen_lables, use_original_norm=args.use_original_norm, opt = args.opt)
    train_loader = data.DataLoader(train_set,batch_size=args.batch_size,shuffle=False, num_workers=args.workers, collate_fn=custom_collate)

    val_set = dataset(mode='val', use_sh = args.use_sh, allow_gen_lables = False, use_original_norm=args.use_original_norm, opt = args.opt)
    val_loader = data.DataLoader(val_set,batch_size=args.batch_size,shuffle=False, num_workers=args.workers, collate_fn=custom_collate)

    test_set = dataset(mode='test', use_sh = args.use_sh, allow_gen_lables = False, use_original_norm=args.use_original_norm)
    test_loader = data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False, num_workers=args.workers, collate_fn=custom_collate)


    class_encoding = SCANNET_COLOR_MAP_20_
    num_classes = len(class_encoding)
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    print("Class-color encoding:", class_encoding)

    class_weights = None
    class_weights = enet_weighing(train_loader, num_classes)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        ignore_index = list(class_encoding).index(0)
        class_weights[ignore_index] = 0

    #print("Class weights:", class_weights)

    return (train_loader, val_loader,test_loader), class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding, writer):
    
    print("\nTraining...\n")
    num_classes = len(class_encoding)
    model = ENet(num_classes, use_sh = args.use_sh).to(device)
    ignore_index = list(class_encoding).index(0)
    criterion = nn.CrossEntropyLoss(weight=class_weights,ignore_index=ignore_index)
    optimizer = optim.Adam(
		model.parameters(),
		lr=args.learning_rate, betas=(args.beta0, args.beta1),
		weight_decay=args.weight_decay)

    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
									 args.lr_decay)
    
    metric = IoU(num_classes, ignore_index=ignore_index)
    if args.resume:
        model, optimizer, start_epoch, best_miou = load_checkpoint(model, optimizer, args.save_dir, args.checkpoint)
        print("Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
    train = Train(model, train_loader, optimizer, criterion, metric, device, writer, class_encoding=CLASS_LABELS_20_)
    val = Test(model, val_loader, criterion, metric, device, writer, class_encoding=CLASS_LABELS_20_, log_image_every=args.log_image_every)
    best_miou = 0
    for epoch in range(start_epoch, args.epochs):
        save_checkpoint(model, optimizer, epoch + 1, best_miou,
									  args)
        print(">>>> [Epoch: {0:d}] Training".format(epoch))
        lr_updater.step()
        epoch_loss, (iou, miou) = train.run_epoch(args.print_step, epoch)

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
			  format(epoch, epoch_loss, miou))
        
        if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(args.print_step, epoch)

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
                best_iou = iou
                save_checkpoint(model, optimizer, epoch + 1, best_miou,
									  args)

    results = {"best_moiu": best_miou}
    for key, iou in zip(CLASS_LABELS_20_, np.nan_to_num(best_iou,nan=-1)):
        
        results.update({"iou_at_best_moiu/"+key:float(iou)})
    writer2 = SummaryWriter(log_dir=args.logs_dir)
    writer2.add_hparams(hparams, results, run_name = args.exp_name)
    
    
    #SCANNET_COLOR_MAP_20_array
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
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)

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
    
@gin.configurable()
def hparams_setup( batch_size=4, learning_rate=5e-3, epochs=200, beta0=0.9, beta1=0.999, weight_decay=2e-4,
                  lr_decay_epochs=100, lr_decay=0.5, validate_every=1, use_sh = False, allow_gen_lables = False,
                  use_original_norm=False, opt = False):
    params  = {
        "batch_size":batch_size,
        "learning_rate":learning_rate,
        'epochs':epochs,
        'beta0':beta0,
        'beta1':beta1,
        'weight_decay':weight_decay,
        'lr_decay_epochs':lr_decay_epochs,
        'lr_decay':lr_decay,
        'validate_every':validate_every,
        'use_sh':use_sh,
        'allow_gen_lables':allow_gen_lables,
        'use_original_norm': use_original_norm,
        'opt': opt
    }
    import random
    for key in params.keys():
        if isinstance(params[key], list):
            params[key] = random.choice(params[key])
    return params

@gin.configurable() 
def additional_setup (logs_dir='./',exp_name='', workers=0, print_step=25, cls_weight_pth = None\
    , add_timestamp = True, save_every_step=100, save_many_checkpoints=False, log_image_every=100):
    
    return {
        'workers':workers,
        'print_step':print_step,
        'save_every_step': save_every_step,
        'exp_name':exp_name,
        'logs_dir': logs_dir,
        'cls_weight_pth':cls_weight_pth,
        'add_timestamp':add_timestamp,
        'save_many_checkpoints':save_many_checkpoints,
        'log_image_every':log_image_every,
    }
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser(
        prog="ENet Runner for ScanNet and Perfception",
        description="Takes as input the perfception renders and the corresponding ground truth for semantic segmentation"
    )
    parser.add_argument('-res','--resume', default=False)
    parser.add_argument('-m','--mode',default="train")
    parser.add_argument('-c','--checkpoint', default=None, help="checkpoint to load name")
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )

    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.ginc,[])
    ### Add dataloader part here
    hparams = hparams_setup()
    setup_ = additional_setup()
    args = vars(args)
    args.update(hparams)
    args.update(setup_)
    args = argparse.Namespace(**args)
    if args.add_timestamp:
        import datetime
        ct = datetime.datetime.now()
        ts = str(int(ct.timestamp()))
        args.exp_name = args.exp_name +"_" + ts
    args.save_dir = os.path.join(args.logs_dir, args.exp_name)
    loaders, w_class, class_encoding = load_dataset(dataset)  
    train_loader, val_loader, test_loader = loaders
    if args.mode.lower() in {'train', 'full'}:
        #print("Weight classes", w_class.shape)
        writer = SummaryWriter(log_dir=args.save_dir)
        model = train(train_loader, val_loader, w_class, class_encoding,writer)
        if args.mode.lower() == 'full':
            test(model, test_loader, w_class, class_encoding)
    elif args.mode.lower() == 'test':
        num_classes = len(class_encoding)
        model = ENet(num_classes).to(device)
        optimizer = optim.Adam(model.parameters())
        model = load_checkpoint(model, optimizer, args.save_dir,
									  args.checkpoint)[0]
        print(model)
        test(model, test_loader, w_class, class_encoding)
