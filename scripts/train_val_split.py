
from distutils.command.config import config
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import numpy as np
np.random.seed(13)
from configs.train import train_scenes_full

mod_scenes = [d[:-3] for d in train_scenes_full]
scenes = np.random.choice(np.unique(mod_scenes), 141, replace= False).tolist()

configs_path = '/workspace/AD_efnet/configs/'

train_opt = []
train = []
val_opt = []
val = []
for scene in train_scenes_full:
    if scene[:-3] in scenes:
        val.append(scene)
    else:
        train.append(scene)
        
for t in train:
    #if t[]
    if t[-2:] == '00':
        train_opt.append(t)

for t in val:
    #if t[]
    if t[-2:] == '00':
        val_opt.append(t)
        
with open(os.path.join(configs_path,"train.py"), "a") as outfile:
    outfile.write("\ntrain_scenes =[\n")
    
    outfile.write("\n".join(str("'" +item+"',") for item in train))
    
    outfile.write("]\n")
    
    outfile.write("train_scenes_opt =[\n")
    
    outfile.write("\n".join(str("'" +item+"',") for item in train_opt[::6]))
    
    outfile.write("]\n")

with open(os.path.join(configs_path,"val.py"), "a") as outfile:
    outfile.write("\nval_scenes =[\n")
    
    outfile.write("\n".join(str("'" +item+"',") for item in val))
    
    outfile.write("]\n")
    
    outfile.write("val_scenes_opt =[\n")
    
    outfile.write("\n".join(str("'" +item+"',") for item in val_opt[::6]))
    
    outfile.write("]\n")
