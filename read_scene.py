import os
import torch
import numpy as np


print_separator = "================="

class Scene:
    def __init__(self, perf_folder, orig_scannet_folder,scene_name):
        super().__init__()
        self.perf_folder = perf_folder
        self.scannet_folder = orig_scannet_folder
        self.scene_name = scene_name
        scene_folder = os.path.join(self.perf_folder, self.scene_name)
        
        ckpt_path = os.path.join(scene_folder, "last.ckpt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(scene_folder, "data.ckpt")
        self.ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        init_path = os.path.join(scene_folder, "init.npy")
        self.init_array = np.load(init_path)

        ##    
        thick_path = os.path.join(scene_folder, "thick.npy")
        self.thick_array = np.load(thick_path)

        ##
        trans_info_path = os.path.join(scene_folder,"trans_info.npz")
        self.trans_info = np.load(trans_info_path)

        self.parse_ckpt(self.ckpt)
    
    def parse_ckpt(self,ckpt):
        """
        Reads and parses the ckpt 'state_dict'
        """
        density = ckpt["state_dict"]["model.density_data"].detach().cpu()
        links_idx = ckpt["state_dict"]["model.links_idx"].detach().cpu()
        sh_data = ckpt['state_dict']['model.sh_data'].detach().cpu()
        valid = torch.where(density > 0.0)[0].long()
        density, links_idx = density[valid], links_idx[valid].long()
        sh_data = sh_data[valid].numpy().astype(np.float64)

        resolution = (
            ckpt["reso_list"]["reso_idx"] 
            if "reso_list" in ckpt.keys() else 
            [256, 256, 256]
        )

        links_idx = torch.stack(
            [
                links_idx // (resolution[1] * resolution[2]),
                links_idx % (resolution[1] * resolution[2]) // resolution[2],
                -links_idx % resolution[2],
            ],
            -1,
        )

        self.density = density
        self.links_idx = links_idx
        self.sh_data = sh_data
        self.resolution = resolution


    def debug_ckpt(self):
        ckpt = self.ckpt
        print(print_separator)
        print("Printing CKPT")
        for key in ckpt.keys():
            print(key)
            print(ckpt[key])
        print(print_separator)
        print("Printing State_Dict")
        for key in ckpt['state_dict'].keys():
            print(key)
            print(ckpt['state_dict'][key])
        print(print_separator)
        print("Printing SH")
        print("Length: ", len(ckpt['state_dict']['model.sh_data']))
        print(ckpt['state_dict']['model.sh_data'][0])
        print(print_separator)

    def inspect_init(self):
        init = self.init_array
        print(print_separator)
        print("Printing init")
        print("Length: ",len(init))
        print(init)
        print(print_separator)
        
    def inspect_thick(self):
        thick = self.thick_array
        print(print_separator)
        print("Printing -thick-")
        print("Length: ", len(thick))
        print(thick)
        print(print_separator)

    def inspect_trans(self):
        trans_info = self.trans_info
        print(print_separator)
        print("Printing trans info")
        for file in trans_info.files:
            print(file)
            print(trans_info[file])
        print(print_separator)
