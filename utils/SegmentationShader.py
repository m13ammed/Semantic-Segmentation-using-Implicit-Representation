import torch
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from configs.scannet_constants import Map_to_20
from typing import Optional, Union
import time
Device = Union[str, torch.device]
class SegmentationShader(ShaderBase):
    
    def __init__(self, device, cameras ) -> None:
        super().__init__(device, cameras)
        self.keys = torch.Tensor(Map_to_20).to(device)
        
        
    def forward(self, fragments: Fragments, meshes: Meshes, labels: torch.Tensor, rgb=None, **kwargs) -> torch.Tensor:
        #labels = kwargs['labels']
        pix_to_face = fragments.pix_to_face
        meshes_tensor = torch.cat(meshes.faces_list())
        vert_idx = meshes_tensor[pix_to_face].squeeze(3)
        if labels.dim()==2:
            labels_20_k = torch.take(labels, vert_idx)#labels[vert_idx].squeeze(4)
        else:
            labels_20_k = torch.gather(labels, vert_idx, dim=1)
        labels_20,_ = torch.mode(labels_20_k, dim=-1) #majority vote
        #self.pallete = self.pallete.to(meshes.device)
        
        #index = torch.bucketize((meshes[...,0,-1]*255).type(torch.uint8), self.pallete)
        #labels = torch.take(self.keys, labels_20.type(torch.int64))
        labels_20 = torch.take(self.keys.view(-1,1), labels_20.long())#self.keys[labels_20.long()]
        if rgb is not None:
            color = rgb[vert_idx]
            color,_ = torch.mode(color, dim=3)
            color /= 255.0
            #color = color[...,0,:]
        else:
            color = rgb
        return labels_20, color
    
