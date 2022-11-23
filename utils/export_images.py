import numpy as np
from PIL import Image
import os

def export_images(target_images, show_only=False, batch=0,batch_size=5,frame_skip=20, scene_name="1"):
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    for idx in range(target_images.shape[0]):
        image = (clamp_and_detach(target_images[idx, ..., :3]))
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        img = Image.fromarray(tensor)
        if(show_only):
            img.show()
        else:
            os.makedirs(f"./out/", exist_ok=True)
            os.makedirs(f"./out/{scene_name}/", exist_ok=True)
            frame_id = batch*batch_size*frame_skip+idx*frame_skip
            img.save(os.path.join("./out/",scene_name,str(frame_id)+".png"))