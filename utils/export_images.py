import numpy as np
from PIL import Image
import os
import gin
@gin.configurable()
def export_images(labels, target_images, show_only=False, scene_name="1", frame_ids = [], output_segmentation = './out/', colored =False):
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    for frame_id, label, imgg in zip(frame_ids, labels, target_images):
        if(colored):
            image = (clamp_and_detach(imgg[ ..., :3]))
            tensor = image*255
            tensor = np.array(tensor, dtype=np.uint8)
            if np.ndim(tensor)>3:
                assert tensor.shape[0] == 1
                tensor = tensor[0]
            img = Image.fromarray(tensor)
        else: ##GT values
            image = np.array(label.cpu(), dtype=np.uint8)
            img = image#Image.fromarray(image)
        if(show_only):
            img.show()
        else:
            os.makedirs(output_segmentation, exist_ok=True)
            path_scene = os.path.join(output_segmentation, scene_name)
            os.makedirs(path_scene, exist_ok=True)
            if(colored):
                img.save(os.path.join(path_scene,str(frame_id)+".png"))
            else:
                np.save(os.path.join(path_scene,str(frame_id)+".npy"),img)