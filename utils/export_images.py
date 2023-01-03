import numpy as np
from PIL import Image
import os
<<<<<<< HEAD

def export_images(labels, target_images, show_only=False, batch=0,batch_size=5,frame_skip=20, scene_name="1", colored=True, frame_id = 0):
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    for idx in range(target_images.shape[0]):
        if(colored):
            image = (clamp_and_detach(target_images[idx, ..., :3]))
=======
import gin
@gin.configurable()
def export_images(labels, target_images, show_only=False, scene_name="1", frame_ids = [], output_segmentation = './out/', colored =False):
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    for frame_id, label, imgg in zip(frame_ids, labels, target_images):
        if(colored):
            image = (clamp_and_detach(imgg[ ..., :3]))
>>>>>>> main
            tensor = image*255
            tensor = np.array(tensor, dtype=np.uint8)
            if np.ndim(tensor)>3:
                assert tensor.shape[0] == 1
                tensor = tensor[0]
            img = Image.fromarray(tensor)
        else: ##GT values
<<<<<<< HEAD
            image = np.array(labels[idx].cpu(), dtype=np.uint8)
=======
            image = np.array(label.cpu(), dtype=np.uint8)
>>>>>>> main
            img = image#Image.fromarray(image)
        if(show_only):
            img.show()
        else:
<<<<<<< HEAD
            os.makedirs(f"./out/", exist_ok=True)
            os.makedirs(f"./out/{scene_name}/", exist_ok=True)
            if(colored):
                img.save(os.path.join("./out/",scene_name,str(frame_id)+".png"))
            else:
                np.save(os.path.join("./out/",scene_name,str(frame_id)+".npy"),img)
=======
            os.makedirs(output_segmentation, exist_ok=True)
            path_scene = os.path.join(output_segmentation, scene_name)
            os.makedirs(path_scene, exist_ok=True)
            if(colored):
                img.save(os.path.join(path_scene,str(frame_id)+".png"))
            else:
                np.save(os.path.join(path_scene,str(frame_id)+".npy"),img)
>>>>>>> main
