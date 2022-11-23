import torch
from utils.dataloader import DataLoader
from utils.export_images import export_images
from utils.scannet_scene import ScanNetScene
from generate_groundtruth import generate_groundtruth_render
from pytorch3d.io import IO
import sys, getopt

if __name__ == "__main__":
    argv = sys.argv[1:]
    frame_skip=20
    compressed=False
    show_only=False
    batch_size=5
    try:
        opts, args = getopt.getopt(argv,"hf:c:s:",["frame_skip=","compressed=", "show_only="])
    except getopt.GetoptError:
        print ('main.py -f <frame_skip> -c <true|false> -s <true|false>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -f <frame_skip> -c <true|false> -s <true|false>')
            sys.exit()
        elif opt in ("-f", "--frame_skip"):
            frame_skip = int(arg)
        elif opt in ("-c", "--compressed"):
            compressed = bool(arg)
        elif opt in ("-s", "--show_only"):
            show_only = bool(arg)


    scannet_scenes = DataLoader(frame_skip=frame_skip).get_scannet_scenes()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else: device = torch.device("cpu")

    scannet_scene: ScanNetScene
    for scannet_scene in scannet_scenes:
        print(f"Generating Renders for scene {scannet_scene}")
        segmented_mesh = IO().load_mesh(scannet_scene.polygon, device=device)
        for batch in range(len(scannet_scene.pose_ids)):
            cameras, target_images = generate_groundtruth_render(
                scannet_scene=scannet_scene,
                mesh=segmented_mesh, 
                device=device, 
                batch_id=batch,
                batch_size=batch_size,
                compressed=compressed
            )
            export_images(target_images=target_images, show_only = show_only, batch=batch, batch_size=batch_size,frame_skip=frame_skip)



    