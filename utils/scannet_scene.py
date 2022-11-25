import numpy as np

class ScanNetScene:
    """
    :scene_name: path to the scene
    """
    def __init__(
        self,
        scene_name: str = "",
        pose_ids: np.array = np.array([]),
        poses = None,
        intrinsics = None,
        polygon:str = ""
    ):
        self.scene_name = scene_name
        self.pose_ids = pose_ids
        self.poses = poses
        self.intrinsics = intrinsics
        self.polygon = polygon