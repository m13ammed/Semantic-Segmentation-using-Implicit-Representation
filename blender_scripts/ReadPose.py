import bpy
import numpy as np
import math
import os


def getCamParams(cam):
    location, rotation = cam.matrix_world.decompose()[0:2]
    ## 4x4 raw matrix
    cam_4x4 = np.array(cam.matrix_world)
    return cam_4x4

def get_K(cam):
    cam_data = cam.data
    print(cam_data.type)
    if(cam_data.type!= 'PERSP'):
        raise ValueError('Camera type must be perspective')
    
    f_in_mm = cam_data.lens
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    
    
    ## Assuming it is horizontal
    ## cam_data.sensor_fit should be AUTO
    
    print(scene.render.pixel_aspect_x * resolution_x_in_px)
    print(scene.render.pixel_aspect_y * resolution_y_in_px)
    
    sensor_size_in_mm = cam_data.sensor_width
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / resolution_x_in_px
    fx = 1 / pixel_size_mm_per_px
    fy = 1 / pixel_size_mm_per_px / pixel_aspect_ratio
    
    cx = resolution_x_in_px / 2 - cam_data.shift_x * resolution_x_in_px
    cy = resolution_y_in_px / 2 + cam_data.shift_y * resolution_x_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels
    
    K=[
        [fx, skew, cx],
        [0, fy, cy],
        [0,0,1]
    ]
    
    return np.array(K)

def K_intrinsics(cam):
    # get the relevant data
    cam = cam.data
    scene = bpy.context.scene
    # assume image is not scaled
    assert scene.render.resolution_percentage == 100
    # assume angles describe the horizontal field of view
    assert cam.sensor_fit != 'VERTICAL'

    f_in_mm = cam.lens
    sensor_width_in_mm = cam.sensor_width

    w = scene.render.resolution_x
    h = scene.render.resolution_y

    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect

    # yes, shift_x is inverted. WTF blender?
    c_x = w * (0.5 - cam.shift_x)
    # and shift_y is still a percentage of width..
    c_y = h * 0.5 + w * cam.shift_y

    K = [[f_x, 0, c_x],
         [0, f_y, c_y],
         [0,   0,   1]]
         
    return np.array(K)

def set_intrinsics(cam, read_k):
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    sensor_width_in_mm = cam.data.sensor_width
    
    
    
    fx = read_k[0,0]
    fy = read_k[1,1]
    
    cx = read_k[0,2]
    cy = read_k[1,2]
    
    cam.data.shift_x = -(cx / w - 0.5)
    cam.data.shift_y = (cy - 0.5 * h)/w
    cam.data.lens = fx / w * sensor_width_in_mm
    
    pixel_aspect = fy / fx
    
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = pixel_aspect
    
def set_extrinsics(cam, read_extrinsics):
    pose = read_extrinsics.copy()
    cam.matrix_world = pose.T
    x = cam.rotation_euler[0]
    x = x - math.radians(180)
    cam.rotation_euler[0] = x
    
def extract_gt_extrinsics(cam):
    scene = bpy.context.scene
    m_before = np.array(cam.matrix_world)
    x = cam.rotation_euler[0]
    x = x - math.radians(180)
    cam.rotation_euler[0] = x
    scene.frame_set(scene.frame_current)
    m = np.array(cam.matrix_world)
    x = x + math.radians(180)
    cam.rotation_euler[0] = x
    return m

def view_all_poses(intrinsics, pose_dir):
    read_k = intrinsics
    scene = bpy.context.scene
    for file in sorted(os.listdir(pose_dir)):
        if(file == ".DS_Store"): continue
        pose_file = os.path.join(pose_dir, file)
        pose_id = file.split(".")[0]
        camera_data = bpy.data.cameras.new(name=pose_id)
        camera_object = bpy.data.objects.new(pose_id, camera_data)
        scene.collection.objects.link(camera_object)
        cam = bpy.data.objects[pose_id]
        set_intrinsics(cam, read_k)
        read_extrinsics = np.loadtxt(pose_file)
        set_extrinsics(cam, read_extrinsics)
        
def new_cam_from_pose(name, intrinsics, pose):
    camera_data = bpy.data.cameras.new(name=name)
    camera_object = bpy.data.objects.new(name, camera_data)
    scene.collection.objects.link(camera_object)
    cam = bpy.data.objects[name]
    set_intrinsics(cam, intrinsics)
    set_extrinsics(cam, pose)
#    x = cam.rotation_euler[0]
#    x = x - math.radians(180)
#    cam.rotation_euler[0] = x
#    y = cam.rotation_euler[1]
##    y = y - math.radians(180)
#    cam.rotation_euler[1] = y
        

def rotate_point_matrix(init_pose, angle, rot_point):
    # Translate point to origin
    point = rot_point.copy()
    point[1] -= 3
    translate_matrix = np.eye(4)
    translate_matrix[:3,3:] = -point

    # Rotate point
    angle_rad = math.radians(angle)
    rotate_matrix = [
        [math.cos(angle_rad), -math.sin(angle_rad), 0, 0],
        [math.sin(angle_rad), math.cos(angle_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    translate_back_matrix = np.eye(4)
    translate_back_matrix[:3, 3:] = point
    transformation_matrix = np.zeros(4)

    transformation_matrix = translate_back_matrix @ rotate_matrix
    transformation_matrix = transformation_matrix @ translate_matrix
    
    return transformation_matrix @ init_pose
    
def create_dome(init_pose, angle, point, step_size, intrinsics):
    for i in range(step_size, angle, step_size):
#        print("=========")
#        print(init_pose, angle,point, step_size, intrinsics)
        new_pose = rotate_point_matrix(init_pose.copy(), i, point.copy())
#        print(new_pose)
        new_cam_from_pose("dome_"+str(i), intrinsics, new_pose)
        
        
if __name__ == "__main__":
#    cam = bpy.data.objects['Camera']
    
    
    scene = bpy.context.scene
    bpy.context.scene.render.resolution_x = 1296
    bpy.context.scene.render.resolution_y = 968
#    getCamParams(cam)
#    K = get_K(cam)
#    K2 = K_intrinsics(cam)
    
    ## Okay so now we have the intrinsics and extrinsics (I think)
    ## Let's try to read the pose and see
    
    intrinsics_dir = "/Users/kghandour/development/ScanNet/scans/scene0000_00/intrinsic/intrinsic_color.txt"
    pose_dir = "/Users/kghandour/development/ScanNet/scans/scene0000_00/pose/"
    
    read_k = np.loadtxt(intrinsics_dir)
#    view_all_poses(read_k, pose_dir)
    ext_cam = bpy.data.objects['new_extrinsics']
    pose = extract_gt_extrinsics(ext_cam)
    print(pose)
    
    create_dome(pose, 360, pose[:3,3:], 15, read_k)
    
    np.savetxt("/Users/kghandour/development/text_ext.txt", pose)
        
 
#    
#    
#    K3 = K_intrinsics(cam)
#    
#    read_extrinsics = np.loadtxt(pose_dir)
#    print(read_extrinsics)
#    
#    set_extrinsics(cam, read_extrinsics)
#    gt_pose = extract_gt_extrinsics(cam)
#    print(gt_pose)
    
    
    
#    camera_data = bpy.data.cameras.new(name='20')
#    camera_object = bpy.data.objects.new('20', camera_data)
#    scene.collection.objects.link(camera_object)
#    cam20 = bpy.data.objects['20']
#    
#    read_extrinsics2 = np.loadtxt(pose_dir_2)
#    set_extrinsics(cam20, read_extrinsics2)
    
    
    
    
    