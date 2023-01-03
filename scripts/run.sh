#python render_scannet.py --ginc ../configs/render_scannet.gin --scene_name plenoxel_scannet_scene0000_00
#python render_scannet.py --ginc ../configs/render_scannet.gin --scene_name plenoxel_scannet_scene0000_01
#python render_scannet.py --ginc ../configs/render_scannet.gin --scene_name plenoxel_scannet_scene0000_02
#python render_scannet.py --ginc ../configs/render_scannet.gin --scene_name plenoxel_scannet_scene0001_00
#python render_scannet.py --ginc ../configs/render_scannet.gin --scene_name plenoxel_scannet_scene0001_01


for d in /home/rozenberszki/Downloads/PeRFception-ScanNet/plenoxel_scannet_scene0***_**; do 


x=$(basename $d)
echo $x;
python render_scannet.py --ginc ../configs/render_scannet.gin --scene_name $x ; 


done