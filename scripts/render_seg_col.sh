for d in /home/rozenberszki/Downloads/PeRFception-ScanNet/plenoxel_scannet_scene0***_**; do 


x=$(basename $d)
echo $x;
python render_segmentation.py --scene_name $x --colored=True ; 


done