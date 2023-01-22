for d in /home/rozenberszki/Downloads/New_Poses/plenoxel_scannet_scene0***_**; do 


x=$(basename $d)
echo $x;
python clean_render_novel_views.py --scene_name $x --colored=True; 


done