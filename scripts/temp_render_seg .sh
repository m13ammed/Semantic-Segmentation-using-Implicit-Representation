for d in /home/rozenberszki/Downloads/PeRFception-ScanNet/plenoxel_scannet_scene03**_**; do 


x=$(basename $d)
echo $x;
python render_segmentation.py --scene_name $x  ; 


done
for d in /home/rozenberszki/Downloads/PeRFception-ScanNet/plenoxel_scannet_scene04**_**; do 


x=$(basename $d)
echo $x;
python render_segmentation.py --scene_name $x  ; 


done
for d in /home/rozenberszki/Downloads/PeRFception-ScanNet/plenoxel_scannet_scene05**_**; do 


x=$(basename $d)
echo $x;
python render_segmentation.py --scene_name $x  ; 


done
for d in /home/rozenberszki/Downloads/PeRFception-ScanNet/plenoxel_scannet_scene06**_**; do 


x=$(basename $d)
echo $x;
python render_segmentation.py --scene_name $x  ; 


done
for d in /home/rozenberszki/Downloads/PeRFception-ScanNet/plenoxel_scannet_scene07**_**; do 


x=$(basename $d)
echo $x;
python render_segmentation.py --scene_name $x  ; 


done