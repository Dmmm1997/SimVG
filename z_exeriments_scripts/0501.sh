
file_names=('configs/segmentation/beit3/beit3-seg_512_refcoco-unc.py' 'configs/segmentation/seqtr/seqtr_segm_refcoco-unc.py')
for related_filename in $file_names
do
  # train
  CUDA_VISIBLE_DEVICES=0,1 PORT=29500 bash tools/dist_train.sh $related_filename 2

  # test -----
  # basename without .py
  file_name_without_suffix=$(basename "$related_filename" .py)
  file_dir_suffix=$source_dir/$file_name_without_suffix
  checkpoint_dir=$(echo "$file_dir_suffix" | sed 's/configs/work_dir/g')
  latest_folder=$(ls -t "$checkpoint_dir" | head -n 1)
  echo $latest_folder
  # checkpoint=$checkpoint_dir/$latest_folder/det_best.pth
  checkpoint=$checkpoint_dir/$latest_folder/latest.pth
  echo $checkpoint
  CUDA_VISIBLE_DEVICES=0,1 PORT=29510 bash tools/dist_test.sh $related_filename 2 --load-from $checkpoint
  # test -----
done
