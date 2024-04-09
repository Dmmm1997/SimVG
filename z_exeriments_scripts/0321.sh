source_dir=configs/paper_exp/sota_model/sota_large
file_names=$(ls $source_dir)
for file_name in $file_names
do
  related_filename=$source_dir/$file_name
  # train
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PORT=29500 bash tools/dist_train.sh $related_filename 2

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
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PORT=29510 bash tools/dist_test.sh $related_filename 2 --load-from $checkpoint
  # test -----
done
