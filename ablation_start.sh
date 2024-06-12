source_dir=configs/unimodel/pretrain
# file_names=$(ls $source_dir)
# for file_name in $file_names
# file_names=(B2S0.01-S2B0.05.py B2S0.05-S2B0.025.py S2B0.05.py)
file_names=(uni-256-pixel-level-seg0.03-box0.01-nogoogle-ep30.py)
for file_name in "${file_names[@]}"
do
  related_filename=$source_dir/$file_name
  # train
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 PORT=29510 bash tools/dist_train.sh $related_filename 6

  # test -----
  # basename without .py
  file_name_without_suffix=$(basename "$related_filename" .py)
  file_dir_suffix=$source_dir/$file_name_without_suffix
  checkpoint_dir=$(echo "$file_dir_suffix" | sed 's/configs/work_dir/g')
  latest_folder=$(ls -t "$checkpoint_dir" | head -n 1)
  echo $latest_folder
  checkpoint=$checkpoint_dir/$latest_folder/segm_best.pth
  echo $checkpoint
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 PORT=29520 bash tools/dist_test.sh $related_filename 6 --load-from $checkpoint
  test -----
done
