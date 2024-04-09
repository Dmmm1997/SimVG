# source_dir=configs/paper_exp/pretrain/finetune_mix
# # file_names=$(ls $source_dir)
# file_names=("noema#finetune#refcoco+.py")
# for file_name in $file_names
# do
#   related_filename=$source_dir/$file_name
#   # train
#   CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 bash tools/dist_train.sh $related_filename 4

#   # test -----
#   # basename without .py
#   file_name_without_suffix=$(basename "$related_filename" .py)
#   file_dir_suffix=$source_dir/$file_name_without_suffix
#   checkpoint_dir=$(echo "$file_dir_suffix" | sed 's/configs/work_dir/g')
#   latest_folder=$(ls -t "$checkpoint_dir" | head -n 1)
#   echo $latest_folder
#   checkpoint=$checkpoint_dir/$latest_folder/det_best.pth
#   echo $checkpoint
#   CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29511 bash tools/dist_test.sh $related_filename 4 --load-from $checkpoint
#   # test -----
# done


source_dir=configs/paper_exp/pretrain/two-stage_distill_mix
file_names=$(ls $source_dir)
for file_name in $file_names
do
  related_filename=$source_dir/$file_name
  # train
  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 bash tools/dist_train.sh $related_filename 4

  # test -----
  # basename without .py
  file_name_without_suffix=$(basename "$related_filename" .py)
  file_dir_suffix=$source_dir/$file_name_without_suffix
  checkpoint_dir=$(echo "$file_dir_suffix" | sed 's/configs/work_dir/g')
  latest_folder=$(ls -t "$checkpoint_dir" | head -n 1)
  echo $latest_folder
  checkpoint=$checkpoint_dir/$latest_folder/det_best.pth
  echo $checkpoint
  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29511 bash tools/dist_test.sh $related_filename 4 --load-from $checkpoint
  # test -----
done



