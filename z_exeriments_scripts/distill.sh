# source_dir=configs/paper_exp/token_ablation/token_distill
# file_names=$(ls $source_dir)
# for file_name in $file_names
# do
#   related_filename=$source_dir/$file_name
#   bash tools/dist_train.sh $related_filename 2
# done

# source_dir=configs/paper_exp/token_ablation/decoder_pretrain_token_distill/distill_weigth
# file_names=$(ls $source_dir)
# for file_name in $file_names
# do
#   related_filename=$source_dir/$file_name
#   bash tools/dist_train.sh $related_filename 2
# done

# source_dir=configs/paper_exp/token_ablation/balanced_distill
# file_names=$(ls $source_dir)
# for file_name in $file_names
# do
#   related_filename=$source_dir/$file_name
#   bash tools/dist_train.sh $related_filename 2
# done


source_dir=configs/paper_exp/token_ablation/two-stage_distill
file_names=$(ls $source_dir)
for file_name in $file_names
do
  related_filename=$source_dir/$file_name
  # train
  # bash tools/dist_train.sh $related_filename 2

  # test -----
  # basename without .py
  file_name_without_suffix=$(basename "$related_filename" .py)
  file_dir_suffix=$source_dir/$file_name_without_suffix
  checkpoint_dir=$(echo "$file_dir_suffix" | sed 's/configs/work_dir/g')
  latest_folder=$(ls -t "$checkpoint_dir" | head -n 1)
  echo $latest_folder
  checkpoint=$checkpoint_dir/$latest_folder/det_best.pth
  echo $checkpoint
  python tools/test.py $related_filename --load-from $checkpoint
  # test -----
done
