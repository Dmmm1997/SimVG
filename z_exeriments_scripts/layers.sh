wokeyi# exp decoder layers
# source_dir=configs/paper_exp/decoder_ablation/layers
# file_names=$(ls $source_dir)
# for file_name in $file_names
# do
#   related_filename=$source_dir/$file_name
#   bash tools/dist_train.sh $related_filename 2
# done


# exp mlp layers
source_dir=configs/paper_exp/token_ablation/layers
file_names=$(ls $source_dir)
for file_name in $file_names
do
  related_filename=$source_dir/$file_name
  bash tools/dist_train.sh $related_filename 2
done
