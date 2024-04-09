# -----train----
# bash tools/dist_train.sh "configs/fgqg/normal_exp/ViTBaseP32-1.0decoder-1.0token-30ep-512hw-3layer-tgqg-refcocounc.py" 2
# bash tools/dist_train.sh "configs/fgqg/normal_exp/ViTBaseP32-1.0decoder-1.0token-30ep-512hw-3layer-3query-refcocounc.py" 2
# bash tools/dist_train.sh "configs/fgqg/normal_exp/ViTBaseP32-1.0decoder-1.0token-30ep-512hw-3layer-tgqg-textaug-refcocounc.py" 2
# -----test----
# bash tools/dist_test.sh configs/beit3/detection/pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco.py 2 --load-from work_dir/beit3_finetune_coco/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco/det_best.pth
# bash tools/dist_test.sh configs/beit3/detection/pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco+.py 2 --load-from work_dir/beit3_finetune_coco/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco+/det_best.pth
# bash tools/dist_test.sh configs/beit3/detection/pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcocog.py 2 --load-from work_dir/beit3_finetune_coco/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcocog/det_best.pth

# bash tools/dist_train.sh "configs/exp_for_paper/decoder_ablation/ViTBaseP32-1.0decoder-30ep-512hw-1layer-refcocounc.py" 2
# bash tools/dist_train.sh "configs/exp_for_paper/decoder_ablation/ViTBaseP32-1.0decoder-30ep-512hw-2layer-refcocounc.py" 2
# bash tools/dist_train.sh "configs/exp_for_paper/decoder_ablation/ViTBaseP32-1.0decoder-30ep-512hw-3layer-refcocounc.py" 2

# bash tools/dist_train.sh "configs/exp_for_paper/decoder_ablation/ViTBaseP32-1.0decoder-30ep-512hw-4layer-refcocounc.py" 2
# bash tools/dist_train.sh "configs/exp_for_paper/decoder_ablation/ViTBaseP32-1.0decoder-30ep-512hw-5layer-refcocounc.py" 2
# bash tools/dist_train.sh "configs/exp_for_paper/decoder_ablation/ViTBaseP32-1.0decoder-30ep-512hw-6layer-refcocounc.py" 2

# bash z_exeriments_scripts/layers.sh
# bash z_exeriments_scripts/distill.sh


bash tools/dist_train.sh "configs/paper_exp/sota_model/phrase/noema_ViTBaseP32-1.0decoder-20ep-640hw-tgqg_layer2_flickr30k_distill.py" 2
bash tools/dist_train.sh "configs/paper_exp/sota_model/phrase/noema_ViTBaseP32-1.0decoder-20ep-640hw-tgqg_layer2_referit_distill.py" 2
# bash tools/dist_train.sh "configs/beit3/exp_for_paper/baseline/ViTBaseP32-1.0decoder-40ep-640hw-refcocogumd.py" 2
# bash tools/dist_train.sh "configs/beit3/exp_for_paper/baseline/ViTBaseP32-1.0decoder-40ep-640hw-refcocoggoogle.py" 2


# bash tools/dist_train.sh "configs/beit3/detection/multibranch_0314/(3-14)noema#1.0token#1.0decoder#1.0auxdistill.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/multibranch_0314/(3-14)noema#1.0token#1.0decoder#mlp_aux_loss.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/distill/(3-13)-noema-1.0decoder#1.0token#loadpretrain#mlp1layer.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/distill/(3-13)-noema-1.0decoder#1.0token#loadpretrain#mlp6layer.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/distill/(3-13)-noema-1.0decoder#1.0token#loadpretrain#share_predicthead.py" 2


