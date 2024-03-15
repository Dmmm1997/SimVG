# -----train----
# bash tools/dist_train.sh configs/beit3/detection/pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco.py 2
# bash tools/dist_train.sh configs/beit3/detection/pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco+.py 2
# bash tools/dist_train.sh configs/beit3/detection/pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcocog.py 2
# -----test----
# bash tools/dist_test.sh configs/beit3/detection/pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco.py 2 --load-from work_dir/beit3_finetune_coco/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco/det_best.pth
# bash tools/dist_test.sh configs/beit3/detection/pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco+.py 2 --load-from work_dir/beit3_finetune_coco/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco+/det_best.pth
# bash tools/dist_test.sh configs/beit3/detection/pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcocog.py 2 --load-from work_dir/beit3_finetune_coco/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcocog/det_best.pth

# bash tools/dist_train.sh configs/beit3/detection/diff_head/beit3_fchead_det_refcoco-unc_vitbp32_512_maxtoken20.py 2

# bash tools/dist_train.sh configs/beit3/detection/multibranch/decodergt#decoderpredict.py 2
# bash tools/dist_train.sh configs/beit3/detection/multibranch/decodergt#decoderpredict#tokengt.py 2
# bash tools/dist_train.sh configs/beit3/detection/multibranch/decodergt#tokengt.py 2
# bash tools/dist_train.sh configs/beit3/detection/multibranch/no_text_embed#aug_decodergt#decoderpredict#tokengt.py 2

bash tools/dist_train.sh "configs/beit3/detection/multibranch_0314/(3-15)noema#1.0token#1.0decoder#1.0auxdistill_klloss.py" 2
bash tools/dist_train.sh "configs/beit3/detection/multibranch_0314/(3-15)noema#1.0token#1.0decoder#1.0weighted_distill.py" 2
bash tools/dist_train.sh "configs/beit3/detection/multibranch_0314/(3-15)noema#1.0token#1.0decoder#textaug.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/multibranch_0314/(3-14)noema#1.0decoder#1.0token.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/multibranch_0314/(3-14)noema#1.0token#1.0decoder#1.0auxdistill.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/multibranch_0314/(3-14)noema#1.0token#1.0decoder#mlp_aux_loss.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/distill/(3-13)-noema-1.0decoder#1.0token#loadpretrain#mlp1layer.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/distill/(3-13)-noema-1.0decoder#1.0token#loadpretrain#mlp6layer.py" 2
# bash tools/dist_train.sh "configs/beit3/detection/distill/(3-13)-noema-1.0decoder#1.0token#loadpretrain#share_predicthead.py" 2
