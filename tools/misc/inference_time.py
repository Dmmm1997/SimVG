# -*- coding: utf-8 -*-
import argparse
import torch
from thop import profile, clever_format
from simvg.models import build_model
import time
from mmcv import Config, DictAction
from simvg.datasets import build_dataset, build_dataloader
from simvg.datasets import extract_data
from tqdm import tqdm

def calc_flops_params(model,
                      inputs
                      ):
    total_ops, total_params = profile(
        model, inputs, verbose=False)
    macs, params = clever_format([total_ops, total_params], "%.3f")
    return macs, params


parser = argparse.ArgumentParser(description='Inference Time')
parser.add_argument('--config', default='work_dir/unimodel/pretrain/AAAI/uni-320/20240630_113050/20240630_113050_uni-320.py',
                    type=str, help='save model path')
parser.add_argument('--checkpoint', default='work_dir/unimodel/pretrain/AAAI/uni-320/20240630_113050/segm_best.pth',
                    type=str, help='save model path')
parser.add_argument('--test_samples_number', default=100, type=int, help='width')
args = parser.parse_args()

cfg = Config.fromfile(args.config)

cfg._cfg_dict['data']['samples_per_gpu'] = 1

datasets_cfgs = cfg.data.val_refcoco_unc

datasets = build_dataset(datasets_cfgs)
dataloaders = build_dataloader(cfg, datasets)

model = build_model(cfg.model,word_emb=datasets.word_emb,
                        num_token=datasets.num_token).cuda()
model = model.eval()

for inputs in dataloaders:
    # if "gt_bbox" in inputs:
    #     with_bbox = True
    #     if isinstance(inputs["gt_bbox"], torch.Tensor):
    #         inputs["gt_bbox"] = [inputs["gt_bbox"][ind] for ind in range(inputs["gt_bbox"].shape[0])]
    #         gt_bbox = inputs.pop("gt_bbox")
    #     else:
    #         gt_bbox = inputs.pop("gt_bbox").data[0]
            
    if "gt_bbox" in inputs:
        with_bbox = True
        gt_bbox = inputs.pop("gt_bbox").data[0]
    if "gt_mask_rle" in inputs:
        with_mask = True
        gt_mask = inputs.pop("gt_mask_rle").data[0]
    if "is_crowd" in inputs:
        inputs.pop("is_crowd").data[0]

    img_metas = inputs["img_metas"].data[0]

    inputs = extract_data(inputs)

    predictions = model(
        **inputs,
        return_loss=False,
        rescale=False,
        with_bbox=with_bbox,
        visual=False,
    )
    tmp_inputs = inputs
    break

tmp_inputs["return_loss"]=False
tmp_inputs["with_bbox"]=True

# 预热
for _ in range(10):
    model(**tmp_inputs)

since = time.time()
for _ in tqdm(range(args.test_samples_number)):
    model(**tmp_inputs)

print("inference_time = {}ms/iter".format((time.time()-since)/args.test_samples_number*1000))

# thop计算MACs
macs, params = calc_flops_params(
    model, tmp_inputs)

print("total_macs:{}, total_params:{}".format(macs, params))