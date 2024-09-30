import torchvision
import torch
from collections import defaultdict
import os
import json
from tqdm import tqdm


class ModulatedDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file):
        super(ModulatedDetection, self).__init__(img_folder, ann_file)

    def __getitem__(self, idx):
        img, target = super(ModulatedDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        image_id = coco_img["original_id"]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
        height, width = coco_img["height"], coco_img["width"]
        bboxes = []
        bboxes = [t["bbox"] for t in target]
        target = {
            "image_id": image_id,
            "annotations": target,
            "expressions": caption,
            "height": height,
            "width": width,
            "bbox": bboxes,
        }
        return target


img_folder = "/data/datasets/detection/coco2014/train2014"
ann_file = "/home/dmmm/demo_mirror/REC/gRefCOCO/mdetr/mdetr_annotations"

result_savepath_json = "data/annotations/grefs/instances.json"

res_for_save = defaultdict(list)
for subset in ["train", "val", "testA", "testB"]:
    # for subset in ["val"]:
    format_dict = {}
    print("Start {}".format(subset))
    ann_file_subname = os.path.join(ann_file, "finetune_grefcoco_{}.json".format(subset))
    dataset = ModulatedDetection(img_folder, ann_file_subname)
    
    for ind in tqdm(range(len(dataset.ids))):
        target = dataset.__getitem__(ind)
        if subset=="train":
            img_id = target["image_id"]
            annotations = target["annotations"]
            expressions = target["expressions"]
            height = target["height"]
            width = target["width"]
            bbox = target["bbox"]
            if img_id not in format_dict:
                format_dict[img_id] = {
                    "image_id": img_id,
                    "annotations": [annotations],
                    "expressions": [expressions],
                    "height": height,
                    "width": width,
                    "bbox": [bbox],
                }
            else:
                format_dict[img_id]["annotations"].append(annotations)
                format_dict[img_id]["expressions"].append(expressions)
                format_dict[img_id]["bbox"].append(bbox)
        else:
            img_id = target["image_id"]
            annotations = target["annotations"]
            expressions = target["expressions"]
            height = target["height"]
            width = target["width"]
            bbox = target["bbox"]
            format_dict[ind] = {
                "image_id": img_id,
                "annotations": [annotations],
                "expressions": [expressions],
                "height": height,
                "width": width,
                "bbox": [bbox],
            }
    
    for k, v in format_dict.items():
        res_for_save[subset].append(v)
    print("Done {}".format(subset))

with open(result_savepath_json, "w") as file:
    json.dump(res_for_save, file, indent=4)
