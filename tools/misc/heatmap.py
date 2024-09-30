import cv2
import numpy as np

from mmcv.runner import load_checkpoint
from simvg.datasets.pipelines import Compose
from simvg.datasets import extract_data
from mmcv.parallel import collate
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv import Config
import random
import argparse
import os
from simvg.models import build_model
from simvg.datasets import build_dataset

from simvg.utils.gradcam_models import GradCAM_BeiT, GradCAM_SeqTR

np.random.seed(300)


def prepare_img(cfg, img, text, token2idx=None):
    """
    prepare function
    """
    cfg.data.val.pipeline[0].type = "LoadFromRawSource"
    test_pipeline = Compose(cfg.data.val.pipeline) 
    result = {}
    ann = {}
    ann["bbox"] = [0, 0, 0, 0]
    ann["category_id"] = 0
    ann["expressions"] = [text]
    result["ann"] = ann
    result["which_set"] = "val"
    result["filepath"] = img
    result["token2idx"] = token2idx
    result["imgsfile"] = img

    data = test_pipeline(result)
    data = collate([data], samples_per_gpu=1)
    inputs = extract_data(data)

    return inputs


def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.0
    return np.uint8(image)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    heatmap = cv2.resize(heatmap, dsize=(image.shape[1], image.shape[0]))

    # merge heatmap to original image
    # cam = heatmap + np.float32(image)
    return (heatmap * 255).astype(np.uint8)


def draw_label_type(draw_img, bbox, label, line=5, label_color=None):
    if label_color == None:
        label_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    labelSize = cv2.getTextSize(label + "0", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img, bbox[:2], bbox[2:], color=label_color, thickness=line)
    else:
        cv2.rectangle(draw_img, bbox[:2], bbox[2:], color=label_color, thickness=line)


def plot_cam_image(img, mask, box, class_id, score, bbox_index, COLORS, label_names):
    """
    Merge the CAM map to original image
    """
    height, width = img.shape[:2]

    image_tmp = img.copy()
    x1, y1, x2, y2 = box
    # predict_box = img[y1:y2, x1:x2]
    image_heatmap = gen_cam(img, mask)
    image_cam = img * 0.4 + image_heatmap * 0.6

    image_tmp = image_cam
    image_tmp = cv2.rectangle(image_tmp, (x1, y1), (x2, y2), COLORS[class_id], int(width / 112))

    label = label_names[class_id]
    ref_length = min(height, width)
    cv2.putText(
        image_tmp,
        label + ": " + "%.2f" % (score * 100) + "%",
        (x1, y1 - int(height / 100)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.001 * ref_length + 0.72,
        COLORS[class_id],
        2,
    )

    return image_tmp


def mkdir(name):
    """
    Create folder
    """
    isExists = os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0


def main(args):
    # Init your model
    config = args.config
    cfg = Config.fromfile(config)
    checkpoint_path = args.checkpoint
    device = args.device
    text = args.text
    dataset = build_dataset(cfg.data.train)
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    label_names = ["object"]

    model.CLASSES = label_names
    COLORS = np.random.uniform(0, 255, size=(len(label_names), 3))

    # grad_cam = GradCAM_BeiT(model, 'head.transformer.decoder.layers.0')
    grad_cam = GradCAM_SeqTR(model, "head.transformer.encoder.layers.3")

    image = cv2.imread(args.image_path)
    data = prepare_img(cfg, args.image_path, text, dataset.token2idx)

    # First is the data, second is the index of the predicted bbox
    mask, box, class_id, score = grad_cam(**data)

    COLORS = np.random.uniform(0, 255, size=(len(label_names), 3))

    # draw_image = image_cam.copy()
    draw_image = plot_cam_image(image, mask, box, class_id, score, args.bbox_index, COLORS, label_names)

    mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.image_path.split("/")[-1].split(".")[0] + "-bbox-id-" + str(args.bbox_index) + "xyh.jpg")

    cv2.imwrite(save_path, draw_image)


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM")
    # general
    parser.add_argument(
        "--config",
        type=str,
        default="work_dir/paper_exp/pretrain/finetune_large_two_stage_distill/1-refcoco/20240517_120009/20240517_120009_1-refcoco.py",
        help="RetinaNet configuration.",
    )
    parser.add_argument("--checkpoint", type=str, default="work_dir/paper_exp/pretrain/finetune_large_two_stage_distill/1-refcoco/20240517_120009/det_best.pth", help="checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:0", help="device.")
    parser.add_argument("--image-path", type=str, default="data/demo.jpg", help="image path.")
    parser.add_argument("--text", type=str, default="the chair", help="image path.")
    parser.add_argument("--bbox-index", type=int, default=0, help="index.")
    parser.add_argument("--save-dir", type=str, default="visualize/GradCAM/SeqTR", help="save dir.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
