from simvg.models.builder import build_model
from visualizer import get_local
get_local.activate() # 激活装饰器
import argparse
import os.path
from functools import partial
from simvg.datasets.builder import build_dataset
import cv2
import mmcv
import numpy as np
from mmcv import Config, DictAction
from simvg.datasets import extract_data
from mmcv.parallel import collate
from torch.utils.data import DataLoader
from simvg.utils.det_cam_visualizer import DetAblationLayer, DetBoxScoreTarget, DetCAMModel, DetCAMVisualizer, EigenCAM, FeatmapAM, reshape_transform
from copy import deepcopy
import torch

try:
    from pytorch_grad_cam import AblationCAM, EigenGradCAM, GradCAM, GradCAMPlusPlus, LayerCAM, XGradCAM
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install ' "3rd party package pytorch_grad_cam.")

GRAD_FREE_METHOD_MAP = {
    "ablationcam": AblationCAM,
    "eigencam": EigenCAM,
    # 'scorecam': ScoreCAM, # consumes too much memory
    "featmapam": FeatmapAM,
}

GRAD_BASE_METHOD_MAP = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "xgradcam": XGradCAM, "eigengradcam": EigenGradCAM, "layercam": LayerCAM}

ALL_METHODS = list(GRAD_FREE_METHOD_MAP.keys() | GRAD_BASE_METHOD_MAP.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CAM")
    parser.add_argument("--img", default="data/demo.jpg", help="Image file")
    parser.add_argument("--text", default="the chair", help="Text")
    parser.add_argument(
        "--config",
        default="work_dir/paper_exp/pretrain/finetune_large_two_stage_distill/1-refcoco/20240517_120009/20240517_120009_1-refcoco.py",
        help="Config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dir/paper_exp/pretrain/finetune_large_two_stage_distill/1-refcoco/20240517_120009/det_best.pth",
        help="Checkpoint file",
    )
    parser.add_argument("--method", default="gradcam", help="Type of method to use, supports " f'{", ".join(ALL_METHODS)}.')
    parser.add_argument(
        "--target-layers",
        default=["vis_enc.beit3.encoder.layers[11]"],
        # default=["head.transformer.decoder.layers[2].attentions[0].attn"],
        nargs="+",
        type=str,
        help="The target layers to get CAM, if not set, the tool will " "specify the backbone.layer3",
    )
    parser.add_argument("--preview-model", default=False, action="store_true", help="To preview all the model layers")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Bbox score threshold")
    parser.add_argument("--topk", type=int, default=10, help="Topk of the predicted result to visualizer")
    parser.add_argument(
        "--max-shape",
        nargs="+",
        type=int,
        default=20,
        help="max shapes. Its purpose is to save GPU memory. " "The activation map is scaled and then evaluated. " "If set to -1, it means no scaling.",
    )
    parser.add_argument("--no-norm-in-bbox", action="store_true", help="Norm in bbox of cam image")
    parser.add_argument("--aug-smooth", default=False, action="store_true", help="Wether to use test time augmentation, default not to use")
    parser.add_argument(
        "--eigen-smooth", default=False, action="store_true", help="Reduce noise by taking the first principle componenet of " "``cam_weights*activations``"
    )
    parser.add_argument("--out-dir", default="visualize/results/", help="dir to output file")

    # Only used by AblationCAM
    parser.add_argument("--batch-size", type=int, default=1, help="batch of inference of AblationCAM")
    parser.add_argument(
        "--ratio-channels-to-ablate",
        type=int,
        default=0.5,
        help="Making it much faster of AblationCAM. " "The parameter controls how many channels should be ablated",
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    if args.method.lower() not in (GRAD_FREE_METHOD_MAP.keys() | GRAD_BASE_METHOD_MAP.keys()):
        raise ValueError(f"invalid CAM type {args.method}," f' supports {", ".join(ALL_METHODS)}.')

    return args


def init_model_cam(args, cfg):
    model = DetCAMModel(cfg, args.checkpoint, args.score_thr, device=args.device)
    if args.preview_model:
        print(model.detector)
        print("\n Please remove `--preview-model` to get the CAM.")
        return

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f"model.detector.{target_layer}"))
        except Exception as e:
            print(model.detector)
            raise RuntimeError("layer does not exist", e)

    extra_params = {"batch_size": args.batch_size, "ablation_layer": DetAblationLayer(), "ratio_channels_to_ablate": args.ratio_channels_to_ablate}

    if args.method in GRAD_BASE_METHOD_MAP:
        method_class = GRAD_BASE_METHOD_MAP[args.method]
        is_need_grad = True
        assert args.no_norm_in_bbox is False, "If not norm in bbox, the " "visualization result " "may not be reasonable."
    else:
        method_class = GRAD_FREE_METHOD_MAP[args.method]
        is_need_grad = False

    max_shape = args.max_shape
    if not isinstance(max_shape, list):
        max_shape = [args.max_shape]
    assert len(max_shape) == 1 or len(max_shape) == 2
    det_cam_visualizer = DetCAMVisualizer(
        method_class,
        model,
        target_layers,
        reshape_transform=partial(reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
        is_need_grad=is_need_grad,
        extra_params=extra_params,
    )
    return model, det_cam_visualizer


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_model(cfg.model,
                        word_emb=datasets[0].word_emb,
                        num_token=datasets[0].num_token)
    
    datasets = list(map(build_dataset, datasets_cfgs))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets[1:]))
    
    model = model.cuda()

    images = args.img
    text = args.text
    if not isinstance(images, list):
        images = [images]
    for image_path in images:
        image = cv2.imread(image_path)
        model.set_input_data(image_path, text)
        predictions = model()[0]

        bboxes = predictions["bboxes"]
        labels = predictions["labels"]
        assert bboxes is not None and len(bboxes) > 0
        targets = [DetBoxScoreTarget(bboxes=bboxes, labels=labels)]
        
        # attn_map = get_local.cache['EncoderLayer.forward'][-1]
        decoder_atten_weight_list = []
        for tmp_weight in get_local.cache['MultiheadAttention.forward']:
            if tmp_weight.shape[-1] == 400:
                decoder_atten_weight_list.append(tmp_weight)
        decoder_atten_weight_list = decoder_atten_weight_list[-3:]

        for idx, decoder_attn_map in enumerate(decoder_atten_weight_list):
            decoder_attn_map = decoder_attn_map.reshape(1, 1, 20, 20)
            decoder_attn_map_min = decoder_attn_map.min()
            decoder_attn_map_max = decoder_attn_map.max()
            decoder_attn_map = (decoder_attn_map - decoder_attn_map_min) / (decoder_attn_map_max - decoder_attn_map_min)
            decoder_attn_map_ = torch.nn.functional.interpolate(torch.from_numpy(decoder_attn_map), image.shape[:-1], mode="bilinear")[
                0, 0]
            image_with_bounding_boxes = det_cam_visualizer.show_cam(image, bboxes, labels, decoder_attn_map_.numpy(), False)
            if args.out_dir:
                mmcv.mkdir_or_exist(args.out_dir)
                # out_file = os.path.join(args.out_dir, os.path.basename(image_path))
                out_file = './vis_cross/decoder-layer-' + str(idx) + '-' + os.path.basename(image_path)
                mmcv.imwrite(image_with_bounding_boxes, out_file)


        # encoder
        # for token in range(1):
        #     for head in range(1):
        #         for layer in range(12, 13):
        #             attn_map = get_local.cache['EncoderLayer.forward'][layer][head, 0, -1 * token, 1 : -20].reshape(1, 1, 20, 20)
        #             attn_map_min = attn_map.min()
        #             attn_map_max = attn_map.max()
        #             attn_map = (attn_map - attn_map_min) / (attn_map_max - attn_map_min)
        #             attn_map_ = torch.nn.functional.interpolate(torch.from_numpy(attn_map), image.shape[:-1], mode="bilinear")[0,0]
        #             image_with_bounding_boxes = det_cam_visualizer.show_cam(image, bboxes, labels, attn_map_.numpy(), False)
        #             if args.out_dir:
        #                 mmcv.mkdir_or_exist(args.out_dir)
        #                 # out_file = os.path.join(args.out_dir, os.path.basename(image_path))
        #                 out_file = './vis_cross/layer-'+str(layer) + '-token-' + str(token) + '-head-' + str(head) + '-'+ os.path.basename(image_path)
        #                 mmcv.imwrite(image_with_bounding_boxes, out_file)
        #             else:
        #                 cv2.namedWindow(os.path.basename(image_path), 0)
        #                 cv2.imshow(os.path.basename(image_path), image_with_bounding_boxes)
        #                 cv2.waitKey(0)



if __name__ == "__main__":
    main()
