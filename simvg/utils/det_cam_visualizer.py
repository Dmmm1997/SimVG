import bisect
import copy

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from simvg.datasets import extract_data

try:
    from pytorch_grad_cam import AblationCAM, AblationLayer, ActivationsAndGradients
    from pytorch_grad_cam.base_cam import BaseCAM
    from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
    from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install ' "3rd party package pytorch_grad_cam.")

# from mmdet.core import get_classes
# from mmdet.datasets import replace_ImageToTensor
from simvg.datasets.pipelines import Compose
from simvg.models import build_model


def reshape_transform(feats, max_shape=(20, 20), is_need_grad=False):
    """Reshape and aggregate feature maps when the input is a multi-layer
    feature map.
    Takes these tensors with different sizes, resizes them to a common shape,
    and concatenates them.
    """
    if len(max_shape) == 1:
        max_shape = max_shape * 2
    feats = feats[:, 1 : -20].transpose(-1,-2).reshape(1, -1, 20, 20)   ###  这里的reshape的数值之前的那个可能不太对
    # feat = feats[0][:, 1 : -20].transpose(-1,-2)
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import os
    # os.makedirs('./vis_heat', exist_ok=True)
    #
    # plt.figure(figsize=(4, 4))
    # sns.heatmap(feats[0,1].cpu().detach().numpy(), cmap='viridis')
    # plt.title('heat')
    # plt.savefig('./vis_heat/1.png')
    # plt.close()
    # import pdb
    # pdb.set_trace()

    if isinstance(feats, torch.Tensor):
        feats = [feats]
    else:
        if is_need_grad:
            raise NotImplementedError("The `grad_base` method does not " "support output multi-activation layers")

    max_h = max([im.shape[-2] for im in feats])
    max_w = max([im.shape[-1] for im in feats])
    if -1 in max_shape:
        max_shape = (max_h, max_w)
    else:
        max_shape = (min(max_h, max_shape[0]), min(max_w, max_shape[1]))

    activations = []
    for feat in feats:
        activations.append(torch.nn.functional.interpolate(torch.abs(feat), max_shape, mode="bilinear"))

    activations = torch.cat(activations, axis=1)
    return activations


class DetCAMModel(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""

    def __init__(self, cfg, checkpoint, score_thr, device="cuda:0"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.score_thr = score_thr
        self.checkpoint = checkpoint
        self.detector = self.build_detector()

        self.return_loss = False
        self.input_data = None
        self.img = None

    def build_detector(self):
        cfg = copy.deepcopy(self.cfg)

        detector = build_model(cfg.model)

        if self.checkpoint is not None:
            load_checkpoint(detector, self.checkpoint, map_location="cpu")
            detector.CLASSES = ["object"]
            # if "CLASSES" in checkpoint.get("meta", {}):
            #     detector.CLASSES = checkpoint["meta"]["CLASSES"]
            # else:
            #     import warnings
            #     warnings.simplefilter("once")
            #     warnings.warn("Class names are not saved in the checkpoint's " "meta data, use COCO classes by default.")
            #     detector.CLASSES = get_classes("coco")

        detector.to(self.device)
        detector.eval()
        return detector

    def set_return_loss(self, return_loss):
        self.return_loss = return_loss

    def set_input_data(self, img, text, bboxes=None, labels=None):
        self.img = img
        cfg = copy.deepcopy(self.cfg)
        if self.return_loss:
            assert bboxes is not None
            assert labels is not None
            cfg.data.val.pipeline[0].type = "LoadFromRawSource"
            test_pipeline = Compose(cfg.data.val.pipeline)
            result = {}
            ann = {}
            ann["bbox"] = bboxes
            ann["category_id"] = 0
            ann["expressions"] = [text]
            result["ann"] = ann
            result["which_set"] = "val"
            result["filepath"] = img

            data = test_pipeline(result)
            data = collate([data], samples_per_gpu=1)
            data = extract_data(data)

            # just get the actual data from DataContainer
            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]
                
        else:
            # set loading pipeline type
            # cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
            cfg.data.val.pipeline[0].type = "LoadFromRawSource"
            test_pipeline = Compose(cfg.data.val.pipeline)
            result = {}
            ann = {}
            ann["bbox"] = [0,0,0,0]
            ann["category_id"] = 0
            ann["expressions"] = [text]
            result["ann"] = ann
            result["which_set"] = "val"
            result["filepath"] = img

            data = test_pipeline(result)
            data = collate([data], samples_per_gpu=1)
            data = extract_data(data)

            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]
            else:
                for m in self.detector.modules():
                    assert not isinstance(m, RoIPool), "CPU inference with RoIPool is not supported currently."

        self.input_data = data

    def __call__(self, *args, **kwargs):
        if self.return_loss:
            loss = self.detector(return_loss=True, **self.input_data)[0]
            return [loss]
        else:
            with torch.no_grad():
                self.input_data.pop("gt_bbox")
                results = self.detector(
                    return_loss=False, rescale=True, **self.input_data)[0]
                bboxes = results["pred_bboxes"]
                labels = results["predict_classes"]
                
                return [{'bboxes': bboxes.cpu().detach().numpy(), 'labels': labels.cpu().detach().numpy(), 'segms': None}]

class DetAblationLayer(AblationLayer):

    def __init__(self):
        super(DetAblationLayer, self).__init__()
        self.activations = None

    def set_next_batch(self, input_batch_index, activations, num_channels_to_ablate):
        """Extract the next batch member from activations, and repeat it
        num_channels_to_ablate times."""
        if isinstance(activations, torch.Tensor):
            return super(DetAblationLayer, self).set_next_batch(input_batch_index, activations, num_channels_to_ablate)

        self.activations = []
        for activation in activations:
            activation = activation[input_batch_index, :, :, :].clone().unsqueeze(0)
            self.activations.append(activation.repeat(num_channels_to_ablate, 1, 1, 1))

    def __call__(self, x):
        """Go over the activation indices to be ablated, stored in
        self.indices.
        Map between every activation index to the tensor in the Ordered Dict
        from the FPN layer.
        """
        result = self.activations

        if isinstance(result, torch.Tensor):
            return super(DetAblationLayer, self).__call__(x)

        channel_cumsum = np.cumsum([r.shape[1] for r in result])
        num_channels_to_ablate = result[0].size(0)  # batch
        for i in range(num_channels_to_ablate):
            pyramid_layer = bisect.bisect_right(channel_cumsum, self.indices[i])
            if pyramid_layer > 0:
                index_in_pyramid_layer = self.indices[i] - channel_cumsum[pyramid_layer - 1]
            else:
                index_in_pyramid_layer = self.indices[i]
            result[pyramid_layer][i, index_in_pyramid_layer, :, :] = -1000
        return result


class DetCAMVisualizer:
    """mmdet cam visualization class.
    Args:
        method:  CAM method. Currently supports
           `ablationcam`,`eigencam` and `featmapam`.
        model (nn.Module): MMDet model.
        target_layers (list[torch.nn.Module]): The target layers
            you want to visualize.
        reshape_transform (Callable, optional): Function of Reshape
            and aggregate feature maps. Defaults to None.
    """

    def __init__(self, method_class, model, target_layers, reshape_transform=None, is_need_grad=False, extra_params=None):
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.is_need_grad = is_need_grad

        if method_class.__name__ == "AblationCAM":
            batch_size = extra_params.get("batch_size", 1)
            ratio_channels_to_ablate = extra_params.get("ratio_channels_to_ablate", 1.0)
            self.cam = AblationCAM(
                model,
                target_layers,
                use_cuda=True if "cuda" in model.device else False,
                reshape_transform=reshape_transform,
                batch_size=batch_size,
                ablation_layer=extra_params["ablation_layer"],
                ratio_channels_to_ablate=ratio_channels_to_ablate,
            )
        else:
            self.cam = method_class(
                model,
                target_layers,
                use_cuda=True if "cuda" in model.device else False,
                reshape_transform=reshape_transform,
            )
            if self.is_need_grad:
                self.cam.activations_and_grads.release()

        self.classes = model.detector.CLASSES
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def switch_activations_and_grads(self, model):
        self.cam.model = model

        if self.is_need_grad is True:
            self.cam.activations_and_grads = ActivationsAndGradientsBeiT(model, self.target_layers, self.reshape_transform)
            self.is_need_grad = False
        else:
            self.cam.activations_and_grads.release()
            self.is_need_grad = True

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        return self.cam(img, targets, aug_smooth, eigen_smooth)[0, :]

    def show_cam(self, image, boxes, labels, grayscale_cam, with_norm_in_bboxes=False, draw_boxes=True):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        if with_norm_in_bboxes is True:
            boxes = boxes.astype(np.int32)
            renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
            images = []
            for x1, y1, x2, y2 in boxes:
                img = renormalized_cam * 0
                img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
                images.append(img)

            renormalized_cam = np.max(np.float32(images), axis=0)
            renormalized_cam = scale_cam_image(renormalized_cam)
        else:
            renormalized_cam = grayscale_cam

        cam_image_renormalized = show_cam_on_image(image / 255, renormalized_cam, use_rgb=False)
        if draw_boxes:
            cam_image_renormalized = self._draw_boxes(boxes, labels, cam_image_renormalized)
        return cam_image_renormalized

    def _draw_boxes(self, boxes, labels, image):
        for i, box in enumerate(boxes):
            label = labels[i]
            color = self.COLORS[label]
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(image, self.classes[label], (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)
        return image


class DetBoxScoreTarget:
    """For every original detected bounding box specified in "bboxes",
    assign a score on how the current bounding boxes match it,
        1. In Bbox IoU
        2. In the classification score.
        3. In Mask IoU if ``segms`` exist.
    If there is not a large enough overlap, or the category changed,
    assign a score of 0.
    The total score is the sum of all the box scores.
    """

    def __init__(self, bboxes, labels, segms=None, match_iou_thr=0.5, device="cuda:0"):
        assert len(bboxes) == len(labels)
        self.focal_bboxes = torch.from_numpy(bboxes).to(device=device)
        self.focal_labels = labels
        if segms is not None:
            assert len(bboxes) == len(segms)
            self.focal_segms = torch.from_numpy(segms).to(device=device)
        else:
            self.focal_segms = [None] * len(labels)
        self.match_iou_thr = match_iou_thr

        self.device = device

    def __call__(self, results):
        output = torch.tensor([0.0], device=self.device)

        if "loss_total" in results:
            # grad_base_method
            for loss_key, loss_value in results.items():
                if "loss_total" not in loss_key:
                    continue
                if isinstance(loss_value, list):
                    output += sum(loss_value)
                else:
                    output += loss_value
            return output
        else:
            # grad_free_method
            if len(results["bboxes"]) == 0:
                return output

            pred_bboxes = torch.from_numpy(results["bboxes"]).to(self.device)
            pred_labels = results["labels"]
            pred_segms = results["segms"]

            if pred_segms is not None:
                pred_segms = torch.from_numpy(pred_segms).to(self.device)
            import pdb
            pdb.set_trace()
            for focal_box, focal_label, focal_segm in zip(self.focal_bboxes, self.focal_labels, self.focal_segms):
                ious = torchvision.ops.box_iou(focal_box[None], pred_bboxes[..., :4])
                index = ious.argmax()
                if ious[0, index] > self.match_iou_thr and pred_labels[index] == focal_label:
                    # TODO: Adaptive adjustment of weights based on algorithms
                    score = ious[0, index] + pred_bboxes[..., 4][index]
                    output = output + score

                    if focal_segm is not None and pred_segms is not None:
                        segms_score = (focal_segm * pred_segms[index]).sum() / (focal_segm.sum() + pred_segms[index].sum() + 1e-7)
                        output = output + segms_score
            return output


# TODO: Fix RuntimeError: element 0 of tensors does not require grad and
#  does not have a grad_fn.
#  Can be removed once the source code is fixed.
class EigenCAM(BaseCAM):

    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        super(EigenCAM, self).__init__(model, target_layers, use_cuda, reshape_transform, uses_gradients=False)

    def get_cam_image(self, input_tensor, target_layer, target_category, activations, grads, eigen_smooth):
        return get_2d_projection(activations)


class FeatmapAM(EigenCAM):
    """Visualize Feature Maps.
    Visualize the (B,C,H,W) feature map averaged over the channel dimension.
    """

    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        super(FeatmapAM, self).__init__(model, target_layers, use_cuda, reshape_transform)

    def get_cam_image(self, input_tensor, target_layer, target_category, activations, grads, eigen_smooth):
        return np.mean(activations, axis=1)


class ActivationsAndGradientsBeiT:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation[0])
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output[0], "requires_grad") or not output[0].requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output[0].register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
