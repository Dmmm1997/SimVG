import numpy as np
import torch
import cv2
import random
import torch.nn.functional as F

from mmdet.core.utils import select_single_mlvl
from mmdet.core import (bbox2roi, multiclass_nms)

class GradCAM_BeiT(object):
    """
    Grad CAM for RetinaNet in mmdetection framework

    查清楚bbox_head.get_bboxes这里

    simple_test
        simple_test_bboxes
            self.forward()
            self.get_bboxes
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        # output = output[0][:, 1 : -20].transpose(-1,-2).reshape(1, -1, 16, 16)
        output = output.reshape(1,-1,16,16)
        self.feature = output
        
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        :return:
        """
        # output_grad = output_grad[0][:, 1 : -20].transpose(-1,-2).reshape(1, -1, 16, 16)
        output_grad = output_grad[0].reshape(1,-1,16,16)
        self.gradient = output_grad
        

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Rewriten version
        """
        assert len(cls_scores) == len(bbox_preds)
        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.net.bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=False)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=False)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id, detach=False)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self.net.bbox_head._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def __call__(self, 
                 img,
                    ref_expr_inds,
                    img_metas,
                    text_attention_mask=None,
                    gt_bbox=None,
                    with_bbox=False,
                    with_mask=False,
                    index=0):
        """
        :param image: cv2 format, single image
        :param index: Which bounding box
        :return:
        """
        self.net.zero_grad()
        # Important
        # feat = self.net.extract_feat(data['img'][0].cuda())
        B, _, H, W = img.shape
        img_feat, text_feat, cls_feat = self.net.extract_visual_language(img, ref_expr_inds, text_attention_mask)
        img_feat = img_feat.transpose(-1, -2).reshape(B, -1, H // 32, W // 32)
        output = self.net.head.forward_test(img_feat, img_metas, text_feat=text_feat, cls_feat = cls_feat,  with_bbox=with_bbox, with_mask=with_mask)
        
        output_token_branch_logits = output["token_branch_output"]["pred_logits"]
        output_token_branch_boxes = output["token_branch_output"]["pred_boxes"]
        output_decoder_branch_logits = output["decoder_branch_output"]["pred_logits"]
        output_decoder_branch_boxes = output["decoder_branch_output"]["pred_boxes"]
        
        score = output_decoder_branch_logits[0][index][0]
        bbox = output_decoder_branch_boxes[0][index][0]
        bbox.backward()

        gradient = self.gradient[0] # [C,H,W]
        weight = torch.mean(gradient, axis=(1, 2))  # [C]
        feature = self.feature[0]  # [C,H,W]

        # print(gradient.shape, weight.shape, feature.shape)
        
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = torch.sum(cam, axis=0)  # [H,W]
        cam = torch.relu(cam)  # ReLU

        # Normalization
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        # resize to 224*224
        
        predictions_token_branch = self.net.get_predictions(output["token_branch_output"], img_metas, rescale=True)
        predictions_decoder_branch = self.net.get_predictions(output["decoder_branch_output"], img_metas, rescale=True)
        box = predictions_decoder_branch["pred_bboxes"][index].cpu().detach().numpy().astype(np.int32)
        
        class_id = 0
        return cam.cpu().detach().numpy(), box, class_id, score.cpu().detach().numpy()


def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
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
    # heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb

    # merge heatmap to original image
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap

def draw_label_type(draw_img,bbox,label, line = 5,label_color=None):
    if label_color == None:
        label_color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]

    # label = str(bbox[-1])
    labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img,
                      bbox[:2],
                      bbox[2:],
                      color=label_color,
                      thickness=line)
    else:
        cv2.rectangle(draw_img,
                      bbox[:2],
                      bbox[2:],
                      color=label_color,
                      thickness=line)