# -*- coding: utf-8 -*-

import os
import cv2
import random
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def multiclass_metrics(pred, gt):
  """
  check precision and recall for predictions.
  Output: overall = {precision, recall, f1}
  """
  eps=1e-6
  overall = {'precision': -1, 'recall': -1, 'f1': -1}
  NP, NR, NC = 0, 0, 0  # num of pred, num of recall, num of correct
  for ii in range(pred.shape[0]):
    pred_ind = np.array(pred[ii]>0.5, dtype=int)
    gt_ind = np.array(gt[ii]>0.5, dtype=int)
    inter = pred_ind * gt_ind
    # add to overall
    NC += np.sum(inter)
    NP += np.sum(pred_ind)
    NR += np.sum(gt_ind)
  if NP > 0:
    overall['precision'] = float(NC)/NP
  if NR > 0:
    overall['recall'] = float(NC)/NR
  if NP > 0 and NR > 0:
    overall['f1'] = 2*overall['precision']*overall['recall']/(overall['precision']+overall['recall']+eps)
  return overall


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def save_segmentation_map(iou, phrase, bbox, target_bbox, input, mode, batch_start_index, \
                          merge_pred=None, pred_conf_visu=None, save_path='./visulizations_refcoco/'):
    n = input.shape[0]
    save_path=save_path+mode

    input=input.data.cpu().numpy()
    input=input.transpose(0,2,3,1)
    save_txt_path = save_path + 'phrase'
    for ii in range(n):
        os.system('mkdir -p %s/'%(save_path))
        os.system('mkdir -p %s/' % (save_txt_path))
        imgs = input[ii,:,:,:].copy()
        org_imgs = input[ii,:,:,:].copy()
        imgs = (imgs*np.array([0.299, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255.
        # imgs = imgs.transpose(2,0,1)
        imgs = np.array(imgs, dtype=np.float32)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        org_imgs = (org_imgs*np.array([0.299, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255.
        org_imgs = np.array(org_imgs, dtype=np.float32)
        org_imgs = cv2.cvtColor(org_imgs, cv2.COLOR_RGB2BGR)

        cv2.rectangle(imgs, (bbox[ii,0], bbox[ii,1]), (bbox[ii,2], bbox[ii,3]), (255,0,0), 4)
        cv2.rectangle(imgs, (target_bbox[ii,0], target_bbox[ii,1]),
                      (target_bbox[ii,2], target_bbox[ii,3]), (0,255,0), 4)

        cv2.imwrite('%s/pred_gt_%s.png'%(save_path,batch_start_index+ii),imgs)
        cv2.imwrite('%s/pred_gt_org_%s.png' % (save_path, batch_start_index + ii), org_imgs)

        with open(os.path.join(save_txt_path, 'phrase_' + str(batch_start_index+ii) + '.txt'), 'w') as f:
            f.write(phrase[ii])
            f.write('\n')
            f.write(str(iou[ii]))
            f.write('\n')
            pred1 = str(bbox[ii,0]) + ',' + str(bbox[ii,1]) + ',' + str(bbox[ii,2]) + ',' + str(bbox[ii,3])
            f.write(pred1)
            f.write('\n')
            gt = str(target_bbox[ii, 0]) + ',' + \
                 str(target_bbox[ii, 1]) + ',' + \
                 str(target_bbox[ii, 2]) + ',' + str(target_bbox[ii, 3])
            f.write(gt)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, epoch, lr, drop_list=(70, 100)):
    if epoch in drop_list:
        lr = lr * 0.1
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 0.1
    else:
        return


def save_checkpoint(args, state, is_best, epoch, filename='default'):
    if filename == 'default':
        filename = f'model_{args.dataset}_batch_{args.batch_size}'

    model_path = f'{args.savepath}/model/{filename}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # checkpoint_name = f'{model_path}/model_{args.dataset}_Epoch_{epoch}_checkpoint.pth.tar'
    checkpoint_name = f'{model_path}/model_{args.dataset}_checkpoint.pth.tar'
    best_name = f'{model_path}/model_{args.dataset}_best.pth.tar'
    torch.save(state, checkpoint_name)
    # pass
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)



def get_optimizer(args, model):
    # optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        visu_param = model.module.visual_encoder.cnn.parameters()   # set lr=1e-5 for CNN backbone
        rest_param = [param for param in model.parameters() if param not in visu_param]
        visu_param = list(model.module.visual_encoder.cnn.parameters())
        optimizer = torch.optim.SGD([{'params': rest_param},
                                     {'params': visu_param, 'lr': args.lr_backbone}],
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)

    elif args.optimizer == 'adamW':
        if "vit" in args.backbone:
            visu_param = model.module.visual_encoder.parameters()  # set lr=1e-5 for CNN backbone
        else:
            visu_param = model.module.visual_encoder.cnn.parameters()
        rest_param = [param for param in model.parameters() if param not in visu_param]
        visu_param = list(model.module.visual_encoder.parameters())
        visu_param = [p for p in visu_param if p.requires_grad]
        # visu_param = list(visu_param)
        optimizer = torch.optim.AdamW([{'params': rest_param},
                                       {'params': visu_param, 'lr': args.lr_backbone}],
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)

    elif args.optimizer == 'RMSprop':
        visu_param = model.module.visual_encoder.cnn.parameters()  # set lr=1e-5 for CNN backbone
        rest_param = [param for param in model.parameters() if param not in visu_param]
        visu_param = list(model.module.visual_encoder.cnn.parameters())
        optimizer = torch.optim.RMSprop([{'params': rest_param},
                                         {'params': visu_param, 'lr': args.lr_backbone}],
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)

    else:
        raise NotImplementedError('Not Implemented Optimizer')

    return optimizer