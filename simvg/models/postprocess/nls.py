import torch

def hardnls(pred_seg,pred_box):
    mask=torch.zeros_like(pred_seg)
    pred_box = pred_box[:, :4].long()
    for i in range(pred_seg.size()[0]):
        mask[i,pred_box[i][1]:pred_box[i][3]+1,pred_box[i][0]:pred_box[i][2]+1]=1.
    return pred_seg*mask

def asnls(pred_seg,pred_box,weight_score=None,lamb_au=-1.,lamb_bu=2,lamb_ad=1.,lamb_bd=0):
    assert weight_score is not None
    #asnls
    mask = torch.ones_like(pred_seg)*weight_score.unsqueeze(1).unsqueeze(1)*lamb_ad+lamb_bd
    pred_box=pred_box[:,:4].long()
    for i in range(pred_seg.size()[0]):
        mask[i,pred_box[i,1]:pred_box[i,3] + 1, pred_box[i,0]:pred_box[i,2] + 1, ...]=weight_score[i].item()*lamb_au+lamb_bu
    return pred_seg*mask