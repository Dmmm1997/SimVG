from simvg.models import VIS_ENCODERS
from simvg.utils import get_root_logger, is_main
from simvg.models.utils import freeze_params
import timm
from torch import nn
from mmcv.runner import force_fp32

@VIS_ENCODERS.register_module()
class VIT(nn.Module):
    def __init__(self,
                 freeze_layer=-1,
                 model_name="vit_small_patch16_384",
                 img_size=(640,640),
                 pretrained=True,
                 dynamic_img_size=False):
        super(VIT, self).__init__()
        self.vit = timm.create_model(model_name, img_size=img_size, pretrained=pretrained, dynamic_img_size=dynamic_img_size)
        if freeze_layer>=0:
            self.frozen_stages = freeze_layer if freeze_layer<=len(self.vit.blocks) else len(self.vit.blocks)
            self._freeze_stages()
        
    def _freeze_stages(self):
        if self.frozen_stages>=0:
            for i in range(1, self.frozen_stages + 1):
                m = self.vit.blocks[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    @force_fp32(apply_to=('img', ))
    def forward(self, img, y):
        h,w = img.shape[-2:]
        res = self.vit.forward_features(img)[:,1:,:].transpose(1,2).reshape(img.shape[0],-1,h//32, w//32)
        return res