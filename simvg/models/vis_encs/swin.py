from simvg.models import VIS_ENCODERS
from simvg.utils import get_root_logger, is_main
from simvg.models.utils import freeze_params
import timm
from torch import nn
from mmcv.runner import force_fp32

@VIS_ENCODERS.register_module()
class SwinTransformer(nn.Module):
    def __init__(self,
                 freeze_layer=None,
                 model_name="'swin_base_patch4_window7_224'",
                 pretrained=True,
                 out_stage=(1,2,3)):
        super(SwinTransformer, self).__init__()
        self.fp16_enabled = False
        assert isinstance(out_stage, tuple)
        self.out_stage = out_stage

        self.swintransformer = timm.create_model(model_name, pretrained=True, img_size=(480,640))

        self.do_train = False
        if freeze_layer is not None:
            freeze_params(self.swintransformer[:-freeze_layer])
        else:
            self.do_train = True

    @force_fp32(apply_to=('img', ))
    def forward(self, img, y):
        res = []
        x = self.swintransformer.patch_embed(img)
        for i, mod in self.swintransformer.layers:
            x = mod(x)
            if i in self.out_layer:
                res.append(x)
        return res