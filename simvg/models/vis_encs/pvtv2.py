from simvg.models import VIS_ENCODERS
from simvg.utils import get_root_logger, is_main
from simvg.models.utils import freeze_params
import timm
from torch import nn
from mmcv.runner import force_fp32

@VIS_ENCODERS.register_module()
class PyramidVisionTransformerV2(nn.Module):
    def __init__(self,
                 freeze_layer=None,
                 model_name="pvt_v2_b2",
                 pretrained=True,
                 out_stage=(1,2,3)):
        super(PyramidVisionTransformerV2, self).__init__()
        self.fp16_enabled = False
        assert isinstance(out_stage, tuple)
        self.out_stage = out_stage

        self.pvtv2 = timm.create_model(model_name, pretrained=True)

        self.do_train = False
        if freeze_layer is not None:
            freeze_params(self.swintransformer[:-freeze_layer])
        else:
            self.do_train = True

    @force_fp32(apply_to=('img', ))
    def forward(self, img, y):
        res = []
        x = self.pvtv2.patch_embed(img)
        for i in range(len(self.pvtv2.stages)):
            x = self.pvtv2.stages[i](x)
            if i in self.out_stage:
                res.append(x)
        return res