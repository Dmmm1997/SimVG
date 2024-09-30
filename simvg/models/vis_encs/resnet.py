from simvg.models import VIS_ENCODERS
from simvg.utils import get_root_logger, is_main
from simvg.models.utils import freeze_params
import timm
from torch import nn
from mmcv.runner import force_fp32

@VIS_ENCODERS.register_module()
class ResNet(nn.Module):
    def __init__(self,
                 freeze_layer=None,
                 model_name="resnet50",
                 pretrained=True,
                 out_stage=(2,3,4)):
        super(ResNet, self).__init__()
        self.fp16_enabled = False
        assert isinstance(out_stage, tuple)
        self.out_stage = out_stage

        self.resnet = timm.create_model(model_name, features_only=True, out_indices=out_stage, pretrained=True)

        self.do_train = False
        if freeze_layer is not None:
            freeze_params(self.resnet[:-freeze_layer])
        else:
            self.do_train = True

    @force_fp32(apply_to=('img', ))
    def forward(self, img):
        res = self.resnet(img)
        return res