from simvg.models import MODELS, build_vis_enc, build_lan_enc, build_fusion, build_head
from .base import BaseModel
from ..lan_encs import LSTM


@MODELS.register_module()
class OneStageModel(BaseModel):
    def __init__(self, word_emb, num_token, vis_enc, lan_enc, head, fusion):
        super(OneStageModel, self).__init__()
        self.vis_enc = build_vis_enc(vis_enc)
        if lan_enc is not None:
            self.lan_enc = build_lan_enc(
                lan_enc, {"word_emb": word_emb, "num_token": num_token}
            )
        if head is not None:
            self.head = build_head(head)
        if fusion is not None:
            self.fusion = build_fusion(fusion)

    def extract_visual_language(self, img, img_metas, ref_expr_inds):
        if isinstance(self.lan_enc, LSTM):
            y = self.lan_enc(ref_expr_inds)
        else:
            y = self.lan_enc(img_metas)
        x = self.vis_enc(img, y)
        return x, y
