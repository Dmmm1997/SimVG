import json
import numpy
from .utils import tokenize
from .builder import DATASETS
from .pipelines import Compose
from pydantic import ListMinLengthError
from torch.utils.data.dataset import Dataset
from simvg.utils import get_root_logger, is_main
from .vgtr_utils.word_utils import Corpus
import torch
import os


class BaseDataset(Dataset):
    def __init__(self,
                 imgsfile,
                 annsfile,
                 pipeline,
                 which_set='train',
                 img_source=['coco'],
                 word_emb_cfg=None):
        super(BaseDataset, self).__init__()
        assert isinstance(which_set, str) and which_set in [
            'train', 'val', 'testA', 'testB', 'test',
            'val_refcoco_unc', 'val_refcocoplus_unc', 'val_refcocog_umd',
            'val_flickr30k', 'val_referitgame_berkeley']
        self.which_set = which_set
        if len(img_source) == 1:
            assert img_source[0] in ['coco', 'visual-genome', 'flickr', 'saiaprtc12']
            self.imgsfile = imgsfile
        elif len(img_source) > 1:
            assert len(imgsfile) == len(img_source)
            assert isinstance(imgsfile, dict)
            self.imgsfile = imgsfile
        else:
            raise ListMinLengthError

        self.anns_all = json.load(open(annsfile, 'r'))

        self.token2idx, self.idx2token, self.word_emb = tokenize(annsfile,
                                                                 self.anns_all,
                                                                 word_emb_cfg)
        
        if self.anns_all["train"][0].get("data_source", None) is not None:
            self.anns_all["train"] = [ann for ann in self.anns_all["train"] if ann['data_source'] in img_source]
        
        if which_set == 'train':
            self._set_group_flag()
        self.pipeline = Compose(pipeline)
        
        if self.pipeline.transforms[0].use_token_type=="copus":
            self.num_token = len(self.pipeline.transforms[0].copus)
        elif self.pipeline.transforms[0].use_token_type=="bert":
            self.num_token = self.pipeline.transforms[0].tokenizer.vocab_size
        else:
            self.num_token = len(self.token2idx)

    def _set_group_flag(self):
        self.flag = numpy.zeros(len(self), dtype=numpy.uint8)
        for i in range(len(self)):
            ann = self.anns_all[self.which_set][i]
            if ann['width'] / ann['height'] > 1:
                self.flag[i] = 1

    def __getitem__(self, index):
        results = {'ann': self.anns_all[self.which_set][index],
                   'which_set': self.which_set,
                   'token2idx': self.token2idx,
                   'imgsfile': self.imgsfile}

        results = self.pipeline(results)

        return results

    def __len__(self):
        return len(self.anns_all[self.which_set])


@DATASETS.register_module()
class GRefCOCO(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(GRefCOCO, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'GRefCOCO-{which_set} size: {len(self)}')
            

@DATASETS.register_module()
class RefCOCOUNC(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOUNC, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOUNC-{which_set} size: {len(self)}')


@DATASETS.register_module()
class RefCOCOGoogle(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOGoogle, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOGoogle-{which_set} size: {len(self)}')


@DATASETS.register_module()
class RefCOCOgUMD(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOgUMD, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOg-{which_set} size: {len(self)}')


@DATASETS.register_module()
class RefCOCOgGoogle(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOgGoogle, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOg-{which_set} size: {len(self)}')


@DATASETS.register_module()
class RefCOCOPlusUNC(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOPlusUNC, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOPlusUNC-{which_set} size: {len(self)}')


@DATASETS.register_module()
class ReferItGameBerkeley(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(ReferItGameBerkeley, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'ReferItGameBerkeley-{which_set} size: {len(self)}')


@DATASETS.register_module()
class Flickr30k(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(Flickr30k, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'Flick30k-{which_set} size: {len(self)}')


@DATASETS.register_module()
class Mixed(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(Mixed, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'Mixed-{which_set} size: {len(self)}')
            logger.info(f'Mixed tokens: {len(self.token2idx)}')
